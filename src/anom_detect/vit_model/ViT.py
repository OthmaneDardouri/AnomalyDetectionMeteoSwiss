import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as utils
import yaml
from einops import rearrange
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

import anom_detect.vit_model.mdn1 as mdn1
import anom_detect.vit_model.model_res18 as M
import anom_detect.vit_model.pytorch_ssim as pytorch_ssim
import anom_detect.vit_model.spatial as S
from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.early_stopping import EarlyStopping
from anom_detect.logging_utils import get_logger, log_results_block
from anom_detect.visualization import resolve_sample_count, save_segmentation_comparison
from anom_detect.vit_model.mdn1 import add_noise
from anom_detect.vit_model.student_transformer import ViT
from anom_detect.vit_model.utility_fun import Binarization, Filter

logger = get_logger(__name__)


class ViTManager:
    """Trains and evaluates a ViT encoder -> capsule pooling -> CNN decoder
    autoencoder, with a Mixture Density Network modelling the latent
    distribution for anomaly scoring."""

    def __init__(
        self,
        product_class: str,
        config_path: str,
        train_path: str,
        test_path: str,
        num_seg_samples: Optional[int] = 5,
    ) -> None:
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.product_class = product_class
        self.num_seg_samples = num_seg_samples

        self.ssim_loss = pytorch_ssim.SSIM()
        with open(self.config_path) as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config["MODELS_CONFIG"]["vit_autoencoder"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VT_AE().to(self.device)
        self.G_estimate = mdn1.MDN().to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.G_estimate.parameters()),
            lr=float(self.model_config["learning_rate"]),
            weight_decay=0.0001,
        )
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=int(self.model_config.get("num_epochs")),
            eta_min=float(self.model_config.get("min_lr")),
        )
        patience = int(self.model_config.get("patience"))
        delta = float(self.model_config.get("delta"))
        self.early_stopping = EarlyStopping(
            patience=patience, delta=delta, verbose=False
        )
        self.train_dataset = MVTecAD2(
            self.product_class, "train", transform=self.transform,
            config_path=self.config_path,
        )
        self.test_dataset = MVTecAD2(
            self.product_class, "test", transform=self.transform,
            config_path=self.config_path,
        )
        logger.info("learning_rate=%s", float(self.model_config["learning_rate"]))

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Split the training dataset and build train/validation DataLoaders."""
        batch_size = int(self.model_config.get("batch_size"))
        validation_split = float(self.model_config.get("validation_split"))
        num_workers = int(self.model_config.get("num_workers"))

        if len(self.train_dataset) < 2:
            raise ValueError("At least two training images are required for validation.")
        val_size = max(1, int(validation_split * len(self.train_dataset)))
        train_size = len(self.train_dataset) - val_size
        split_generator = torch.Generator().manual_seed(int(self.config.get("SEED", 42)))
        train_subset, val_subset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size], generator=split_generator
        )
        logger.info(
            "Training on %d samples, validating on %d samples.",
            len(train_subset), len(val_subset),
        )

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        return train_loader, val_loader

    def _compute_losses(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass plus the reconstruction/SSIM/Gaussian loss terms.

        Returns (total_loss, loss1, loss2, loss3, vector) so callers can
        both backprop and log the individual components.
        """
        vector, reconstructions = self.model(inputs)
        pi, mu, sigma = self.G_estimate(vector)

        loss1 = F.mse_loss(reconstructions, inputs, reduction="mean")  # Rec Loss
        loss2 = -self.ssim_loss(inputs, reconstructions)  # SSIM structural-similarity loss
        loss3 = mdn1.mdn_loss_function(vector, mu, sigma, pi)  # MDN Gaussian loss

        loss = 5 * loss1 + 0.5 * loss2 + loss3
        return loss, loss1, loss2, loss3, vector

    def _train_one_epoch(self, train_loader: DataLoader, writer: SummaryWriter, epoch: int, num_epochs: int) -> float:
        """Run a single training pass over ``train_loader``. Returns the average loss."""
        self.model.train()
        self.G_estimate.train()
        epoch_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False
        ):
            self.model.zero_grad()
            inputs = batch["sample"].to(self.device)
            loss, loss1, loss2, loss3, vector = self._compute_losses(inputs)

            writer.add_scalar("TRAIN/recon-loss", loss1.item(), epoch)
            writer.add_scalar("TRAIN/ssim loss", loss2.item(), epoch)
            writer.add_scalar("TRAIN/Gaussian loss", loss3.item(), epoch)
            writer.add_histogram("TRAIN/Vectors", vector, epoch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _evaluate(
        self, val_loader: DataLoader, writer: SummaryWriter, epoch: int, num_epochs: int
    ) -> tuple[float, list, torch.Tensor, torch.Tensor]:
        """Run a validation pass. Returns (avg loss, per-image scores, last inputs, last reconstructions)."""
        self.model.eval()
        self.G_estimate.eval()
        val_loss = 0.0
        reconstruction_errors = []
        inputs = reconstructions = None
        with torch.inference_mode():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False
            ):
                inputs = batch["sample"].to(self.device)
                vector, reconstructions = self.model(inputs)
                pi, mu, sigma = self.G_estimate(vector)

                loss1 = F.mse_loss(reconstructions, inputs, reduction="mean")
                loss2 = -self.ssim_loss(inputs, reconstructions)
                loss3 = mdn1.mdn_loss_function(vector, mu, sigma, pi, test=False)
                loss = 5 * loss1 + 0.5 * loss2 + loss3

                diff = inputs - reconstructions  # shape (B, C, H, W)
                anomaly_map = torch.linalg.norm(diff, dim=1)  # (B, H, W)
                img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(
                    anomaly_map, (1, 1)
                ).reshape(anomaly_map.shape[0])
                reconstruction_errors.extend(img_anomaly_score.cpu().numpy())

                writer.add_scalar("VAL/recon-loss", loss1.item(), epoch)
                writer.add_scalar("VAL/ssim loss", loss2.item(), epoch)
                writer.add_scalar("VAL/Gaussian loss", loss3.item(), epoch)
                writer.add_histogram("VAL/Vectors", vector, epoch)

                val_loss += loss.item()
        return val_loss / len(val_loader), reconstruction_errors, inputs, reconstructions

    def train(self) -> None:
        """Train the autoencoder, logging metrics to TensorBoard."""
        os.makedirs(self.train_path, exist_ok=True)
        writer = SummaryWriter(log_dir=self.train_path)

        num_epochs = int(self.model_config.get("num_epochs"))
        batch_size = int(self.model_config.get("batch_size"))
        logger.info("Batch size: %d", batch_size)
        logger.info("Number of epochs: %d", num_epochs)
        logger.info("Validation split: %s", self.model_config.get("validation_split"))
        logger.info("Training path: %s", self.train_path)

        self.train_loader, self.val_loader = self._build_dataloaders()

        logger.info("Starting training...")
        logger.info("%s", summary(self.model, input_size=(batch_size, 3, 512, 512)))
        best_val = float("inf")
        best_epoch = 0
        for epoch in range(num_epochs):
            avg_train_loss = self._train_one_epoch(self.train_loader, writer, epoch, num_epochs)
            self.scheduler.step()
            writer.add_scalar("TRAIN/Loss", avg_train_loss, epoch + 1)

            avg_val_loss, reconstruction_errors, inputs, reconstructions = self._evaluate(
                self.val_loader, writer, epoch, num_epochs
            )
            self.early_stopping.check_early_stop(avg_val_loss)
            mean_rec_error = torch.tensor(reconstruction_errors).mean().item()
            std_rec_error = torch.tensor(reconstruction_errors).std().item()
            logger.info("==========================================")
            logger.info(
                "Epoch: %d/%d || Train | Loss: %.6f || Val | Loss: %.6f | MSE-Mean: %.6f | MSE-Std: %.6f",
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, mean_rec_error, std_rec_error,
            )
            writer.add_scalar("VAL/Loss", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)
            writer.add_image(
                "Reconstructed Image", utils.make_grid(reconstructions), epoch, dataformats="CHW"
            )
            writer.add_image(
                "Original Image", utils.make_grid(inputs), epoch, dataformats="CHW"
            )

            # Save the best epoch based on validation loss, not the data used to fit.
            if avg_val_loss < best_val:
                best_val = avg_val_loss
                best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.train_path, "vit_weights.pth"),
                )
                torch.save(
                    self.G_estimate.state_dict(),
                    os.path.join(self.train_path, "g_weights.pth"),
                )
                logger.info(
                    "Best model updated at Epoch %d with Val-Loss: %.6f", best_epoch, avg_val_loss
                )
            if self.early_stopping.stop_training:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        writer.close()
        logger.info("Training completed.")

    def test(self, upsample: int = 1) -> tuple[float, float, float]:
        """Score the test set. Returns (PRO score, image-level AUC, AUC-PR)."""
        norm_loss_t = []
        normalised_score_t = []
        mask_score_t = []

        patch_size = 64
        num_workers = int(self.model_config.get("num_workers"))
        test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        )

        vit_weights_path = os.path.join(self.train_path, "vit_weights.pth")
        g_weights_path = os.path.join(self.train_path, "g_weights.pth")

        self.model_test = VT_AE(train=False).to(self.device)
        self.model_test.load_state_dict(
            torch.load(vit_weights_path, map_location=self.device)
        )
        self.model_test.eval()

        self.G_estimate_test = mdn1.MDN().to(self.device)
        self.G_estimate_test.load_state_dict(
            torch.load(g_weights_path, map_location=self.device)
        )
        self.G_estimate_test.eval()
        stats_path = os.path.join(self.train_path, "training_statistics.yaml")
        with open(stats_path) as file:
            stats = yaml.safe_load(file)
        threshold = float(stats["threshold"])

        seg_dir = os.path.join(self.test_path, f"segmentation_{self.product_class}")
        n_to_save = resolve_sample_count(self.num_seg_samples, len(self.test_dataset))
        saved = 0

        t_loss_all_normal = []
        t_loss_all_anomaly = []
        for el in test_loader:
            # Get the input image and move to device. Add a batch dimension.
            sample = el["sample"].to(self.device)
            mask = el["ht"].to(self.device)
            n = int(os.path.basename(os.path.dirname(el["image_path"][0])) != "good")

            vector, reconstructions = self.model_test(sample)
            pi, mu, sigma = self.G_estimate_test(vector)

            # Loss calculations
            loss1 = F.mse_loss(reconstructions, sample, reduction="mean")  # Rec Loss
            loss2 = -self.ssim_loss(
                sample, reconstructions
            )  # SSIM loss for structural similarity
            loss3 = mdn1.mdn_loss_function(
                vector, mu, sigma, pi, test=True
            )  # MDN loss for gaussian approximation
            loss = loss1 - loss2 + loss3.max()  # Total loss
            norm_loss_t.append(loss3.detach().cpu().numpy())

            if n == 0:
                t_loss_all_normal.append(loss.detach().cpu().numpy())
            else:
                t_loss_all_anomaly.append(loss.detach().cpu().numpy())

            if upsample == 0:
                # Mask patch
                mask_patch = rearrange(
                    mask.squeeze(0).squeeze(0),
                    "(h p1) (w p2) -> (h w) p1 p2",
                    p1=patch_size,
                    p2=patch_size,
                )
                mask_patch_score = Binarization(mask_patch.sum(1).sum(1), 0.0)
                mask_score_t.append(mask_patch_score)  # Storing all masks
                norm_score = Binarization(norm_loss_t[-1], threshold)
                normalised_score_t.append(norm_score)  # Storing all patch scores
            elif upsample == 1:
                mask_score_t.append(
                    mask.squeeze(0).squeeze(0).cpu().numpy()
                )  # Storing all masks

                m = torch.nn.UpsamplingBilinear2d((512, 512))
                norm_score = norm_loss_t[-1].reshape(
                    -1, 1, 512 // patch_size, 512 // patch_size
                )
                score_map = m(torch.tensor(norm_score))
                score_map = Filter(score_map)

                normalised_score_t.append(score_map)  # Storing all score maps

                if saved < n_to_save:
                    true_label = "anomalous" if n else "good"
                    pred_label = (
                        "anomalous" if float(np.max(norm_loss_t[-1])) > threshold else "good"
                    )
                    save_segmentation_comparison(
                        sample[0],
                        mask_score_t[-1],
                        normalised_score_t[-1],
                        os.path.join(seg_dir, f"sample_{saved}_comparison.png"),
                        true_label,
                        pred_label,
                    )
                    saved += 1

        ## PRO Score
        scores = np.asarray(normalised_score_t).flatten()
        masks = np.asarray(mask_score_t).flatten()
        masks = (masks > 0.5).astype(int)
        PRO_score = roc_auc_score(masks, scores)

        ## Image Anomaly Classification Score (AUC)
        roc_data = np.concatenate((t_loss_all_normal, t_loss_all_anomaly))
        roc_targets = np.concatenate(
            (np.zeros(len(t_loss_all_normal)), np.ones(len(t_loss_all_anomaly)))
        )
        AUC_Score_total = roc_auc_score(roc_targets, roc_data)

        # AUC Precision Recall Curve
        precision, recall, thres = precision_recall_curve(roc_targets, roc_data)
        AUC_PR = auc(recall, precision)

        log_results_block(
            logger,
            f"Test results | model: vit | class: {self.product_class}",
            [
                ("Test images", len(self.test_dataset)),
                ("  normal", len(t_loss_all_normal)),
                ("  anomalous", len(t_loss_all_anomaly)),
                ("Threshold", threshold),
                ("PRO score (pixel AUC)", PRO_score),
                ("AUC score (image)", AUC_Score_total),
                ("AUC-PR score (image)", AUC_PR),
                ("Segmentation maps", f"{saved} saved -> {seg_dir}"),
            ],
        )
        return PRO_score, AUC_Score_total, AUC_PR

    def training_threshold(self) -> float:
        """Calibrate a pixel-score threshold on held-out normal validation data.

        Never touches test masks or labels. The quantile is ``threshold_quantile``
        in the config, defaulting to 99.7%.
        """
        if not hasattr(self, "val_loader"):
            raise RuntimeError("Train the model before calibrating a threshold.")

        vit_weights_path = os.path.join(self.train_path, "vit_weights.pth")
        g_weights_path = os.path.join(self.train_path, "g_weights.pth")
        if not (os.path.isfile(vit_weights_path) and os.path.isfile(g_weights_path)):
            raise FileNotFoundError("Best ViT checkpoints are required for threshold calibration.")

        self.model.load_state_dict(torch.load(vit_weights_path, map_location=self.device))
        self.G_estimate.load_state_dict(torch.load(g_weights_path, map_location=self.device))
        self.model.eval()
        self.G_estimate.eval()

        scores = []
        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Calibrating threshold"):
                images = batch["sample"].to(self.device)
                vector, _ = self.model(images)
                pi, mu, sigma = self.G_estimate(vector)
                scores.append(mdn1.mdn_loss_function(vector, mu, sigma, pi, test=True).flatten().cpu())

        if not scores:
            raise RuntimeError("Validation loader produced no samples for threshold calibration.")
        quantile = float(self.model_config.get("threshold_quantile", 0.997))
        if not 0 < quantile < 1:
            raise ValueError("threshold_quantile must be between 0 and 1.")
        threshold = torch.quantile(torch.cat(scores), quantile).item()
        logger.info("Validation threshold (%.3f%% quantile): %.6f", quantile * 100, threshold)
        return threshold

    def save_model(self, args, threshold: float) -> None:
        """Persist weights, model config, CLI args, and the calibrated threshold."""
        os.makedirs(self.train_path, exist_ok=True)
        model_save_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        torch.save(self.model.state_dict(), model_save_path)
        logger.info("Model weights saved to %s", model_save_path)

        config_save_path = os.path.join(self.train_path, "config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.model_config, file)
        logger.info("Model configuration saved to %s", config_save_path)

        args_save_path = os.path.join(self.train_path, "args.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(vars(args), file)
        logger.info("Arguments saved to %s", args_save_path)

        stats_save_path = os.path.join(self.train_path, "training_statistics.yaml")
        with open(stats_save_path, "w") as file:
            stats = {"threshold": float(threshold)}
            yaml.dump(stats, file)
        logger.info("Training statistics saved to %s", stats_save_path)


class VT_AE(nn.Module):
    """ViT encoder -> capsule pooling (DigitCaps) -> CNN decoder autoencoder."""

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 64,
        num_classes: int = 1,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        train: bool = True,
    ) -> None:

        super().__init__()
        self.vt = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
        )

        self.decoder = M.decoder2(8)
        self.Digcap = S.DigitCaps(
            in_num_caps=((image_size // patch_size) ** 2) * 8 * 8, in_dim_caps=8
        )
        self.register_buffer(
            "mask", torch.ones(1, image_size // patch_size, image_size // patch_size).bool()
        )
        self.Train = train

        if self.Train:
            logger.info("Initializing network weights.........")
            initialize_weights(self.vt, self.decoder)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        encoded = self.vt(x, self.mask)
        if self.Train:
            encoded = add_noise(encoded)
        encoded1, vectors = self.Digcap(encoded.view(b, encoded.size(1) * 8 * 8, -1))
        recons = self.decoder(encoded1.view(b, -1, 8, 8))

        return encoded, recons


# Initialize weight function
def initialize_weights(*models: nn.Module) -> None:
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
