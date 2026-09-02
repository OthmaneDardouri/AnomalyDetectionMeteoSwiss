import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm

from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.early_stopping import EarlyStopping
from anom_detect.logging_utils import get_logger, log_results_block
from anom_detect.resnet_features import ResNetPatchFeatures
from anom_detect.trafo_model.transformers_custom import Transformer
from anom_detect.visualization import resolve_sample_count, save_segmentation_comparison

logger = get_logger(__name__)


class TransAEManager:
    """Trains and evaluates a custom Transformer encoder-decoder that
    reconstructs frozen-ResNet50 patch features (rather than raw pixels).
    """

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

        with open(self.config_path) as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config["MODELS_CONFIG"]["trafo_autoencoder"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerAE()
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.model_config["learning_rate"])
        )

        self.transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
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

        logger.info("learning_rate=%s", float(self.model_config["learning_rate"]))

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Load the training dataset, split it, and wrap both halves in DataLoaders."""
        batch_size = int(self.model_config.get("batch_size"))
        validation_split = float(self.model_config.get("validation_split"))
        num_workers = int(self.model_config.get("num_workers"))

        self.train_dataset = MVTecAD2(
            self.product_class, "train", transform=self.transform,
            config_path=self.config_path,
        )

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

    def _train_one_epoch(self, train_loader: DataLoader, epoch: int, num_epochs: int) -> float:
        """Run a single training pass over ``train_loader``. Returns the average loss."""
        self.model.train()
        epoch_loss = 0.0
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training", leave=False
        ):
            inputs = batch["sample"].to(self.device)
            outputs, features = self.model(inputs)
            loss = self.criterion(outputs, features)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _evaluate(self, val_loader: DataLoader, epoch: int, num_epochs: int) -> tuple[float, list]:
        """Run a validation pass. Returns (average loss, per-image anomaly scores)."""
        self.model.eval()
        val_loss = 0.0
        reconstruction_errors = []
        with torch.inference_mode():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False
            ):
                inputs = batch["sample"].to(self.device)
                outputs, features = self.model(inputs)
                loss = self.criterion(outputs, features)

                diff = features - outputs  # (B, C, H, W)
                anomaly_map = torch.linalg.norm(diff, dim=1)  # (B, H, W)
                img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(
                    anomaly_map, (1, 1)
                ).reshape(anomaly_map.shape[0])

                reconstruction_errors.extend(img_anomaly_score.cpu().numpy())
                val_loss += loss.item()
        return val_loss / len(val_loader), reconstruction_errors

    def train(self) -> None:
        """Train the autoencoder, logging metrics to TensorBoard."""
        writer = SummaryWriter(log_dir=self.train_path)

        num_epochs = int(self.model_config.get("num_epochs"))
        batch_size = int(self.model_config.get("batch_size"))
        logger.info("Batch size: %d", batch_size)
        logger.info("Number of epochs: %d", num_epochs)
        logger.info("Validation split: %s", self.model_config.get("validation_split"))
        logger.info("Training path: %s", self.train_path)
        self.model.to(self.device)

        self.train_loader, self.val_loader = self._build_dataloaders()

        logger.info("Starting training...")
        logger.info("%s", summary(self.model, input_size=(batch_size, 3, 256, 256)))
        best_val = float("inf")
        best_epoch = 0
        for epoch in range(num_epochs):
            avg_train_loss = self._train_one_epoch(self.train_loader, epoch, num_epochs)
            self.scheduler.step()
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)

            avg_val_loss, reconstruction_errors = self._evaluate(self.val_loader, epoch, num_epochs)
            mean_rec_error = torch.tensor(reconstruction_errors).mean().item()
            std_rec_error = torch.tensor(reconstruction_errors).std().item()
            logger.info("==========================================")
            logger.info(
                "Epoch: %d/%d || Train | Loss: %.6f || Val | Loss: %.6f | MSE-Mean: %.6f | MSE-Std: %.6f",
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss, mean_rec_error, std_rec_error,
            )
            writer.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
            writer.add_scalar("Reconstruction/Mean", mean_rec_error, epoch + 1)
            writer.add_scalar("Reconstruction/Std", std_rec_error, epoch + 1)

            if mean_rec_error < best_val:
                best_val = mean_rec_error
                best_epoch = epoch + 1
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.train_path, "autoencoder_weights.pth"),
                )
                logger.info(
                    "Best model updated at Epoch %d with MSE-Mean: %.6f", best_epoch, mean_rec_error
                )

            # One series only: a second metric would corrupt the patience counter.
            self.early_stopping.check_early_stop(avg_val_loss)
            if self.early_stopping.stop_training:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Reload the best epoch, or compute_thresh() would threshold the final
        # epoch's weights while test() loads the best epoch's.
        best_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        if os.path.isfile(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            logger.info("Reloaded best checkpoint (epoch %d) for thresholding.", best_epoch)

        writer.close()
        logger.info("Training completed.")

    def test(self) -> dict:
        """Score every test image and log ROC-AUC and threshold accuracy."""
        weights_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        test_scores = []
        test_labels = []

        self.test_dataset = MVTecAD2(
            self.product_class, "test", transform=self.transform,
            config_path=self.config_path,
        )
        batch_size = int(self.model_config.get("batch_size"))
        num_workers = int(self.model_config.get("num_workers"))
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        stats_path = os.path.join(self.train_path, "training_statistics.yaml")
        with open(stats_path) as file:
            stats = yaml.safe_load(file)
        threshold = float(stats["threshold"])

        seg_dir = os.path.join(self.test_path, f"segmentation_{self.product_class}")
        n_to_save = resolve_sample_count(self.num_seg_samples, len(self.test_dataset))
        saved = 0

        for el in test_loader:
            sample = el["sample"].to(self.device)
            # Any folder but 'good' is an anomaly; classic MVTec AD names them
            # per defect type, so a "bad" substring match labels everything normal.
            gt_anomaly = np.array(
                [Path(path).parent.name != "good" for path in el["image_path"]], dtype=int
            )

            with torch.inference_mode():
                reconstructed, features = self.model(sample)

            diff = features - reconstructed  # shape (B, C, H, W)
            anomaly_map = torch.linalg.norm(diff, dim=1)  # (B, H, W)
            img_anomaly_score = (
                torch.nn.functional.adaptive_max_pool2d(anomaly_map, (1, 1))
                .reshape(anomaly_map.shape[0])
                .cpu()
                .numpy()
            )

            # Continuous scores: thresholding here would discard the ranking
            # ROC AUC measures. The threshold is applied separately below.
            test_scores.extend(img_anomaly_score)
            test_labels.extend(gt_anomaly)

            if saved < n_to_save:
                # anomaly_map is at feature-map resolution; upsample to match
                # the pixel-level ground-truth mask.
                upsampled_map = torch.nn.functional.interpolate(
                    anomaly_map.unsqueeze(1), size=sample.shape[-2:], mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                for i in range(sample.shape[0]):
                    if saved >= n_to_save:
                        break
                    true_label = "anomalous" if gt_anomaly[i] else "good"
                    pred_label = "anomalous" if img_anomaly_score[i] > threshold else "good"
                    save_segmentation_comparison(
                        sample[i],
                        el["ht"][i],
                        upsampled_map[i],
                        os.path.join(seg_dir, f"sample_{saved}_comparison.png"),
                        true_label,
                        pred_label,
                    )
                    saved += 1

        if len(set(test_labels)) < 2:
            raise ValueError("ROC AUC requires both normal and anomalous test samples.")
        roc_auc = roc_auc_score(test_labels, test_scores)
        accuracy = float(np.mean((np.asarray(test_scores) > threshold) == test_labels))
        log_results_block(
            logger,
            f"Test results | model: trafo | class: {self.product_class}",
            [
                ("Test images", len(test_labels)),
                ("  normal", int(np.sum(np.asarray(test_labels) == 0))),
                ("  anomalous", int(np.sum(np.asarray(test_labels) == 1))),
                ("Threshold", threshold),
                ("ROC AUC (image)", roc_auc),
                ("Accuracy at threshold", accuracy),
                ("Segmentation maps", f"{saved} saved -> {seg_dir}"),
            ],
        )

        if roc_auc < 0.5:
            logger.warning(
                "ROC AUC is less than 0.5. The model might be worse than random. "
                "Consider redesigning the Autoencoder."
            )

        return {"roc_auc_image": float(roc_auc), "accuracy_at_threshold": accuracy}

    def compute_thresh(self) -> tuple[float, float, float]:
        """Score the training set; returns (mean_error, std_error, mean + 3*std)."""
        self.model.eval()
        anomaly_scores = []

        for el in tqdm(self.train_loader, desc="Processing train dataset"):
            sample = el["sample"].to(self.device)
            with torch.inference_mode():
                reconstructed, features = self.model(sample)

            diff = features - reconstructed  # shape (B, C, H, W)
            anomaly_map = torch.linalg.norm(diff, dim=1)  # (B, H, W)
            img_anomaly_score = torch.nn.functional.adaptive_max_pool2d(
                anomaly_map, (1, 1)
            ).reshape(anomaly_map.shape[0])

            anomaly_scores.extend(img_anomaly_score.cpu().numpy())

        logger.info("Mean Anomaly Score: %s", np.mean(anomaly_scores))
        mean_error = np.mean(anomaly_scores)
        std_error = np.std(anomaly_scores)

        threshold = mean_error + 3 * std_error
        logger.info("Mean Error (μ): %s", mean_error)
        logger.info("Standard Deviation (σ): %s", std_error)
        logger.info("Threshold: %s", threshold)

        plt.hist(
            anomaly_scores,
            bins=30,
            density=True,
            alpha=0.7,
            color="blue",
            label="Training Errors",
        )
        plt.axvline(mean_error, color="green", linestyle="--", label="Mean (μ)")
        plt.axvline(threshold, color="red", linestyle="--", label="Threshold (μ + 3σ)")
        plt.title("Histogram of Training Errors")
        plt.xlabel("Error")
        plt.ylabel("Density")
        plt.legend()
        save_path = os.path.join(self.train_path, "training_errors_histogram.png")
        plt.savefig(save_path)
        plt.close()
        return mean_error, std_error, threshold

    def save_model(self, args, mean_error: float, std_error: float, threshold: float) -> None:
        """Persist weights, model config, CLI args, and training stats to train_path."""
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
            stats = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "threshold": float(threshold),
            }
            yaml.dump(stats, file)
        logger.info("Training statistics saved to %s", stats_save_path)


class TransformerAE(nn.Module):
    """Reconstructs frozen-ResNet50 patch features with a Transformer
    encoder-decoder, using a learned query embedding as the decoder target."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ResNetPatchFeatures()
        patch_size = 32
        self.d_model = 512
        self.transformer = Transformer(d_model=self.d_model, num_heads=8, num_layers=4)
        self.query_embed = nn.Parameter(torch.randn(patch_size * patch_size, self.d_model))
        self.tokenizer = nn.Conv2d(1536, self.d_model, kernel_size=1)
        self.proj = nn.Conv2d(self.d_model, 1536, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.backbone(x)

        tokenized = self.tokenizer(feature_maps)  # 1536 -> d_model channels
        B, C, H, W = tokenized.shape
        tokens = tokenized.reshape(B, C, H * W).permute(0, 2, 1)  # (B, seq, d_model)
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)

        src = tokens * math.sqrt(self.d_model)
        tgt = queries * math.sqrt(self.d_model)
        transformed = self.transformer(src, tgt)

        transformed = transformed.permute(0, 2, 1).reshape(B, self.d_model, H, W)
        return self.proj(transformed), feature_maps


