import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.logging_utils import get_logger, log_results_block
from anom_detect.visualization import resolve_sample_count, save_segmentation_comparison

logger = get_logger(__name__)


class BaseAEManager:
    """Trains and evaluates a plain convolutional autoencoder on raw pixels.

    Reconstructs 224x224 images with MSE loss (no pretrained backbone); the
    per-image reconstruction error is the anomaly score. The simplest and
    fastest of the five models, used as a baseline.
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
        self.model_config = self.config["MODELS_CONFIG"]["base_autoencoder"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Autoencoder()
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(self.model_config["learning_rate"])
        )

        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )

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
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def _evaluate(self, val_loader: DataLoader, epoch: int, num_epochs: int) -> tuple[float, list]:
        """Run a validation pass. Returns (average loss, per-image reconstruction errors)."""
        self.model.eval()
        val_loss = 0.0
        reconstruction_errors = []
        with torch.inference_mode():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation", leave=False
            ):
                inputs = batch["sample"].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)

                per_image_error = torch.mean((inputs - outputs) ** 2, dim=(1, 2, 3))
                reconstruction_errors.extend(per_image_error.cpu().numpy())

                val_loss += loss.item()
        return val_loss / len(val_loader), reconstruction_errors

    def train(self) -> None:
        """Train the autoencoder with a training and validation phase.

        Splits the training dataset into training/validation subsets, then
        iteratively updates the model weights using the training data while
        monitoring performance on the validation data. Metrics are logged to
        TensorBoard.
        """
        os.makedirs(self.train_path, exist_ok=True)
        writer = SummaryWriter(log_dir=self.train_path)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Total number of parameters in the autoencoder: %d", total_params)

        num_epochs = int(self.model_config.get("num_epochs"))
        logger.info("Batch size: %s", self.model_config.get("batch_size"))
        logger.info("Number of epochs: %d", num_epochs)
        logger.info("Validation split: %s", self.model_config.get("validation_split"))
        logger.info("Training path: %s", self.train_path)

        self.model.to(self.device)
        train_loader, val_loader = self._build_dataloaders()

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            avg_train_loss = self._train_one_epoch(train_loader, epoch, num_epochs)
            writer.add_scalar("Loss/Train", avg_train_loss, epoch + 1)

            avg_val_loss, reconstruction_errors = self._evaluate(val_loader, epoch, num_epochs)
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

        writer.close()
        logger.info("Training completed.")

    def test(self) -> dict:
        """Score every test image and log ROC-AUC and threshold accuracy."""
        weights_path = os.path.join(self.train_path, "autoencoder_weights.pth")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
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
        if not os.path.isfile(stats_path):
            raise FileNotFoundError(f"Training statistics not found: {stats_path}")
        with open(stats_path, encoding="utf-8") as file:
            stats = yaml.safe_load(file) or {}
        if "threshold" not in stats:
            raise ValueError(f"Missing threshold in training statistics: {stats_path}")
        threshold = float(stats["threshold"])

        seg_dir = os.path.join(self.test_path, f"segmentation_{self.product_class}")
        n_to_save = resolve_sample_count(self.num_seg_samples, len(self.test_dataset))
        saved = 0

        for el in test_loader:
            sample = el["sample"].to(self.device)
            gt_anomaly = np.array(
                [Path(path).parent.name != "good" for path in el["image_path"]], dtype=int
            )

            with torch.inference_mode():
                reconstructed = self.model(sample)
            # Keep continuous reconstruction errors for ROC AUC; threshold only
            # for the separately reported operating-point accuracy.
            error = (
                torch.mean((sample - reconstructed) ** 2, dim=(1, 2, 3)).cpu().numpy()
            )
            test_scores.extend(error)
            test_labels.extend(gt_anomaly)

            if saved < n_to_save:
                # Per-pixel squared error, same resolution as the input/mask
                # (no upsampling needed -- this autoencoder reconstructs the
                # full image rather than a downsampled feature map).
                pixel_error_map = torch.mean((sample - reconstructed) ** 2, dim=1)
                for i in range(sample.shape[0]):
                    if saved >= n_to_save:
                        break
                    true_label = "anomalous" if gt_anomaly[i] else "good"
                    pred_label = "anomalous" if error[i] > threshold else "good"
                    save_segmentation_comparison(
                        sample[i],
                        el["ht"][i],
                        pixel_error_map[i],
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
            f"Test results | model: base | class: {self.product_class}",
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

        # Returned so callers -- notebooks, and the MLflow layer in cli.py --
        # get the same number the block above prints. PatchCore and DFR
        # already returned theirs.
        return {"roc_auc_image": float(roc_auc), "accuracy_at_threshold": accuracy}

    def compute_thresh(self) -> tuple[float, float, float]:
        """Score the training set and derive a threshold = mean + 3*std.

        Returns:
            (mean_error, std_error, threshold)
        """
        self.model.to(self.device)
        self.model.eval()
        anomaly_scores = []
        for el in tqdm(self.train_dataset, desc="Processing train dataset"):
            sample = el["sample"].to(self.device)
            with torch.inference_mode():
                reconstructed = self.model(sample)

            squared_difference = (sample - reconstructed) ** 2
            difference_image = (
                torch.mean(squared_difference, dim=0).squeeze(0).cpu().numpy()
            )
            anomaly_score = np.mean(difference_image)
            anomaly_scores.append(anomaly_score)

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
            stats = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "threshold": float(threshold),
            }
            yaml.dump(stats, file)
        logger.info("Training statistics saved to %s", stats_save_path)


class Autoencoder(nn.Module):
    """Simple 3-layer conv encoder / transposed-conv decoder over 224x224 RGB images."""

    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 112, 112)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (B, 128, 56, 56)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (B, 256, 28, 28)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (B, 128, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (B, 64, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (B, 3, 224, 224)
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
