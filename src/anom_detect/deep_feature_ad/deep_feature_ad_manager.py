import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.deep_feature_ad.deep_feature_anomaly_detector import DeepFeatureAnomalyDetector
from anom_detect.logging_utils import get_logger, log_results_block
from anom_detect.visualization import resolve_sample_count, save_segmentation_comparison

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger = get_logger(__name__)


class DeepFeatureADManager:
    """Trains, calibrates and tests the deep-feature reconstruction (DFR) autoencoder.

    Threshold calibration is its own step, ``compute_threshold()``, called
    after ``train()``. ``product_class == "foundational"`` trains one shared
    set of weights across every class in ``FOUNDATIONAL_OBJECTS``, saving one
    threshold per class.

    Dataset construction is lazy so ``serve.py`` can build this class purely
    to reuse its ``device``/``transform``/``detector`` with no dataset mounted.
    """

    def __init__(
        self,
        product_class: str,
        config_path: str,
        train_path: str,
        test_path: str,
        num_seg_samples: Optional[int] = 5,
    ) -> None:
        self.product_class = product_class
        self.config_path = config_path
        self.train_path = train_path
        self.test_path = test_path
        self.num_seg_samples = num_seg_samples

        with open(self.config_path) as file:
            self.config = yaml.safe_load(file)
        self.model_config = self.config["MODELS_CONFIG"]["DeepFeatureAE"]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("USING DEVICE: %s", self.device)

        self.detector = DeepFeatureAnomalyDetector(
            layer_hooks=self.model_config["layer_hooks"],
            latent_dim=self.model_config["latent_dim"],
            smooth=self.model_config["smooth"],
            is_bn=self.model_config["is_bn"],
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.detector.autoencoder.parameters(), lr=float(self.model_config["learning_rate"])
        )

        self.transform = transforms.Compose([
            transforms.Resize((self.model_config["input_size"], self.model_config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        # Populated by compute_threshold() / load_computed_thresholds().
        self.thresholds = None
        self.sigma_multiplier = None
        self.mean_error = None
        self.std_error = None
        self.anomaly_scores = None

        # Built lazily -- see the class docstring.
        self.train_dataset = self.train_loader = self.val_loader = self.val_subset = None
        self.scheduler = None

    # ------------------------------------------------------------ train() --

    def train(self) -> None:
        """Train on ``product_class``, writing checkpoints under ``train_path``."""
        if self.product_class == "foundational":
            self._train_foundational()
        else:
            self._ensure_single_class_loaders()
            self._train_single_class()

    def _ensure_single_class_loaders(self) -> None:
        if self.train_dataset is not None:
            return
        self.train_dataset, self.train_loader, self.val_loader, self.val_subset = (
            self._build_train_val_loaders(self.product_class)
        )
        self.scheduler = self._build_scheduler(self.train_loader)

    def _build_train_val_loaders(self, product_class: str):
        """Returns ``(train_dataset, train_loader, val_loader, val_subset)``.

        ``train_dataset`` is the full, unsplit dataset -- what
        ``compute_threshold()`` scores.
        """
        train_dataset = MVTecAD2(
            product_class, "train",
            transform=self.transform, config_path=self.config_path,
        )
        if len(train_dataset) < 2:
            raise ValueError("At least two training images are required for validation.")
        val_size = max(1, int(self.model_config["validation_split"] * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        split_generator = torch.Generator().manual_seed(int(self.config.get("SEED", 42)))
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=split_generator
        )
        train_loader = DataLoader(train_subset, batch_size=self.model_config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.model_config["batch_size"], shuffle=False)
        return train_dataset, train_loader, val_loader, val_subset

    def _build_scheduler(self, train_loader: DataLoader):
        return torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            steps_per_epoch=len(train_loader),
            epochs=self.model_config["num_epochs"],
            max_lr=float(self.model_config["learning_rate"]),
            pct_start=0.3,
            anneal_strategy="linear",
        )

    def _train_single_class(self, save_checkpoint: bool = True) -> None:
        self.detector.train()
        stats = self.detector.get_stats()
        logger.info("Number of layers: %d", stats["num_layers"])
        logger.info("Backbone parameters (frozen): %s", f"{stats['backbone_params']:,}")
        logger.info("Autoencoder parameters: %s", f"{stats['autoencoder_params']:,}")
        logger.info("Trainable parameters: %s", f"{stats['trainable_params']:,}")
        logger.info(
            "Training on %d samples, validating on %d samples.",
            len(self.train_dataset), len(self.val_subset),
        )

        best_val_loss = float("inf")
        num_epochs = self.model_config["num_epochs"]
        for epoch in range(num_epochs):
            avg_train_loss = self._train_one_epoch()
            avg_val_loss, best_val_loss = self._evaluate(best_val_loss, save_checkpoint=save_checkpoint)
            logger.info(
                "Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f",
                epoch + 1, num_epochs, avg_train_loss, avg_val_loss,
            )

    def _train_one_epoch(self) -> float:
        self.detector.train()
        train_loss = 0.0
        train_batches = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch - Training ({self.product_class})"):
            images = batch["sample"].to(self.device)

            features, reconstructed = self.detector(images)
            loss = self.criterion(reconstructed, features)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_loss += loss.item()
            train_batches += 1

        return train_loss / train_batches

    def _evaluate(self, best_val_loss: float, save_checkpoint: bool = True) -> tuple[float, float]:
        self.detector.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch - Validation ({self.product_class})"):
                images = batch["sample"].to(self.device)
                features, reconstructed = self.detector(images)
                loss = self.criterion(reconstructed, features)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoint:
                self._save_checkpoint()
        return avg_val_loss, best_val_loss

    def _save_checkpoint(self) -> None:
        model_dir = os.path.join(self.train_path, "checkpoints")
        os.makedirs(model_dir, exist_ok=True)

        weights_path = os.path.join(model_dir, f"{self.product_class}_dfad_weights.pth")
        torch.save(self.detector.state_dict(), weights_path)
        logger.info("Model saved to %s", weights_path)

        config_save_path = os.path.join(model_dir, f"{self.product_class}_dfad_config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.config, file)
        logger.info("Config saved to %s", config_save_path)

    def _train_foundational(self) -> None:
        for product_class in self.config["FOUNDATIONAL_OBJECTS"]:
            logger.info("Training on %s class...", product_class)
            self.product_class = product_class
            self.train_dataset, self.train_loader, self.val_loader, self.val_subset = (
                self._build_train_val_loaders(product_class)
            )
            self.scheduler = self._build_scheduler(self.train_loader)
            # Weights are shared across classes; saved once below.
            self._train_single_class(save_checkpoint=False)

        self.product_class = "foundational"
        self._save_checkpoint()

    # ----------------------------------------------------- compute_threshold() --

    def compute_threshold(self) -> Optional[float]:
        """Calibrate threshold(s) from the *training* score distribution and save them.

        Returns the operating threshold, or ``None`` in the foundational case,
        which writes one file per class instead.
        """
        mode = self.model_config["threshold_computation_mode"]

        if self.product_class == "foundational":
            for product_class in self.config["FOUNDATIONAL_OBJECTS"]:
                logger.info("Computing thresholds for %s class...", product_class)
                train_dataset = MVTecAD2(
                    product_class, "train",
                    transform=self.transform, config_path=self.config_path,
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=self.model_config["batch_size"], shuffle=False
                )
                scores, mean, std, thresholds, sigma_multiplier = self._score_training_set(
                    train_loader, mode
                )
                self._save_thresholds(
                    product_class, scores, mean, std, thresholds, sigma_multiplier, foundational=True
                )
        else:
            self._ensure_single_class_loaders()
            train_loader = DataLoader(
                self.train_dataset, batch_size=self.model_config["batch_size"], shuffle=False
            )
            scores, mean, std, thresholds, sigma_multiplier = self._score_training_set(
                train_loader, mode
            )
            self.anomaly_scores, self.mean_error, self.std_error = scores, mean, std
            self.thresholds, self.sigma_multiplier = thresholds, sigma_multiplier
            self._save_thresholds(
                self.product_class, scores, mean, std, thresholds, sigma_multiplier, foundational=False
            )
            # thresholds[0] is the operating one -- the entry `serve` reads back.
            return float(thresholds[0]) if thresholds else None
        return None

    def _score_training_set(self, train_loader: DataLoader, mode: str):
        if mode == "standard":
            sigma_multiplier = 3.0
        elif mode == "aggressive":
            sigma_multiplier = 1.0
        elif mode == "conservative":
            sigma_multiplier = 5.0
        elif mode == "all":
            sigma_multiplier = [3.0, 1.0, 5.0]
        else:
            raise ValueError(
                f"Invalid threshold computation mode: {mode}. Must be 'standard', "
                "'aggressive', 'conservative', or 'all'"
            )

        self.detector.eval()
        anomaly_scores = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Computing thresholds"):
                images = batch["sample"].to(self.device)
                features, reconstructed = self.detector(images)
                error_map = self.detector.compute_reconstruction_error(features, reconstructed)
                scores = self.detector.compute_anomaly_score(error_map, k=10)
                anomaly_scores.extend(scores.cpu().numpy())

        anomaly_scores = np.array(anomaly_scores)
        mean = float(np.mean(anomaly_scores))
        std = float(np.std(anomaly_scores))
        logger.info("Mean Anomaly Score: %s, Std Anomaly Score: %s", mean, std)

        if isinstance(sigma_multiplier, list):
            thresholds = [mean + m * std for m in sigma_multiplier]
            logger.info("Computed thresholds: %s", thresholds)
        else:
            thresholds = mean + sigma_multiplier * std
            logger.info("Computed threshold: %s", thresholds)

        return anomaly_scores, mean, std, thresholds, sigma_multiplier

    def _save_thresholds(
        self, product_class, anomaly_scores, mean, std, thresholds, sigma_multiplier, foundational: bool
    ) -> None:
        if foundational:
            threshold_file = os.path.join(self.train_path, f"{product_class}_foundational_thresholds.yaml")
        else:
            threshold_file = os.path.join(self.train_path, f"{product_class}_thresholds.yaml")

        thresholds_out = [float(t) for t in thresholds] if isinstance(thresholds, list) else float(thresholds)

        threshold_info = {
            "mode": self.model_config["threshold_computation_mode"],
            "sigma_multiplier": sigma_multiplier,
            "mean_error": float(mean),
            "std_error": float(std),
            "thresholds": thresholds_out,
            "num_samples": len(anomaly_scores),
            "train_scores": [float(score) for score in anomaly_scores],
        }
        with open(threshold_file, "w") as file:
            yaml.safe_dump({"thresholds": threshold_info}, file)
        logger.info("Thresholds saved to: %s", threshold_file)

    def load_computed_thresholds(self, threshold_file: str) -> None:
        with open(threshold_file) as file:
            threshold_info = yaml.safe_load(file)["thresholds"]
        self.sigma_multiplier = threshold_info["sigma_multiplier"]
        self.mean_error = threshold_info["mean_error"]
        self.std_error = threshold_info["std_error"]
        self.thresholds = threshold_info["thresholds"]
        logger.info("Loaded thresholds: %s", self.thresholds)

    # -------------------------------------------------------------- test() --

    def test(self) -> float:
        """Score the test split and return the image-level ROC AUC.

        Loads weights and thresholds from the ``<class>_dfad_weights.pth`` /
        ``<class>_thresholds.yaml`` files a training run writes under
        ``train_path``.
        """
        weights_path = Path(self.train_path) / "checkpoints" / f"{self.product_class}_dfad_weights.pth"
        threshold_path = Path(self.train_path) / f"{self.product_class}_thresholds.yaml"

        if not weights_path.is_file():
            raise FileNotFoundError(
                f"No trained weights at {weights_path}. Train first with:\n"
                f"  python train_test.py --model_name deep_feature_ad "
                f"--product_class {self.product_class} --mode train --train_path {self.train_path}"
            )
        self.detector.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.detector.eval()

        thresholds = self._load_thresholds(threshold_path)
        operating_threshold = thresholds[0] if thresholds else None

        test_dataset = self._build_test_dataset(self.product_class)
        test_loader = DataLoader(test_dataset, batch_size=self.model_config["batch_size"], shuffle=False)

        seg_dir = os.path.join(self.test_path, f"segmentation_{self.product_class}")
        n_to_save = resolve_sample_count(self.num_seg_samples, len(test_dataset))
        input_size = self.model_config["input_size"]
        saved = 0

        test_scores = []
        test_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                images = batch["sample"].to(self.device)
                image_paths = batch["image_path"]

                features, reconstructed = self.detector(images)
                error_map = self.detector.compute_reconstruction_error(features, reconstructed)
                scores = self.detector.compute_anomaly_score(error_map, k=10)

                # Parent folder name, not a path substring -- see TestClassicMVTecLayout.
                labels = [int(Path(path).parent.name != "good") for path in image_paths]
                test_scores.extend(scores.cpu().numpy())
                test_labels.extend(labels)

                if saved < n_to_save:
                    seg_maps = self.detector.get_segmentation_map(
                        error_map, target_size=(input_size, input_size)
                    )
                    for i in range(images.shape[0]):
                        if saved >= n_to_save:
                            break
                        true_label = "anomalous" if labels[i] else "good"
                        if operating_threshold is not None:
                            pred_label = "anomalous" if float(scores[i]) > operating_threshold else "good"
                        else:
                            pred_label = "n/a"
                        save_segmentation_comparison(
                            images[i],
                            batch["ht"][i],
                            seg_maps[i],
                            os.path.join(seg_dir, f"sample_{saved}_comparison.png"),
                            true_label,
                            pred_label,
                            mean=IMAGENET_MEAN,
                            std=IMAGENET_STD,
                        )
                        saved += 1

        test_scores = np.array(test_scores)
        test_labels = np.array(test_labels)
        roc_auc = roc_auc_score(test_labels, test_scores)

        accuracies = []
        if thresholds is not None:
            for threshold in thresholds:
                predictions = (test_scores > threshold).astype(int)
                accuracies.append(np.mean(predictions == test_labels))

        rows = [
            ("Test images", len(test_scores)),
            ("  normal", int(np.sum(test_labels == 0))),
            ("  anomalous", int(np.sum(test_labels == 1))),
        ]
        if accuracies:
            rows.append(("Threshold (operating)", float(operating_threshold)))
            rows.extend(
                (f"Accuracy at threshold {i + 1}", float(acc)) for i, acc in enumerate(accuracies)
            )
        else:
            rows.append(("Threshold", "none -- predictions unscored"))
        rows.append(("ROC AUC (image)", float(roc_auc)))
        rows.append(("Segmentation maps", f"{saved} saved -> {seg_dir}"))
        log_results_block(
            logger, f"Test results | model: deep_feature_ad | class: {self.product_class}", rows
        )

        self._save_plots(test_labels, test_scores, roc_auc, thresholds)

        os.makedirs(seg_dir, exist_ok=True)
        scores_path = os.path.join(seg_dir, "scores.json")
        with open(scores_path, "w") as file:
            json.dump(
                {
                    "product_class": self.product_class,
                    "threshold": float(operating_threshold) if operating_threshold else None,
                    "scores": [float(score) for score in test_scores],
                },
                file,
                indent=2,
            )
        logger.info("Per-image scores saved to %s", scores_path)

        return float(roc_auc)

    def _build_test_dataset(self, product_class: str) -> MVTecAD2:
        return MVTecAD2(
            product_class, "test",
            transform=self.transform, config_path=self.config_path,
        )

    def _load_thresholds(self, threshold_path: Path) -> Optional[list]:
        if not threshold_path.is_file():
            return None
        with open(threshold_path) as file:
            threshold_info = yaml.safe_load(file)

        thresholds = None
        if isinstance(threshold_info, dict):
            inner = threshold_info.get("thresholds")
            if isinstance(inner, dict) and "thresholds" in inner:
                thresholds = inner["thresholds"]
            elif "thresholds" in threshold_info:
                thresholds = threshold_info["thresholds"]
            elif "threshold" in threshold_info:
                thresholds = threshold_info["threshold"]
        elif isinstance(threshold_info, list):
            thresholds = threshold_info

        if thresholds is not None and not isinstance(thresholds, list):
            thresholds = [thresholds]
        if thresholds is not None:
            logger.info("Using thresholds: %s", thresholds)
        return thresholds

    def _save_plots(
        self, test_labels: np.ndarray, test_scores: np.ndarray, roc_auc: float, thresholds: Optional[list]
    ) -> None:
        """Save ROC curve and score distribution plots directly under ``test_path``."""
        os.makedirs(self.test_path, exist_ok=True)

        fpr, tpr, _ = roc_curve(test_labels, test_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {self.product_class.capitalize()} (AUC = {roc_auc:.4f})")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_save_path = os.path.join(self.test_path, "roc_curve.png")
        plt.savefig(roc_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("ROC curve saved to: %s", roc_save_path)

        plt.figure(figsize=(12, 6))
        plt.hist(test_scores[test_labels == 0], bins=30, alpha=0.7, label="Normal", color="blue", density=True)
        plt.hist(test_scores[test_labels == 1], bins=30, alpha=0.7, label="Anomaly", color="red", density=True)
        if thresholds is not None:
            colors = ["black", "purple", "orange"]
            threshold_names = ["Standard (3σ)", "Aggressive (1σ)", "Conservative (5σ)"]
            for i, threshold in enumerate(thresholds):
                color = colors[i] if i < len(colors) else "gray"
                name = threshold_names[i] if i < len(threshold_names) else f"Threshold {i + 1}"
                plt.axvline(threshold, color=color, linestyle="--", label=f"{name}: {threshold:.4f}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Density")
        plt.title(f"Anomaly Score Distribution - {self.product_class.capitalize()} (ROC AUC = {roc_auc:.4f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        dist_save_path = os.path.join(self.test_path, "score_distribution.png")
        plt.savefig(dist_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Score distribution saved to: %s", dist_save_path)

    # -------------------------------------------------------- visualization --

    def plot_anomalies_thresholds(self) -> None:
        """Plot the training-score histogram with each computed threshold marked."""
        thresholds = self.thresholds if isinstance(self.thresholds, list) else [self.thresholds]
        sigma_multiplier = (
            self.sigma_multiplier if isinstance(self.sigma_multiplier, list) else [self.sigma_multiplier]
        )

        fig, axes = plt.subplots(nrows=len(thresholds), ncols=1, figsize=(10, 6 * len(thresholds)))
        axes = np.atleast_1d(axes)

        for i, threshold in enumerate(thresholds):
            ax = axes[i]
            ax.hist(self.anomaly_scores, bins=50, density=True, alpha=0.7, color="blue", label="Training Errors")
            ax.axvline(self.mean_error, color="green", linestyle="--", label=f"Mean (μ): {self.mean_error:.4f}")
            ax.axvline(
                threshold, color="red", linestyle="--",
                label=f"Threshold (μ + {sigma_multiplier[i]}σ): {threshold:.4f}",
            )
            ax.set_title(
                f"Histogram of Training Reconstruction Errors "
                f"({self.model_config['threshold_computation_mode'].upper()} Mode)"
            )
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(
            self.train_path, f"training_errors_histogram_{self.model_config['threshold_computation_mode']}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Histogram saved to: %s", save_path)

    def generate_segmentation_maps(self, num_examples: int = 5, foundational: bool = False) -> None:
        """Save up to ``num_examples`` original/error-map/overlay/mask figures
        for anomalous test images."""
        self.detector.eval()
        seg_output_dir = os.path.join(self.test_path, "segmentation_maps")
        if foundational:
            seg_output_dir = os.path.join(seg_output_dir, self.product_class)
        os.makedirs(seg_output_dir, exist_ok=True)

        test_dataset = self._build_test_dataset(self.product_class)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        saved = 0
        with torch.no_grad():
            for batch in test_loader:
                if saved >= num_examples:
                    break

                images = batch["sample"].to(self.device)
                image_path = batch["image_path"][0]
                if Path(image_path).parent.name == "good":
                    continue
                i = saved
                saved += 1

                features, reconstructed = self.detector(images)
                error_map = self.detector.compute_reconstruction_error(features, reconstructed)
                seg_map = self.detector.get_segmentation_map(
                    error_map=error_map,
                    target_size=(self.model_config["input_size"], self.model_config["input_size"]),
                )

                thresholds = self.thresholds if isinstance(self.thresholds, list) else [self.thresholds]

                original_image = images[0].cpu().numpy().transpose(1, 2, 0)
                original_image = original_image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
                original_image = np.clip(original_image, 0, 1)

                seg_map = seg_map[0].cpu().numpy()

                fig, axes = plt.subplots(2, 3, figsize=(15, 8))
                axes[0, 0].imshow(original_image)
                axes[0, 0].set_title("Original Image")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(seg_map, cmap="jet")
                axes[0, 1].set_title("Segmentation Map")
                axes[0, 1].axis("off")

                axes[0, 2].imshow(original_image)
                axes[0, 2].imshow(seg_map, cmap="jet", alpha=0.5)
                axes[0, 2].set_title("Overlay Segmentation Map")
                axes[0, 2].axis("off")

                for j, threshold in enumerate(thresholds):
                    axes[1, j].imshow(original_image)
                    axes[1, j].imshow(seg_map > threshold, cmap="gray")
                    axes[1, j].set_title(f"Mask - Threshold: {threshold:.4f}")
                    axes[1, j].axis("off")

                fig.tight_layout()
                save_path = os.path.join(seg_output_dir, f"segmentation_map_{i}.png")
                fig.savefig(save_path)
                plt.close(fig)
                logger.info("Segmentation map saved for image %d at %s", i + 1, save_path)
