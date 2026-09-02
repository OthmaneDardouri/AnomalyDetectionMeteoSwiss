import argparse
import json
import os
from math import isqrt
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from anom_detect.dataset_preprocessor import MVTecAD2, resolve_dataset_path
from anom_detect.logging_utils import get_logger, log_results_block
from anom_detect.resnet_features import ResNetPatchFeatures
from anom_detect.visualization import resolve_sample_count, save_segmentation_comparison

logger = get_logger(__name__)


class PatchFeatureExtractor(ResNetPatchFeatures):
    """Flattens the shared backbone's feature map into one row per spatial
    patch, ready to be stored in / matched against a PatchCore memory bank."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = super().forward(x)
        return feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])


class PatchCoreManager:
    """Memory-bank anomaly detector: stores a subsample of normal-image patch
    features, then scores new images by their nearest-neighbour distance to
    that bank.

    Two deliberate simplifications versus the paper: the memory bank is a
    uniform random subsample rather than a greedy coreset, and scoring uses
    the 1-NN distance rather than a re-weighted k-NN score.
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

        self.model_config = self.config["MODELS_CONFIG"].get("patchcore", {})
        self.subsample_ratio = float(self.model_config.get("memory_bank_subsample_ratio", 0.1))
        self.threshold_multiplier = float(self.model_config.get("threshold_multiplier", 2.0))
        self.distance_chunk_size = int(self.model_config.get("distance_chunk_size", 10_000))
        self.seed = int(self.config.get("SEED", 42))
        if not 0 < self.subsample_ratio <= 1:
            raise ValueError("memory_bank_subsample_ratio must be in (0, 1].")
        if self.distance_chunk_size < 1:
            raise ValueError("distance_chunk_size must be positive.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.feature_extractor = PatchFeatureExtractor().to(self.device).eval()

        self.train_dataset = MVTecAD2(self.product_class, "train", self.transform, config_path=self.config_path)
        self.test_dataset = MVTecAD2(self.product_class, "test", transform=self.transform, config_path=self.config_path)

        self.memory_bank = None
        self.sub_memory_bank = None
        self.threshold = None
        self.train_scores = None

        self._check_directories()

    def _extract_features(self, dataset: Dataset, desc: str) -> torch.Tensor:
        """Run the frozen backbone over every image in ``dataset`` and stack the patch features."""
        features = []
        for x in tqdm(dataset, desc=desc, total=len(dataset)):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))
                features.append(patches.detach())
        return torch.cat(features, dim=0)

    def _score_dataset(self, dataset: Dataset, desc: str) -> list[float]:
        """Score every image in ``dataset`` by its max nearest-neighbour distance to the memory bank."""
        scores = []
        for x in tqdm(dataset, desc=desc, total=len(dataset)):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))
                dist_score = self._nearest_distances(patches)
                scores.append(dist_score.max().item())
        return scores

    def train(self) -> None:
        """Extract features from training data and build the memory bank."""
        logger.info("=== Training PatchCore for class: %s ===", self.product_class)

        train_output_dir = os.path.join(self.train_path, f"train_patchcore_{self.product_class}")
        os.makedirs(train_output_dir, exist_ok=True)
        self.train_output_dir = train_output_dir

        self.memory_bank = self._extract_features(
            self.train_dataset, desc=f"[{self.product_class}] Feature Extraction (Train)"
        )

        subsample_size = max(1, int(self.memory_bank.shape[0] * self.subsample_ratio))
        rng = np.random.default_rng(self.seed)
        selected_patches = rng.choice(self.memory_bank.shape[0], size=subsample_size, replace=False)
        self.sub_memory_bank = self.memory_bank[selected_patches]

        logger.info("Memory bank created with %d patches", self.memory_bank.shape[0])
        logger.info("Subsampled to %d patches", self.sub_memory_bank.shape[0])

    def compute_thresh(self) -> tuple[float, float, float]:
        """Compute threshold = mean + threshold_multiplier * std of training scores."""
        if self.sub_memory_bank is None:
            raise ValueError("Must call train() before compute_thresh()")

        logger.info("Computing threshold from training data...")

        y_score_max = self._score_dataset(
            self.train_dataset, desc=f"[{self.product_class}] Scoring (Train)"
        )

        self.train_scores = y_score_max
        mean = np.mean(y_score_max)
        std = np.std(y_score_max)
        self.threshold = mean + self.threshold_multiplier * std

        logger.info("Mean training score: %.4f", mean)
        logger.info("Std training score: %.4f", std)
        logger.info("Threshold: %.4f", self.threshold)

        plt.figure()
        plt.hist(y_score_max, bins=30, alpha=0.7, color="blue", label="Training Scores")
        plt.axvline(mean, color='green', linestyle='dashed', linewidth=1, label='Mean')
        plt.axvline(self.threshold, color='r', linestyle='dashed', linewidth=1, label='Threshold')
        plt.title(f"Training Scores Histogram - {self.product_class}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(os.path.join(self.train_output_dir, "training_errors_histogram.png"))
        plt.close()

        return mean, std, self.threshold

    def test(self) -> dict:
        """Score the test dataset, save sample segmentation comparisons, and return ROC AUC."""
        if self.sub_memory_bank is None or self.threshold is None:
            self.load_model()  # not trained in this session

        logger.info("=== Testing PatchCore for class: %s ===", self.product_class)

        # Same layout as every other model's test(), so runs stay comparable.
        test_output_dir = os.path.join(self.test_path, f"segmentation_{self.product_class}")
        os.makedirs(test_output_dir, exist_ok=True)

        y_test_score = []
        y_test_true = []
        n_to_save = resolve_sample_count(self.num_seg_samples, len(self.test_dataset))

        for idx, x in enumerate(tqdm(self.test_dataset, desc=f"[{self.product_class}] Inference (Test)", total=len(self.test_dataset))):
            with torch.no_grad():
                image = x["sample"].to(self.device)
                patches = self.feature_extractor(image.unsqueeze(0))

                dist_score = self._nearest_distances(patches)
                side = isqrt(dist_score.numel())
                if side * side != dist_score.numel():
                    raise RuntimeError("Patch features do not form a square segmentation map.")
                seg_map = dist_score.view(1, 1, side, side)
                score = dist_score.max().item()
                y_test_score.append(score)
                label = Path(x["image_path"]).parent.name

                y_test_true.append(0 if label == "good" else 1)

                if idx < n_to_save:
                    self._save_comparison_plot(x, seg_map, score, test_output_dir, idx)

        if len(set(y_test_true)) < 2:
            raise ValueError("ROC AUC requires both normal and anomalous test samples.")
        auc_roc_score = roc_auc_score(y_test_true, y_test_score)
        accuracy = float(
            np.mean((np.asarray(y_test_score) > self.threshold) == np.asarray(y_test_true))
        )
        log_results_block(
            logger,
            f"Test results | model: patchcore | class: {self.product_class}",
            [
                ("Test images", len(y_test_true)),
                ("  normal", y_test_true.count(0)),
                ("  anomalous", y_test_true.count(1)),
                ("Threshold", self.threshold),
                ("ROC AUC (image)", auc_roc_score),
                ("Accuracy at threshold", accuracy),
                ("Segmentation maps", f"{min(n_to_save, len(y_test_true))} saved -> {test_output_dir}"),
            ],
        )

        with open(os.path.join(test_output_dir, f"{self.product_class}_metrics.txt"), "w") as f:
            f.write(f"AUC ROC Score: {auc_roc_score:.4f}\n")
            f.write(f"Threshold used: {self.threshold:.4f}\n")

        scores_path = os.path.join(test_output_dir, "scores.json")
        with open(scores_path, "w") as f:
            json.dump(
                {
                    "product_class": self.product_class,
                    "threshold": float(self.threshold),
                    "scores": [float(s) for s in y_test_score],
                },
                f,
                indent=2,
            )
        logger.info("Per-image scores saved to %s", scores_path)

        # A dict, not a bare AUC: threshold_multiplier moves the accuracy
        # without touching the AUC, so a sweep needs both.
        return {"roc_auc_image": float(auc_roc_score), "accuracy_at_threshold": accuracy}

    def _save_comparison_plot(
        self, sample: dict, seg_map: torch.Tensor, score: float, test_output_dir: str, idx: int
    ) -> None:
        """Save a side-by-side original / ground-truth / predicted-mask comparison figure."""
        interpolated_map = nn.functional.interpolate(seg_map, size=(224, 224), mode='bilinear')

        true_label = "anomalous" if Path(sample["image_path"]).parent.name != "good" else "good"
        pred_label = "anomalous" if score > self.threshold else "good"

        save_segmentation_comparison(
            sample["sample"],
            sample["ht"],
            interpolated_map.squeeze(0),
            os.path.join(test_output_dir, f"sample_{idx}_comparison.png"),
            true_label,
            pred_label,
        )

    def save_model(
        self, args: argparse.Namespace, mean_error: float, std_error: float, threshold: float
    ) -> None:
        """Save the memory bank, config, CLI args, and training statistics."""
        memory_bank_path = os.path.join(self.train_output_dir, "memory_bank.pth")
        torch.save({
            'memory_bank': self.memory_bank,
            'sub_memory_bank': self.sub_memory_bank,
            'threshold': float(threshold),
            'train_scores': self.train_scores
        }, memory_bank_path)
        logger.info("Memory bank saved to %s", memory_bank_path)

        config_save_path = os.path.join(self.train_output_dir, "config.yaml")
        with open(config_save_path, "w") as file:
            yaml.dump(self.config, file)
        logger.info("Configuration saved to %s", config_save_path)

        args_save_path = os.path.join(self.train_output_dir, "args.yaml")
        with open(args_save_path, "w") as file:
            yaml.dump(vars(args), file)
        logger.info("Arguments saved to %s", args_save_path)

        stats_save_path = os.path.join(self.train_output_dir, "training_statistics.yaml")
        with open(stats_save_path, "w") as file:
            stats = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "threshold": float(threshold),
            }
            yaml.dump(stats, file)
        logger.info("Training statistics saved to %s", stats_save_path)

    def load_model(self) -> None:
        """Load a saved memory bank / threshold for testing without retraining."""
        train_output_dir = os.path.join(self.train_path, f"train_patchcore_{self.product_class}")
        memory_bank_path = os.path.join(train_output_dir, "memory_bank.pth")

        if os.path.exists(memory_bank_path):
            checkpoint = torch.load(memory_bank_path, map_location=self.device)
            self.memory_bank = checkpoint['memory_bank'].to(self.device)
            self.sub_memory_bank = checkpoint['sub_memory_bank'].to(self.device)
            self.threshold = checkpoint['threshold']
            self.train_scores = checkpoint.get('train_scores', None)
            logger.info("Model loaded from %s", memory_bank_path)
        else:
            raise FileNotFoundError(f"No saved model found at {memory_bank_path}")

    def _check_directories(self) -> None:
        """Verify the dataset path and this class's train/test folders exist."""
        dataset_path = resolve_dataset_path(self.config, self.config_path)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        product_train_path = os.path.join(dataset_path, self.product_class, "train")
        product_test_path = os.path.join(dataset_path, self.product_class, "test")

        if not os.path.exists(product_train_path):
            raise FileNotFoundError(f"Training data path not found: {product_train_path}")

        if not os.path.exists(product_test_path):
            logger.warning("Test data path not found: %s", product_test_path)

    def _nearest_distances(self, patches: torch.Tensor) -> torch.Tensor:
        """Find nearest memory-bank distances without materialising a huge matrix."""
        if self.sub_memory_bank is None:
            raise RuntimeError("Memory bank has not been initialized.")
        minimum = torch.full((patches.shape[0],), float("inf"), device=self.device)
        for chunk in self.sub_memory_bank.split(self.distance_chunk_size):
            minimum = torch.minimum(minimum, torch.cdist(patches, chunk).min(dim=1).values)
        return minimum
