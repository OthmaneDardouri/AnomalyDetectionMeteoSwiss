import glob
import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_dataset_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load configuration independently from the caller's working directory."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def resolve_dataset_path(
    config: dict, config_path: Optional[Union[str, Path]] = None
) -> Path:
    """Absolute ``DATASET_PATH``, anchored at the config file's directory.

    Anchoring there rather than at the working directory keeps a checked-in
    path like ``data/mvtec_subset`` valid from anywhere.
    """
    raw = str(config.get("DATASET_PATH", "") or "").strip()
    if not raw:
        raise ValueError(
            "DATASET_PATH is missing or empty in the configuration file. Point it "
            "at the directory containing the per-class MVTec AD folders."
        )
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    base = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    return (base.resolve().parent / path).resolve()


class MVTecAD2(Dataset):
    """MVTec AD dataset for one product class and split.

    Handles both classic MVTec AD (one folder per defect type) and MVTec AD 2
    (a single ``bad`` folder) -- see ``_get_pattern``. ``mad2_object`` may be
    ``"all"`` to concatenate every class in ``DATASET_OBJECTS``.
    """

    def __init__(
        self,
        mad2_object: str,
        split: str,
        transform: Optional[Callable] = to_tensor,
        dataset_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        config = load_dataset_config(config_path)
        dataset_objects = config.get("DATASET_OBJECTS", [])
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if mad2_object not in [*dataset_objects, "all"]:
            raise ValueError(f"Unknown MVTec AD 2 object: {mad2_object}")

        self.object = mad2_object
        self.split = split
        self.transform = transform

        self._image_base_dir = (
            Path(dataset_path).expanduser()
            if dataset_path
            else resolve_dataset_path(config, config_path)
        )
        if not self._image_base_dir.is_dir():
            raise FileNotFoundError(
                "Dataset path not found. Set DATASET_PATH in config.yaml or pass "
                f"dataset_path explicitly: {self._image_base_dir}"
            )
        if self.object == "all":
            self._image_paths = []
            for obj in dataset_objects:
                object_dir = self._image_base_dir / obj
                self._image_paths.extend(sorted(self._get_pattern(object_dir)))
        else:
            object_dir = self._image_base_dir / mad2_object
            self._image_paths = sorted(self._get_pattern(object_dir))

    def _get_pattern(self, object_dir: Path) -> list[str]:
        split_dir = object_dir / self.split
        # Discover whichever defect folders exist rather than hardcoding 'bad',
        # which finds zero anomalies on classic MVTec AD.
        subfolders = sorted(d.name for d in split_dir.iterdir() if d.is_dir()) if split_dir.is_dir() else []
        if self.split == "train":
            # Normally 'good' only, but some datasets ship a contaminated 'bad' subset.
            subfolders = [name for name in subfolders if name in ("good", "bad")]

        all_matches = []
        for name in subfolders:
            pattern = str(split_dir / name / "**" / "*.png")
            matches = glob.glob(pattern, recursive=True)
            if not matches:
                matches = glob.glob(pattern.replace("**" + os.sep, ""), recursive=True)
            all_matches.extend(matches)
        return all_matches

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict:
        """The sample image and its path; the ``test`` split also carries ``ht``,
        the ground-truth mask."""
        image_path = self._image_paths[idx]
        sample = default_loader(image_path)
        if self.transform is not None:
            sample = self.transform(sample)

        item = {"sample": sample, "image_path": image_path}
        if self.split == "test":
            item["ht"] = self.get_gt_image(idx)
        return item

    @property
    def image_paths(self) -> list[str]:
        return self._image_paths

    @property
    def has_segmentation_gt(self) -> bool:
        return self.split == "test"

    def get_gt_image(self, idx: int) -> torch.Tensor:
        """The ground-truth mask for sample ``idx``, scaled to 0.0-1.0.

        All zeros for good images and for splits with no segmentation truth.
        """
        image_path = Path(self.image_paths[idx])
        defect_type = image_path.parent.name
        if self.has_segmentation_gt and defect_type != "good":
            object_dir = image_path.parents[2]
            gt_image_path = object_dir / "ground_truth" / defect_type / (
                f"{image_path.stem}_mask{image_path.suffix}"
            )
            if not gt_image_path.is_file():
                raise FileNotFoundError(
                    f"Ground-truth mask not found for {image_path}: {gt_image_path}"
                )
            gt_image = np.array(Image.open(gt_image_path))
        else:
            with Image.open(image_path) as sample_image:
                gt_image = np.zeros(sample_image.size[::-1], dtype=np.uint8)

        gt_tensor = torch.from_numpy(gt_image).unsqueeze(0).float() / 255.0

        # Mirror only the geometric transforms, so the mask stays aligned.
        if self.transform is not None:
            for t in getattr(self.transform, "transforms", []):
                if isinstance(t, (transforms.Resize, transforms.CenterCrop)):
                    gt_tensor = t(gt_tensor)

        return gt_tensor
