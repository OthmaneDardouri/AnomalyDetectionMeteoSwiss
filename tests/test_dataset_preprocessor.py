import pytest
import torch

from anom_detect.dataset_preprocessor import MVTecAD2

PRODUCT_CLASS = "toy"


def test_train_split_has_no_ground_truth(tiny_dataset, tiny_config_path):
    dataset = MVTecAD2(
        PRODUCT_CLASS, "train", config_path=str(tiny_config_path)
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert isinstance(sample["sample"], torch.Tensor)
    assert "ht" not in sample
    assert dataset.has_segmentation_gt is False


def test_test_split_has_ground_truth_masks(tiny_dataset, tiny_config_path):
    dataset = MVTecAD2(
        PRODUCT_CLASS, "test", config_path=str(tiny_config_path)
    )
    assert len(dataset) == 4  # 2 good + 2 bad
    assert dataset.has_segmentation_gt is True

    labels = [0 if "good" in path else 1 for path in dataset.image_paths]
    assert sorted(labels) == [0, 0, 1, 1]

    bad_index = labels.index(1)
    gt = dataset[bad_index]["ht"]
    assert gt.max().item() > 0  # the synthetic mask has an anomalous region

    good_index = labels.index(0)
    gt_good = dataset[good_index]["ht"]
    assert gt_good.max().item() == 0  # good images get an all-zero mask


def test_unknown_product_class_rejected(tiny_dataset, tiny_config_path):
    with pytest.raises(ValueError):
        MVTecAD2(
            "not_a_real_class",
            "train",
            config_path=str(tiny_config_path),
        )
