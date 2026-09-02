"""End-to-end smoke test for the deep-feature autoencoder pipeline; also
covers the deep_feature_ad package's import path."""
from anom_detect.deep_feature_ad.deep_feature_ad_manager import DeepFeatureADManager

PRODUCT_CLASS = "toy"


def test_deep_feature_ad_train_threshold_and_segmentation(tmp_path, tiny_config_path):
    train_path = tmp_path / "train"
    test_path = tmp_path / "test"

    manager = DeepFeatureADManager(
        PRODUCT_CLASS,
        str(tiny_config_path),
        str(train_path),
        str(test_path),
    )

    manager.train()
    assert (train_path / "checkpoints" / f"{PRODUCT_CLASS}_dfad_weights.pth").is_file()

    manager.compute_threshold()
    threshold_file = train_path / f"{PRODUCT_CLASS}_thresholds.yaml"
    assert threshold_file.is_file()

    manager.load_computed_thresholds(str(threshold_file))
    manager.generate_segmentation_maps(num_examples=1)
    manager.plot_anomalies_thresholds()
