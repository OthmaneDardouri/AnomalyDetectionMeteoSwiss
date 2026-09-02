"""Regression tests: each one pins a specific fixed bug so it can't come back."""
from pathlib import Path

import pytest
import torch

from anom_detect.dataset_preprocessor import MVTecAD2
from anom_detect.deep_feature_ad.deep_feature_autoencoder_model import DeepFeatureAutoEncoder
from anom_detect.early_stopping import EarlyStopping
from anom_detect.vit_model.utility_fun import Binarization, Filter

PRODUCT_CLASS = "toy"


class TestClassicMVTecLayout:
    """Anomaly folders are named per defect type, not 'bad'.

    A ``"bad" in path`` label rule marks every classic-MVTec test image normal,
    which makes ROC AUC undefined.
    """

    def test_defect_type_folders_are_discovered(
        self, classic_layout_dataset, classic_layout_config_path
    ):
        dataset = MVTecAD2(
            PRODUCT_CLASS,
            "test",
            config_path=str(classic_layout_config_path),
        )
        # 2 good + 1 crack + 1 cut
        assert len(dataset) == 4

        folders = {Path(p).parent.name for p in dataset.image_paths}
        assert folders == {"good", "crack", "cut"}
        # nothing here is called "bad" -- that is the whole point
        assert "bad" not in folders

    def test_parent_folder_label_rule_separates_classes(
        self, classic_layout_dataset, classic_layout_config_path
    ):
        dataset = MVTecAD2(
            PRODUCT_CLASS,
            "test",
            config_path=str(classic_layout_config_path),
        )
        labels = [int(Path(p).parent.name != "good") for p in dataset.image_paths]
        assert sorted(labels) == [0, 0, 1, 1], "expected 2 normal and 2 anomalous"

        # The substring rule collapses to one class, which is what made
        # roc_auc_score raise on classic MVTec AD.
        old_rule = [int("bad" in p) for p in dataset.image_paths]
        assert len(set(old_rule)) == 1


class TestFeatureExtractorSmoothing:
    """``smooth=False`` must not raise: the smoothed tensors were once bound
    only inside ``if self.smooth:`` but referenced unconditionally."""

    @pytest.mark.slow
    @pytest.mark.parametrize("smooth", [True, False])
    def test_forward_works_with_and_without_smoothing(self, smooth):
        model = DeepFeatureAutoEncoder(
            layer_hooks=["layer2", "layer3"], latent_dim=8, is_bn=True, smooth=smooth
        ).eval()

        with torch.no_grad():
            features, reconstructed = model(torch.rand(1, 3, 64, 64))

        assert features.shape == reconstructed.shape
        assert features.shape[1] == 512 + 1024  # layer2 + layer3 channels


class TestFilter:
    """An unsupported filter type must raise ValueError, not fall through."""

    def test_unsupported_type_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported filter_type"):
            Filter(torch.zeros(1, 1, 8, 8).numpy(), filter_type=99)

    @pytest.mark.parametrize("filter_type", [0, 1])
    def test_supported_types_return_same_shape(self, filter_type):
        score_map = torch.rand(1, 1, 8, 8).numpy()
        assert Filter(score_map, filter_type=filter_type).shape == score_map.shape


class TestBinarization:
    def test_default_returns_binary_mask(self):
        mask = torch.tensor([0.0, 0.4, 0.9]).numpy()
        assert Binarization(mask, thres=0.5).tolist() == [0.0, 0.0, 1.0]

    def test_keep_values_preserves_magnitudes(self):
        mask = torch.tensor([0.0, 0.4, 0.9]).numpy()
        result = Binarization(mask, thres=0.5, keep_values=True).tolist()
        assert result == pytest.approx([0.0, 0.0, 0.9])


class TestEarlyStopping:
    """One metric per instance: feeding two silently corrupts the counter."""

    def test_stops_after_patience_non_improving_epochs(self):
        stopper = EarlyStopping(patience=2, delta=0.0)
        stopper.check_early_stop(1.0)
        assert not stopper.stop_training
        stopper.check_early_stop(1.0)
        assert not stopper.stop_training
        stopper.check_early_stop(1.0)
        assert stopper.stop_training

    def test_improvement_resets_the_counter(self):
        stopper = EarlyStopping(patience=2, delta=0.0)
        stopper.check_early_stop(1.0)
        stopper.check_early_stop(1.0)
        stopper.check_early_stop(0.5)  # improvement
        assert stopper.no_improvement_count == 0
        assert not stopper.stop_training

    def test_delta_defines_what_counts_as_improvement(self):
        stopper = EarlyStopping(patience=1, delta=0.1)
        stopper.check_early_stop(1.0)
        stopper.check_early_stop(0.95)  # better, but by less than delta
        assert stopper.stop_training
