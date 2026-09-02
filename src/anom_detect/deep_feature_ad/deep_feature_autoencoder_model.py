"""DFR model: a frozen ResNet50 feature extractor plus a 1x1-conv autoencoder
that reconstructs those features (never pixels)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

# Channel counts of each hookable ResNet50 stage.
LAYER_CHANNELS = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}


class FeatureExtractor(nn.Module):
    """Frozen ResNet50; concatenates hooked layer activations into one feature map.

    Hooking several layers mixes the spatial detail of the early ones with the
    semantic content of the later ones.
    """

    def __init__(self, layer_hooks=None, smooth=True):
        super().__init__()
        if layer_hooks is None:
            layer_hooks = ["layer2", "layer3"]

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.collected_features = []

        def hook_capture(_, __, output):
            self.collected_features.append(output)

        for layer in layer_hooks:
            getattr(self.backbone, layer)[-1].register_forward_hook(hook_capture)

        # Smoothing trades a little detail for robustness to isolated activations.
        self.smooth = smooth
        self.smoothing_layer = (
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1) if self.smooth else None
        )

    def forward(self, x):
        """[B, 3, H, W] -> [B, sum(layer_channels), h, w].

        Works for any number of hooks: every activation is resized to the
        first (highest-resolution) one and concatenated channel-wise.
        """
        self.collected_features = []

        with torch.no_grad():
            _ = self.backbone(x)

        target_size = self.collected_features[0].shape[-2:]

        aligned_features = []
        for features in self.collected_features:
            if features.shape[-2:] != target_size:
                features = F.interpolate(features, size=target_size, mode="bilinear")
            if self.smooth:
                features = self.smoothing_layer(features)
            aligned_features.append(features)

        return torch.cat(aligned_features, dim=1)


class ConvBlock(nn.Module):
    """Conv -> optional BatchNorm -> ReLU, shared by the Encoder and Decoder.

    kernel_size=1 by default: the ResNet features already carry the spatial
    structure, so the autoencoder only mixes channels. Spatial resolution never
    changes, which is why there are no transposed convolutions.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, is_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if is_bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = x if self.bn is None else self.bn(x)
        return self.relu(x)


class Encoder(nn.Module):
    """Compresses channels in three steps: in -> (in+2L)/2 -> 2L -> L."""

    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super().__init__()
        mid = (in_channels + 2 * latent_dim) // 2
        self.layer1 = ConvBlock(in_channels, mid, is_bn=is_bn)
        self.layer2 = ConvBlock(mid, 2 * latent_dim, is_bn=is_bn)
        self.layer3 = ConvBlock(2 * latent_dim, latent_dim, is_bn=is_bn)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class Decoder(nn.Module):
    """Mirror of :class:`Encoder`: L -> 2L -> (in+2L)/2 -> in."""

    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super().__init__()
        mid = (in_channels + 2 * latent_dim) // 2
        self.layer1 = ConvBlock(latent_dim, 2 * latent_dim, is_bn=is_bn)
        self.layer2 = ConvBlock(2 * latent_dim, mid, is_bn=is_bn)
        self.layer3 = ConvBlock(mid, in_channels, is_bn=is_bn)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))


class AE(nn.Module):
    """Autoencoder over *features*, not pixels.

    The backbone already knows edges and textures, so this only has to learn
    what "normal" looks like in that space.
    """

    def __init__(self, in_channels=1536, latent_dim=100, is_bn=True):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)
        self.decoder = Decoder(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DeepFeatureAutoEncoder(nn.Module):
    """:class:`FeatureExtractor` + :class:`AE`; returns ``(features, reconstructed)``."""

    def __init__(self, layer_hooks=None, latent_dim=100, is_bn=True, smooth=True):
        super().__init__()
        if layer_hooks is None:
            layer_hooks = ["layer2", "layer3"]

        self.feature_extractor = FeatureExtractor(layer_hooks=layer_hooks, smooth=smooth)
        in_channels = sum(LAYER_CHANNELS[layer] for layer in layer_hooks)
        self.autoencoder = AE(in_channels=in_channels, latent_dim=latent_dim, is_bn=is_bn)

    def get_stats(self):
        """Layer and parameter counts, for logging."""
        return {
            "num_layers": len(list(self.modules())),
            "backbone_params": sum(p.numel() for p in self.feature_extractor.backbone.parameters()),
            "autoencoder_params": sum(p.numel() for p in self.autoencoder.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(self, x):
        features = self.feature_extractor(x)
        return features, self.autoencoder(features)
