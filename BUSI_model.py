"""Core and conservative-v2 Modified Double U-Net models for BUSI."""

from __future__ import annotations


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, densenet121


SUPPORTED_VARIANTS = ("core", "v2")
SCREENING_ARCHITECTURE_MODES = ("standard", "adapters_only", "aspp_only")
BN_POLICIES = ("targeted", "legacy")
DEFAULT_PREPROCESSING_PROFILE = "legacy_imagenet"
_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def _normalise_input_size(input_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(input_size, int):
        size = (input_size, input_size)
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        size = (int(input_size[0]), int(input_size[1]))
    else:
        raise TypeError("input_size must be an integer or a (height, width) pair")
    if any(d <= 0 or d % 16 != 0 for d in size):
        raise ValueError(
            f"input_size must contain positive dimensions divisible by 16; got {size}"
        )
    return size


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//ratio, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()
        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)
        return x*y.expand_as(x)

class ASPP(nn.Module):
    """Legacy-layout ASPP with a configurable final atrous branch."""

    def __init__(self, in_c, out_c, c4_dilation=18):
        super().__init__()
        self.c4_dilation = int(c4_dilation)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(
            in_c,
            out_c,
            kernel_size=3,
            padding=self.c4_dilation,
            dilation=self.c4_dilation,
        )
        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0, dilation=1)

    @torch.no_grad()
    def configure_c4_dilation_(self, dilation, centre_only=False):
        """Change c4 geometry in place without replacing checkpoint tensors."""

        dilation = int(dilation)
        if dilation <= 0:
            raise ValueError("ASPP dilation must be positive")
        self.c4_dilation = dilation
        convolution = self.c4.conv[0]
        convolution.dilation = (dilation, dilation)
        convolution.padding = (dilation, dilation)
        if centre_only:
            self.zero_c4_off_centre_()
        return self

    @torch.no_grad()
    def zero_c4_off_centre_(self):
        weight = self.c4.conv[0].weight
        centre = weight[..., 1, 1].clone()
        weight.zero_()
        weight[..., 1, 1].copy_(centre)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x0, x1, x2, x3, x4], axis=1)
        return self.c5(xc)

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.c1 = Conv2D(in_c, out_c)
        self.c2 = Conv2D(out_c, out_c)
        self.a1 = squeeze_excitation_block(out_c)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.a1(x)
        return x

class _XceptionStem(nn.Module):
    """Exact modules that produce legacy-Xception feature output zero."""

    def __init__(self, source):
        super().__init__()
        self.conv1 = source.conv1
        self.bn1 = source.bn1
        self.act1 = source.act1
        self.conv2 = source.conv2
        self.bn2 = source.bn2
        self.act2 = source.act2

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        return self.act2(self.bn2(self.conv2(x)))


class _XceptionStemFeatures(nn.Module):
    """Preserve legacy xception.body keys and list-style output."""

    def __init__(self, source):
        super().__init__()
        self.body = _XceptionStem(source)

    def forward(self, x):
        return [self.body(x)]


class ResidualHandoffAdapter(nn.Module):
    """Bottleneck residual adapter that is an exact identity at creation."""

    def __init__(self, channels, bottleneck, groups):
        super().__init__()
        self.proj_in = nn.Conv2d(channels, bottleneck, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(groups, bottleneck)
        self.act = nn.ReLU(inplace=True)
        self.proj_out = nn.Conv2d(bottleneck, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        return x + self.proj_out(self.act(self.norm(self.proj_in(x))))


class encoder1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = 'DEFAULT' if pretrained else None
        # Construct canonical models before retaining the paper's fragments.
        # This preserves legacy initialisation and exact pretrained tensors.
        vgg = vgg19(weights=weights).features
        densenet = densenet121(weights=weights).features
        xception = timm.create_model('legacy_xception', pretrained=pretrained)

        self.xception = _XceptionStemFeatures(xception)
        self.dense_block2 = nn.Sequential(*list(densenet.children())[4:6])
        self.dense_block3 = nn.Sequential(*list(densenet.children())[6:8])
        self.vgg_block4 = vgg[18:27]
        self.vgg_block5 = vgg[27:36]
        self.proj1 = nn.Conv2d(64, 64, kernel_size=1)
        self.adapter_xception_dense = nn.Identity()
        self.adapter_dense_vgg = nn.Identity()

    def install_v2_adapters(self):
        self.adapter_xception_dense = ResidualHandoffAdapter(64, 16, 4)
        self.adapter_dense_vgg = ResidualHandoffAdapter(256, 64, 8)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        x1 = self.proj1(self.xception(x)[0])
        x1 = F.interpolate(
            x1, size=(input_h, input_w), mode='bilinear', align_corners=False
        )
        x1 = self.adapter_xception_dense(x1)
        x2 = self.dense_block2(x1)
        x3 = self.adapter_dense_vgg(self.dense_block3(x2))
        x4 = self.vgg_block4(x3)
        x5 = self.vgg_block5(x4)
        return x5, [x4, x3, x2, x1]

class decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(64+512, 256)
        self.c2 = conv_block(512, 128)
        self.c3 = conv_block(256, 64)
        self.c4 = conv_block(128, 32)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, s2], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, s3], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, s4], axis=1)
        x = self.c4(x)

        return x

class encoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))

        self.c1 = conv_block(3, 32)
        self.c2 = conv_block(32, 64)
        self.c3 = conv_block(64, 128)
        self.c4 = conv_block(128, 256)

    def forward(self, x):
        x0 = x

        x1 = self.c1(x0)
        p1 = self.pool(x1)

        x2 = self.c2(p1)
        p2 = self.pool(x2)

        x3 = self.c3(p2)
        p3 = self.pool(x3)

        x4 = self.c4(p3)
        p4 = self.pool(x4)

        return p4, [x4, x3, x2, x1]

class decoder2(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = conv_block(832, 256)
        self.c2 = conv_block(640, 128)
        self.c3 = conv_block(320, 64)
        self.c4 = conv_block(160, 32)

    def forward(self, x, skip1, skip2):

        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], axis=1)
        x = self.c1(x)

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], axis=1)
        x = self.c2(x)

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], axis=1)
        x = self.c3(x)

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], axis=1)
        x = self.c4(x)

        return x

class build_doubleunet(nn.Module):
    """Construct the core or conservative-v2 BUSI Modified Double U-Net."""

    def __init__(
        self,
        variant="core",
        num_classes=3,
        preprocessing_profile=DEFAULT_PREPROCESSING_PROFILE,
        input_size=256,
        pretrained=True,
        bn_policy="targeted",
    ):
        super().__init__()
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant {variant!r}; expected one of {SUPPORTED_VARIANTS}"
            )
        if num_classes != 3:
            raise ValueError(
                "BUSI requires exactly three classes: background, benign, malignant"
            )
        if bn_policy not in BN_POLICIES:
            raise ValueError(
                f"Unsupported BN policy {bn_policy!r}; expected one of {BN_POLICIES}"
            )

        self.variant = variant
        self.num_classes = int(num_classes)
        self.preprocessing_profile = str(preprocessing_profile)
        self.input_size = _normalise_input_size(input_size)
        self.pretrained = bool(pretrained)
        self.bn_policy = bn_policy
        self.training_phase = 3
        self.screening_architecture_mode = "standard"
        self._forward_has_run = False

        # Shared construction order matches the historical implementation.
        self.e1 = encoder1(pretrained=pretrained)
        c4_dilation = 3 if variant == "v2" else 18
        self.a1 = ASPP(512, 64, c4_dilation=c4_dilation)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.e2 = encoder2()
        self.a2 = ASPP(256, 64, c4_dilation=c4_dilation)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)

        if variant == "v2":
            # Attach last so same-seed shared tensors remain byte-identical.
            self.e1.install_v2_adapters()
            self.a1.zero_c4_off_centre_()
            self.a2.zero_c4_off_centre_()
        self._enforce_phase_bn_policy()

    def model_metadata(self):
        return {
            "variant": self.variant,
            "num_classes": self.num_classes,
            "class_mapping": {
                "background_or_normal": 0,
                "benign_lesion": 1,
                "malignant_lesion": 2,
            },
            "preprocessing_profile": self.preprocessing_profile,
            "input_size": list(self.input_size),
            "screening_architecture_mode": self.screening_architecture_mode,
            "bn_policy": self.bn_policy,
        }

    def _validate_runtime_input(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape [B, 3, H, W]; got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Input height and width must be divisible by 16; "
                f"got {(height, width)}"
            )

    def forward(self, x):
        self._validate_runtime_input(x)
        self._forward_has_run = True
        x0 = x
        x, skip1 = self.e1(x)
        y1 = self.y1(self.d1(self.a1(x), skip1))
        foreground_attention = torch.softmax(y1, dim=1)[:, 1:].sum(
            dim=1, keepdim=True
        )
        x, skip2 = self.e2(x0 * foreground_attention)
        y2 = self.y2(self.d2(self.a2(x), skip1, skip2))
        return y1, y2

    def _backbone_modules(self):
        xception = self.e1.xception
        dense = nn.ModuleList([self.e1.dense_block2, self.e1.dense_block3])
        vgg = nn.ModuleList([self.e1.vgg_block4, self.e1.vgg_block5])
        return xception, dense, vgg

    @staticmethod
    def _set_requires_grad(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    @staticmethod
    def _set_batchnorm_mode(module, training):
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.train(training)

    def _enforce_phase_bn_policy(self):
        if not self.training or self.bn_policy == "legacy":
            return
        xception, dense, vgg = self._backbone_modules()
        self._set_batchnorm_mode(xception, False)
        self._set_batchnorm_mode(dense, self.training_phase >= 2)
        self._set_batchnorm_mode(vgg, self.training_phase >= 2)

    def set_bn_policy(self, policy):
        """Select targeted study BN handling or ordinary legacy train/eval mode."""

        if policy not in BN_POLICIES:
            raise ValueError(
                f"Unsupported BN policy {policy!r}; expected one of {BN_POLICIES}"
            )
        self.bn_policy = policy
        if policy == "legacy":
            for module in self._backbone_modules():
                self._set_batchnorm_mode(module, self.training)
        else:
            self._enforce_phase_bn_policy()
        return self

    def set_training_phase(self, phase):
        """Apply the approved staged unfreezing and BatchNorm policy."""

        if phase not in (1, 2, 3):
            raise ValueError("training phase must be 1, 2, or 3")
        self.training_phase = int(phase)
        xception, dense, vgg = self._backbone_modules()
        self._set_requires_grad(xception, phase >= 3)
        self._set_requires_grad(dense, phase >= 2)
        self._set_requires_grad(vgg, phase >= 2)
        self._enforce_phase_bn_policy()

    def train(self, mode=True):
        super().train(mode)
        self._enforce_phase_bn_policy()
        return self

    def optimizer_parameter_groups(
        self,
        task_lr=1e-4,
        backbone_lr=1e-5,
        xception_lr=1e-6,
        task_weight_decay=1e-4,
        backbone_weight_decay=1e-5,
    ):
        """Return disjoint optimizer groups with the study decay policy."""

        module_by_parameter = {}
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters(recurse=False):
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                module_by_parameter[full_name] = module

        buckets = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("e1.xception."):
                family = "xception"
            elif name.startswith(("e1.dense_block", "e1.vgg_block")):
                family = "backbone"
            else:
                family = "task"
            module = module_by_parameter[name]
            no_decay = name.endswith(".bias") or isinstance(module, _NORM_TYPES)
            buckets.setdefault((family, no_decay), []).append(parameter)

        settings = {
            "xception": (xception_lr, backbone_weight_decay),
            "backbone": (backbone_lr, backbone_weight_decay),
            "task": (task_lr, task_weight_decay),
        }
        groups = []
        for family in ("task", "backbone", "xception"):
            lr, decay = settings[family]
            for no_decay in (False, True):
                params = buckets.get((family, no_decay), [])
                if params:
                    groups.append(
                        {
                            "name": f"{family}_{'no_decay' if no_decay else 'decay'}",
                            "params": params,
                            "lr": lr,
                            "weight_decay": 0.0 if no_decay else decay,
                        }
                    )
        return groups

    @staticmethod
    def _v2_compatible_state_dict(state_dict):
        converted = dict(state_dict)
        for key in ("a1.c4.conv.0.weight", "a2.c4.conv.0.weight"):
            if key in converted:
                source = converted[key]
                target = torch.zeros_like(source)
                target[..., 1, 1] = source[..., 1, 1]
                converted[key] = target
        return converted

    def load_legacy_state_dict(self, state_dict):
        """Load legacy/core weights while dropping disconnected Xception keys."""

        prepared = dict(state_dict)
        if self.variant == "v2":
            prepared = self._v2_compatible_state_dict(prepared)
        incompatible = self.load_state_dict(prepared, strict=False)
        adapter_prefixes = (
            "e1.adapter_xception_dense.",
            "e1.adapter_dense_vgg.",
        )
        bad_missing = [
            key
            for key in incompatible.missing_keys
            if not (self.variant == "v2" and key.startswith(adapter_prefixes))
        ]
        bad_unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("e1.xception.body.")
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Legacy checkpoint is incompatible with retained model state: "
                f"missing={bad_missing}, unexpected={bad_unexpected}"
            )
        return incompatible


def configure_screening_architecture(model, mode="standard"):
    """Configure one isolated core-architecture screening arm in place.

    This helper is deliberately restricted to a fresh ``variant="core"`` model.
    Configure it before strict checkpoint loading so adapter checkpoint keys are
    present for ``adapters_only``. Reapplying the same mode is idempotent.
    """

    if not isinstance(model, build_doubleunet):
        raise TypeError("model must be a BUSI build_doubleunet instance")
    if mode not in SCREENING_ARCHITECTURE_MODES:
        raise ValueError(
            f"Unsupported screening architecture mode {mode!r}; "
            f"expected one of {SCREENING_ARCHITECTURE_MODES}"
        )
    if mode == model.screening_architecture_mode:
        return model
    if model.variant != "core":
        raise ValueError("nonstandard screening architecture modes require a core model")
    if model.screening_architecture_mode != "standard":
        raise RuntimeError(
            "screening architecture is already configured as "
            f"{model.screening_architecture_mode!r}"
        )
    if model._forward_has_run:
        raise RuntimeError("configure screening architecture before the first forward")

    if mode == "adapters_only":
        model.e1.install_v2_adapters()
    elif mode == "aspp_only":
        model.a1.configure_c4_dilation_(3, centre_only=True)
        model.a2.configure_c4_dilation_(3, centre_only=True)
    model.screening_architecture_mode = mode
    return model


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = build_doubleunet(pretrained=False)
    y1, y2 = model(x)
    print(y1.shape, y2.shape)
