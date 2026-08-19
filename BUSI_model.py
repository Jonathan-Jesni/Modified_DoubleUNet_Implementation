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

LEGACY_ASPP_RATES = (6, 12, 18)


def aspp_rates_for(feature_size):
    """Atrous rates for c2/c3/c4 that keep 3x3 taps inside the feature map.

    The legacy rates (6, 12, 18) come from DeepLabv3 at output stride 16 on 513px
    input, where the ASPP feature map is 33x33.  This network's ASPP runs on
    ``input_size // 16``, which is 16x16 at the BUSI default.  A tap at +/-18 can
    then never land inside the map, so 8 of the 9 weights in c4 receive exactly
    zero gradient and the branch degenerates into a 1x1 convolution; c3 spends
    75% of its taps on zero padding.  These rates keep every branch mostly valid
    while preserving the coarse-to-fine progression ASPP depends on.
    """

    size = int(feature_size)
    if size <= 0:
        raise ValueError("feature_size must be positive")
    if size <= 16:
        rates = (2, 3, 5)
    elif size <= 24:
        rates = (2, 4, 6)
    elif size <= 40:
        rates = (3, 6, 9)
    else:
        rates = LEGACY_ASPP_RATES
    # A 3x3 kernel at rate r spans 2r+1 pixels; keep that within the map.
    limit = max(1, (size - 1) // 2)
    return tuple(min(rate, limit) for rate in rates)


class ASPP(nn.Module):
    """ASPP with resolution-aware atrous rates and a global image-pooling branch."""

    def __init__(self, in_c, out_c, rates=LEGACY_ASPP_RATES, pool_size=1):
        super().__init__()
        rates = tuple(int(rate) for rate in rates)
        if len(rates) != 3 or any(rate <= 0 for rate in rates):
            raise ValueError("ASPP rates must be three positive dilations for c2/c3/c4")
        self.rates = rates
        self.pool_size = int(pool_size)
        if self.pool_size <= 0:
            raise ValueError("ASPP pool_size must be positive")

        # pool_size=1 is DeepLabv3's image-pooling branch and supplies true global
        # context; the historical (2, 2) pooling only ever saw four coarse regions.
        #
        # At pool_size=1 the branch is deliberately BatchNorm-free. BatchNorm2d on a
        # 1x1 map sees exactly one value per channel per sample, so a batch of one
        # raises "Expected more than 1 value per channel when training". The BUSI fit
        # pools are 472-474 samples at batch_size 8 with drop_last=False, so folds
        # 1/2/4 end an epoch on a single-sample batch and would crash mid-run.
        if self.pool_size == 1:
            pool_projection = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            pool_projection = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.pool_size, self.pool_size)),
            pool_projection,
        )
        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.c5 = Conv2D(out_c*5, out_c, kernel_size=1, padding=0, dilation=1)

    @property
    def c4_dilation(self):
        return self.rates[2]

    @torch.no_grad()
    def configure_rates_(self, rates):
        """Change atrous geometry in place without replacing checkpoint tensors."""

        rates = tuple(int(rate) for rate in rates)
        if len(rates) != 3 or any(rate <= 0 for rate in rates):
            raise ValueError("ASPP rates must be three positive dilations for c2/c3/c4")
        self.rates = rates
        for branch, dilation in zip((self.c2, self.c3, self.c4), rates):
            convolution = branch.conv[0]
            convolution.dilation = (dilation, dilation)
            convolution.padding = (dilation, dilation)
        return self

    @torch.no_grad()
    def configure_c4_dilation_(self, dilation, centre_only=False):
        """Change c4 geometry in place without replacing checkpoint tensors."""

        dilation = int(dilation)
        if dilation <= 0:
            raise ValueError("ASPP dilation must be positive")
        self.configure_rates_((self.rates[0], self.rates[1], dilation))
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

class StagedFineTuningMixin:
    """Staged backbone unfreezing and BatchNorm policy for both pipelines.

    Requires the host module to expose ``self.e1`` with the ensemble encoder's
    ``xception`` / ``dense_block2`` / ``dense_block3`` / ``vgg_block4`` /
    ``vgg_block5`` submodules, plus ``self.training_phase`` and ``self.bn_policy``.

    Shared rather than duplicated on purpose. The encoder1 unfreeze fix landed in
    the BUSI pipeline in July and was still missing from CBIS in August, which left
    a randomly-initialised proj1 feeding frozen DenseNet blocks that could never
    adapt to it. One implementation makes that class of drift impossible.
    """

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


class build_doubleunet(StagedFineTuningMixin, nn.Module):
    """Construct the core or conservative-v2 BUSI Modified Double U-Net."""

    def __init__(
        self,
        variant="core",
        num_classes=3,
        preprocessing_profile=DEFAULT_PREPROCESSING_PROFILE,
        input_size=256,
        pretrained=True,
        bn_policy="targeted",
        aspp_rates="auto",
        aspp_pool=1,
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

        # Both ASPP blocks run on input_size // 16: a1 after the four encoder1
        # downsamples, a2 after encoder2's four max-pools.
        self.aspp_feature_size = min(self.input_size) // 16
        self.aspp_pool = int(aspp_pool)
        if isinstance(aspp_rates, str):
            if aspp_rates == "auto":
                resolved_rates = aspp_rates_for(self.aspp_feature_size)
            elif aspp_rates == "legacy":
                resolved_rates = (
                    (*LEGACY_ASPP_RATES[:2], 3)
                    if variant == "v2"
                    else LEGACY_ASPP_RATES
                )
            else:
                raise ValueError(
                    f"Unsupported aspp_rates {aspp_rates!r}; expected 'auto', "
                    "'legacy', or three explicit dilations"
                )
        else:
            resolved_rates = tuple(int(rate) for rate in aspp_rates)
        self.aspp_rates = resolved_rates

        # Shared construction order matches the historical implementation.
        self.e1 = encoder1(pretrained=pretrained)
        self.a1 = ASPP(512, 64, rates=resolved_rates, pool_size=self.aspp_pool)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.e2 = encoder2()
        self.a2 = ASPP(256, 64, rates=resolved_rates, pool_size=self.aspp_pool)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)

        if variant == "v2":
            # Attach last so same-seed shared tensors remain byte-identical.
            self.e1.install_v2_adapters()
            if aspp_rates == "legacy":
                # Historical v2 repair: at rate 18 on a 16x16 map c4's off-centre
                # taps were dead, so they were reset to zero and relearned. The
                # auto schedule removes the degeneracy at its source, which makes
                # the reset redundant - keep it only when reproducing legacy runs.
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
            "aspp_rates": list(self.aspp_rates),
            "aspp_pool": self.aspp_pool,
            "aspp_feature_size": self.aspp_feature_size,
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


def predict_probabilities(model, images, tta=False):
    """Softmax probabilities for both heads, optionally horizontal-flip averaged.

    Horizontal flip is the only test-time transform used here because it is the
    one geometric augmentation training already applies (A.HorizontalFlip in both
    the conservative_ultrasound and CBIS pipelines), so the averaged views stay on
    the distribution the model was fitted to. Ultrasound and mammography both have
    a meaningful vertical axis - depth and chest-wall direction - which is why
    vertical flip is not included.
    """

    p1, p2 = model(images)
    p1 = torch.softmax(p1.float(), dim=1)
    p2 = torch.softmax(p2.float(), dim=1)
    if not tta:
        return p1, p2
    flipped1, flipped2 = model(torch.flip(images, dims=(3,)))
    p1 = 0.5 * (p1 + torch.flip(torch.softmax(flipped1.float(), dim=1), dims=(3,)))
    p2 = 0.5 * (p2 + torch.flip(torch.softmax(flipped2.float(), dim=1), dims=(3,)))
    return p1, p2


class WeightEMA:
    """Exponential moving average of model weights, evaluated alongside the raw ones.

    Kept as a separate shadow copy rather than replacing the live parameters, so a
    screening run can compare both without a second training run.
    """

    def __init__(self, model, decay=0.999):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if value.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach().float(), alpha=1.0 - self.decay
                )

    def state_dict(self):
        return {name: value.clone() for name, value in self.shadow.items()}

    def copy_to(self, model):
        """Load the averaged weights into ``model`` (in place)."""

        target = model.state_dict()
        merged = {
            name: (
                self.shadow[name].to(value.dtype)
                if name in self.shadow
                else value
            )
            for name, value in target.items()
        }
        model.load_state_dict(merged, strict=True)
        return model


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
