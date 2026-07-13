import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, densenet121


NUM_CLASSES = 3


class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
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

        if self.act:
            x = self.relu(x)

        return x


class squeeze_excitation_block(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()

        hidden = max(in_channels // ratio, 1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channel_size, _, _ = x.size()

        y = self.avgpool(x).view(batch_size, channel_size)
        y = self.fc(y).view(batch_size, channel_size, 1, 1)

        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            Conv2D(in_c, out_c, kernel_size=1, padding=0)
        )

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0, dilation=1)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=6, dilation=6)
        self.c3 = Conv2D(in_c, out_c, kernel_size=3, padding=12, dilation=12)
        self.c4 = Conv2D(in_c, out_c, kernel_size=3, padding=18, dilation=18)

        self.c5 = Conv2D(out_c * 5, out_c, kernel_size=1, padding=0, dilation=1)

    def forward(self, x):
        x0 = self.avgpool(x)
        x0 = F.interpolate(x0, size=x.size()[2:], mode="bilinear", align_corners=True)

        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)

        xc = torch.cat([x0, x1, x2, x3, x4], dim=1)
        y = self.c5(xc)

        return y


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


class encoder1(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(weights="DEFAULT").features
        densenet = densenet121(weights="DEFAULT").features

        self.xception = timm.create_model(
            "legacy_xception",
            pretrained=True,
            features_only=True
        )

        self.proj1 = nn.Conv2d(64, 64, kernel_size=1)

        self.dense_block2 = nn.Sequential(*list(densenet.children())[4:6])
        self.dense_block3 = nn.Sequential(*list(densenet.children())[6:8])

        self.vgg_block4 = vgg[18:27]
        self.vgg_block5 = vgg[27:36]

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        xception_features = self.xception(x)

        x1 = xception_features[0]
        x1 = self.proj1(x1)

        x1 = F.interpolate(
            x1,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False
        )

        x2 = self.dense_block2(x1)
        x3 = self.dense_block3(x2)

        x4 = self.vgg_block4(x3)
        x5 = self.vgg_block5(x4)

        return x5, [x4, x3, x2, x1]


class decoder1(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.c1 = conv_block(64 + 512, 256)
        self.drop1 = nn.Dropout2d(p=0.15)
        self.c2 = conv_block(256 + 256, 128)
        self.drop2 = nn.Dropout2d(p=0.15)
        self.c3 = conv_block(128 + 128, 64)
        self.drop3 = nn.Dropout2d(p=0.15)
        self.c4 = conv_block(64 + 64, 32)
        self.drop4 = nn.Dropout2d(p=0.15)

    def forward(self, x, skip):
        s1, s2, s3, s4 = skip

        x = self.up(x)
        x = torch.cat([x, s1], dim=1)
        x = self.drop1(self.c1(x))

        x = self.up(x)
        x = torch.cat([x, s2], dim=1)
        x = self.drop2(self.c2(x))

        x = self.up(x)
        x = torch.cat([x, s3], dim=1)
        x = self.drop3(self.c3(x))

        x = self.up(x)
        x = torch.cat([x, s4], dim=1)
        x = self.drop4(self.c4(x))

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
        x1 = self.c1(x)
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

        self.c1 = conv_block(64 + 512 + 256, 256)
        self.drop1 = nn.Dropout2d(p=0.15)
        self.c2 = conv_block(256 + 256 + 128, 128)
        self.drop2 = nn.Dropout2d(p=0.15)
        self.c3 = conv_block(128 + 128 + 64, 64)
        self.drop3 = nn.Dropout2d(p=0.15)
        self.c4 = conv_block(64 + 64 + 32, 32)
        self.drop4 = nn.Dropout2d(p=0.15)

    def forward(self, x, skip1, skip2):
        x = self.up(x)
        x = torch.cat([x, skip1[0], skip2[0]], dim=1)
        x = self.drop1(self.c1(x))

        x = self.up(x)
        x = torch.cat([x, skip1[1], skip2[1]], dim=1)
        x = self.drop2(self.c2(x))

        x = self.up(x)
        x = torch.cat([x, skip1[2], skip2[2]], dim=1)
        x = self.drop3(self.c3(x))

        x = self.up(x)
        x = torch.cat([x, skip1[3], skip2[3]], dim=1)
        x = self.drop4(self.c4(x))

        return x


class build_doubleunet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.num_classes = num_classes

        self.e1 = encoder1()
        self.a1 = ASPP(512, 64)
        self.d1 = decoder1()
        self.y1 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0)

        self.e2 = encoder2()
        self.a2 = ASPP(256, 64)
        self.d2 = decoder2()
        self.y2 = nn.Conv2d(32, num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x0 = x

        x, skip1 = self.e1(x)
        x = self.a1(x)
        x = self.d1(x, skip1)
        y1 = self.y1(x)

        prob_y1 = torch.softmax(y1, dim=1)

        foreground_attention = prob_y1[:, 1:, :, :].sum(dim=1, keepdim=True)

        input_x = x0 * foreground_attention

        x, skip2 = self.e2(input_x)
        x = self.a2(x)
        x = self.d2(x, skip1, skip2)
        y2 = self.y2(x)

        return y1, y2


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    model = build_doubleunet(num_classes=3)

    y1, y2 = model(x)

    print("Input :", x.shape)
    print("Output 1:", y1.shape)
    print("Output 2:", y2.shape)