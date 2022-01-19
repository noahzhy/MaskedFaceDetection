import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchstat import stat


__all___ = [
    'PPLCNet_x0_25', 'PPLCNet_x0_35', 'PPLCNet_x0_5', 'PPLCNet_x0_75', 
    'PPLCNet_x1_0', 'PPLCNet_x1_5', 'PPLCNet_x2_0', 'PPLCNet_x2_5'
]


def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class SiLU(nn.Module):
    def __init__(self, inplace=False):
        super(SiLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * nn.Sigmoid(x)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0,),
                nn.ReLU(),
                nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
                HardSigmoid(),
                # SiLU(),
                nn.SiLU(),

        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            # here is not same as figue of paper which SE module should be last
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),
                
                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish(),
            )
        else:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish()
            )

    def forward(self, x):
        return self.conv(x)


class PPLCNet(nn.Module):
    def __init__(self, scale=1.0, num_classes=1000, dropout_prob=0.2):
        super(PPLCNet, self).__init__()
        self.cfgs = [
           # k,  c,  s, SE
            [3,  32, 1, 0],

            [3,  64, 2, 0],
            [3,  64, 1, 0],

            [3,  128, 2, 0],
            [3,  128, 1, 0],# stage 1, 28x28, idx=4

            [5,  256, 2, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],
            [5,  256, 1, 0],# stage 2, 14x14, idx=10

            [5,  512, 2, 1],
            [5,  512, 1, 1],# stage 3, 7x7, idx=12
        ]
        self.stage2 = self.cfgs[0:5]
        self.stage3 = self.cfgs[5:11]
        self.stage4 = self.cfgs[11:]
        self.scale = scale

        input_channel = _make_divisible(16 * scale)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            HardSwish()
        ) # input layer

        block = DepSepConv
        stage_names = ["stage2", "stage3", "stage4"]
        
        for idx, stage in enumerate([self.stage2, self.stage3, self.stage4]):
            layers = []
            for k, c, s, use_se in stage:
                output_channel = _make_divisible(c * scale)
                layers.append(block(input_channel, output_channel, k, s, use_se))
                input_channel = output_channel
            setattr(self, stage_names[idx], nn.Sequential(*layers))

        # self.features = nn.Sequential(*layers)

        # # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Conv2d(input_channel, 1280, 1, 1, 0)
        self.hwish = HardSwish()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        # x = self.features(x)
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)
        # return print(C1.size(), C2.size(), C3.size())
        return C2, C3

        # last several layers
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.hwish(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def PPLCNet_x0_25(**kwargs):
    """
    Constructs PPLCNet_x0_25 model
    """
    model = PPLCNet(scale=0.25, **kwargs)

    return model


def PPLCNet_x0_35(**kwargs):
    """
    Constructs PPLCNet_x0_35 model
    """
    model = PPLCNet(scale=0.35, **kwargs)

    return model


def PPLCNet_x0_5(**kwargs):
    """
    Constructs PPLCNet_x0_5 model
    """
    model = PPLCNet(scale=0.5, **kwargs)

    return model


def PPLCNet_x0_75(**kwargs):
    """
    Constructs PPLCNet_x0_75 model
    """
    model = PPLCNet(scale=0.75, **kwargs)

    return model


def PPLCNet_x1_0(**kwargs):
    """
    Constructs PPLCNet_x1_0 model
    """
    model = PPLCNet(scale=1.0, **kwargs)

    return model


def PPLCNet_x1_5(**kwargs):
    """
    Constructs PPLCNet_x1_5 model
    """
    model = PPLCNet(scale=1.5, **kwargs)

    return model


def PPLCNet_x2_0(**kwargs):
    """
    Constructs PPLCNet_x2_0 model
    """
    model = PPLCNet(scale=2.0, **kwargs)

    return model


def PPLCNet_x2_5(**kwargs):
    """
    Constructs PPLCNet_x2_5 model
    """
    model = PPLCNet(scale=2.5, **kwargs)

    return model


if __name__ == "__main__":
    model = PPLCNet_x0_5()
    sample = torch.rand([1, 3, 320, 320])
    out = model(sample)
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    stat(model, (3, 320, 320))
