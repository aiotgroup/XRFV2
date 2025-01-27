import torch
import torch.nn as nn
import torch.nn.functional as F


def UnitSize(source, target):
    source = F.interpolate(source, target.shape[2], mode='linear', align_corners=True)
    return source


class ConvSamePad1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
        super().__init__()

        left_top_pad = right_bottom_pad = kernel_size // 2
        if kernel_size % 2 == 0:
            right_bottom_pad -= 1

        self.layer = nn.Sequential(
            nn.ReflectionPad1d((left_top_pad, right_bottom_pad)),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class StandardUnit(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            ConvSamePad1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.Dropout(p=drop_rate),
            ConvSamePad1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, inputs):
        return self.layer(inputs)


class UNetPP(nn.Module):
    def __init__(self, in_channels, filters=None, deep_supervision=True, layers=3):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.layers = layers
        if filters is None:
            filters = [64, 128, 256, 512, 1024, 2048]
        self.x_00 = StandardUnit(in_channels=in_channels, out_channels=filters[0])
        self.pool0 = nn.MaxPool1d(kernel_size=2)

        self.x_01 = StandardUnit(in_channels=filters[0] * 2, out_channels=filters[0])
        self.x_02 = StandardUnit(in_channels=filters[0] * 3, out_channels=filters[0])
        self.x_03 = StandardUnit(in_channels=filters[0] * 4, out_channels=filters[0])
        self.x_04 = StandardUnit(in_channels=filters[0] * 5, out_channels=filters[0])
        self.x_05 = StandardUnit(in_channels=filters[0] * 6, out_channels=filters[0])

        self.up_10_to_01 = nn.ConvTranspose1d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_11_to_02 = nn.ConvTranspose1d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_12_to_03 = nn.ConvTranspose1d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_13_to_04 = nn.ConvTranspose1d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)
        self.up_14_to_05 = nn.ConvTranspose1d(in_channels=filters[1], out_channels=filters[0], kernel_size=2, stride=2)

        self.x_10 = StandardUnit(in_channels=filters[0], out_channels=filters[1])
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.x_11 = StandardUnit(in_channels=filters[1] * 2, out_channels=filters[1])
        self.x_12 = StandardUnit(in_channels=filters[1] * 3, out_channels=filters[1])
        self.x_13 = StandardUnit(in_channels=filters[1] * 4, out_channels=filters[1])
        self.x_14 = StandardUnit(in_channels=filters[1] * 5, out_channels=filters[1])

        self.up_20_to_11 = nn.ConvTranspose1d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)
        self.up_21_to_12 = nn.ConvTranspose1d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)
        self.up_22_to_13 = nn.ConvTranspose1d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)
        self.up_23_to_14 = nn.ConvTranspose1d(in_channels=filters[2], out_channels=filters[1], kernel_size=2, stride=2)

        self.x_20 = StandardUnit(in_channels=filters[1], out_channels=filters[2])
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.x_21 = StandardUnit(in_channels=filters[2] * 2, out_channels=filters[2])
        self.x_22 = StandardUnit(in_channels=filters[2] * 3, out_channels=filters[2])
        self.x_23 = StandardUnit(in_channels=filters[2] * 4, out_channels=filters[2])

        self.up_30_to_21 = nn.ConvTranspose1d(in_channels=filters[3], out_channels=filters[2], kernel_size=2, stride=2)
        self.up_31_to_22 = nn.ConvTranspose1d(in_channels=filters[3], out_channels=filters[2], kernel_size=2, stride=2)
        self.up_32_to_23 = nn.ConvTranspose1d(in_channels=filters[3], out_channels=filters[2], kernel_size=2, stride=2)

        self.x_30 = StandardUnit(in_channels=filters[2], out_channels=filters[3])
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.x_31 = StandardUnit(in_channels=filters[3] * 2, out_channels=filters[3])
        self.x_32 = StandardUnit(in_channels=filters[3] * 3, out_channels=filters[3])

        self.up_40_to_31 = nn.ConvTranspose1d(in_channels=filters[4], out_channels=filters[3], kernel_size=2, stride=2)
        self.up_41_to_32 = nn.ConvTranspose1d(in_channels=filters[4], out_channels=filters[3], kernel_size=2, stride=2)

        self.x_40 = StandardUnit(in_channels=filters[3], out_channels=filters[4])
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.x_41 = StandardUnit(in_channels=filters[4] * 2, out_channels=filters[4])

        self.up_50_to_41 = nn.ConvTranspose1d(in_channels=filters[5], out_channels=filters[4], kernel_size=2, stride=2)

        self.x_50 = StandardUnit(in_channels=filters[4], out_channels=filters[5])

    def forward(self, inputs, L=4):
        if not (1 <= L <= 5):
            raise ValueError("the model pruning factor `L` should be 1 <= L <= 5")
        # torch.Size([4, 512, 256])                    
        x_00_output = self.x_00(inputs)                     # torch.Size([4, 64, 256])

        x_10_output = self.x_10(self.pool0(x_00_output))    # torch.Size([4, 128, 128])
        x_10_up_sample = self.up_10_to_01(x_10_output)      # torch.Size([4, 64, 256])
        if x_10_up_sample.shape[2] != x_00_output.shape[2]:
            x_10_up_sample = UnitSize(x_10_up_sample, x_00_output)
        x_01_output = self.x_01(torch.cat([x_00_output, x_10_up_sample], 1))    # torch.Size([4, 64, 256])

        if L == 1:
            return x_01_output

        x_20_output = self.x_20(self.pool1(x_10_output))    # torch.Size([4, 256, 64])
        x_20_up_sample = self.up_20_to_11(x_20_output)      # torch.Size([4, 128, 128])
        if x_20_up_sample.shape[2] != x_10_output.shape[2]:
            x_20_up_sample = UnitSize(x_20_up_sample, x_10_output)
        x_11_output = self.x_11(torch.cat([x_10_output, x_20_up_sample], 1))    # torch.Size([4, 128, 128])
        x_11_up_sample = self.up_11_to_02(x_11_output)      # torch.Size([4, 64, 256])
        if x_11_up_sample.shape[2] != x_00_output.shape[2]:
            x_11_up_sample = UnitSize(x_11_up_sample, x_00_output)
        x_02_output = self.x_02(torch.cat([x_00_output, x_01_output, x_11_up_sample], 1))   # torch.Size([4, 64, 256])

        if L == 2:
            if self.deep_supervision:
                return x_01_output + x_02_output
            else:
                return x_02_output

        x_30_output = self.x_30(self.pool2(x_20_output))
        x_30_up_sample = self.up_30_to_21(x_30_output)
        if x_30_up_sample.shape[2] != x_20_output.shape[2]:
            x_30_up_sample = UnitSize(x_30_up_sample, x_20_output)
        x_21_output = self.x_21(torch.cat([x_20_output, x_30_up_sample], 1))
        x_21_up_sample = self.up_21_to_12(x_21_output)
        if x_21_up_sample.shape[2] != x_10_output.shape[2]:
            x_21_up_sample = UnitSize(x_21_up_sample, x_10_output)
        x_12_output = self.x_12(torch.cat([x_10_output, x_11_output, x_21_up_sample], 1))
        x_12_up_sample = self.up_12_to_03(x_12_output)
        if x_12_up_sample.shape[2] != x_00_output.shape[2]:
            x_12_up_sample = UnitSize(x_12_up_sample, x_00_output)
        x_03_output = self.x_03(torch.cat([x_00_output, x_01_output, x_02_output, x_12_up_sample], 1))

        if L == 3:
            if self.deep_supervision:
                
                return x_01_output + x_02_output + x_03_output
            else:
                return x_03_output

        x_40_output = self.x_40(self.pool3(x_30_output))
        x_40_up_sample = self.up_40_to_31(x_40_output)
        if x_40_up_sample.shape[2] != x_30_output.shape[2]:
            x_40_up_sample = UnitSize(x_40_up_sample, x_30_output)
        x_31_output = self.x_31(torch.cat([x_30_output, x_40_up_sample], 1))
        x_31_up_sample = self.up_31_to_22(x_31_output)
        if x_31_up_sample.shape[2] != x_20_output.shape[2]:
            x_31_up_sample = UnitSize(x_31_up_sample, x_20_output)
        x_22_output = self.x_22(torch.cat([x_20_output, x_21_output, x_31_up_sample], 1))
        x_22_up_sample = self.up_22_to_13(x_22_output)
        if x_22_up_sample.shape[2] != x_10_output.shape[2]:
            x_22_up_sample = UnitSize(x_22_up_sample, x_10_output)

        x_13_output = self.x_13(torch.cat([x_10_output, x_11_output, x_12_output, x_22_up_sample], 1))
        x_13_up_sample = self.up_13_to_04(x_13_output)
        if x_13_up_sample.shape[2] != x_00_output.shape[2]:
            x_13_up_sample = UnitSize(x_13_up_sample, x_00_output)
        x_04_output = self.x_04(torch.cat([x_00_output, x_01_output, x_02_output, x_03_output, x_13_up_sample], 1))
        if L == 4:
            if self.deep_supervision:
                return x_01_output + x_02_output + x_03_output + x_04_output
            else:
                return x_04_output

        x_50_output = self.x_50(self.pool4(x_40_output))
        x_50_up_sample = self.up_50_to_41(x_50_output)
        if x_50_up_sample.shape[2] != x_40_output.shape[2]:
            x_50_up_sample = UnitSize(x_50_up_sample, x_40_output)
        x_41_output = self.x_41(torch.cat([x_40_output, x_50_up_sample], 1))
        x_41_up_sample = self.up_41_to_32(x_41_output)
        if x_41_up_sample.shape[2] != x_30_output.shape[2]:
            x_41_up_sample = UnitSize(x_41_up_sample, x_30_output)
        x_32_output = self.x_32(torch.cat([x_30_output, x_31_output, x_41_up_sample], 1))
        x_32_up_sample = self.up_32_to_23(x_32_output)
        if x_32_up_sample.shape[2] != x_20_output.shape[2]:
            x_32_up_sample = UnitSize(x_32_up_sample, x_20_output)
        x_23_output = self.x_23(torch.cat([x_20_output, x_21_output, x_22_output, x_32_up_sample], 1))
        x_23_up_sample = self.up_23_to_14(x_23_output)
        if x_23_up_sample.shape[2] != x_10_output.shape[2]:
            x_23_up_sample = UnitSize(x_23_up_sample, x_10_output)
        x_14_output = self.x_14(torch.cat([x_10_output, x_11_output, x_12_output, x_13_output, x_23_up_sample], 1))
        x_14_up_sample = self.up_14_to_05(x_14_output)
        if x_14_up_sample.shape[2] != x_00_output.shape[2]:
            x_14_up_sample = UnitSize(x_14_up_sample, x_00_output)
        x_05_output = self.x_05(
            torch.cat([x_00_output, x_01_output, x_02_output, x_03_output, x_04_output, x_14_up_sample], 1))

        if L == 5:
            if self.deep_supervision:
                return x_01_output + x_02_output + x_03_output + x_04_output + x_05_output
            else:
                return x_05_output