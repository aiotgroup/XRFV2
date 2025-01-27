import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        return x


class TranConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.deconv1(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.depth = depth
        self.model = nn.ModuleList([])
        self.deconvs = nn.ModuleList([])
        self.demodel = nn.ModuleList([])
        self.pool = nn.MaxPool1d(2, stride=2)

        for i in range(depth):
            block = ConvBlock(self.in_channel, self.out_channel)
            self.in_channel = self.out_channel
            self.out_channel = self.out_channel * 2
            self.model.append(block)
            if i != (depth-1):
                self.model.append(self.pool)

        self.out_channel = self.out_channel // 2

        for i in range(depth-1):
            deconv = TranConvBlock(self.out_channel, self.out_channel//2)
            conv = ConvBlock(self.out_channel, self.out_channel//2)
            
            self.out_channel = self.out_channel // 2
            self.deconvs.append(deconv)
            self.demodel.append(conv)


    def forward(self, x):
        outputs = []
        for i in range(len(self.model)):
            out = self.model[i](x)
            x = out
            if i % 2 == 0:
                outputs.append(out)
        length = len(self.deconvs)
        deinput = outputs[-1]
        out_end = None
        for i in range(length):
            deout = self.deconvs[i](deinput)
            if deout.shape[2] != outputs[len(outputs)-2-i].shape[2]:
                deout = F.interpolate(deout, outputs[len(outputs)-2-i].shape[2], mode='linear', align_corners=True)
            out_concat = torch.cat((deout, outputs[len(outputs)-2-i]), dim=1)
            out_end = self.demodel[i](out_concat)
            deinput = out_end
        return out_end
