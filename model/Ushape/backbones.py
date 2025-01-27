import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Ushape.unetpp import UNetPP
from model.Ushape.unet import UNet



class UNetBackbone(nn.Module):

    def __init__(self, in_channels = 512, branch_layer=4, filters = None, deep_supervision = True, layers=3, branch_layers=3):
        super().__init__()

        self.layers = layers
        self.branch_layers = branch_layers

        if filters is None:
            filters = [128, 256, 512, 1025, 2048, 4096]

        # stem network
        self.stem = UNetPP(in_channels=in_channels, filters=filters, deep_supervision=True)

        # main branch with pooling
        self.branch = nn.ModuleList()
        for idx in range(branch_layer):
            self.branch.append(
                UNetPP(in_channels=filters[0], filters=filters, deep_supervision=True)
            )
        
        self.branch_pooling = nn.ModuleList()
        for idx in range(branch_layer):
            self.branch_pooling.append(
                nn.MaxPool1d(kernel_size=2)
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)
    

    def forward(self, x):
        B, C, T = x.size()

        # feature projection
        x = self.stem(x, L=self.layers)
        # prep for outputs
        out_feats = (x, )
        for idx in range(len(self.branch)):
            x = self.branch[idx](self.branch_pooling[idx](x), L=self.branch_layers)
            out_feats += (x, )
        return out_feats
    

class UNetBackbone2(nn.Module):

    def __init__(self, in_channels = 64, branch_layer=4, layers=3, unet_branch_layers=2):
        super().__init__()
        # stem network
        self.stem =  UNet(in_channel=in_channels, out_channel=in_channels, depth=layers)

        # main branch with pooling
        self.branch = nn.ModuleList()
        for idx in range(branch_layer):
            self.branch.append(
                UNet(in_channel=in_channels, out_channel=in_channels, depth=unet_branch_layers)
            )
        
        self.branch_pooling = nn.ModuleList()
        for idx in range(branch_layer):
            self.branch_pooling.append(
                nn.MaxPool1d(kernel_size=2)
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)
    

    def forward(self, x):
        B, C, T = x.size()

        # feature projection
        x = self.stem(x)
        # prep for outputs
        out_feats = (x, )
        for idx in range(len(self.branch)):
            x = self.branch[idx](self.branch_pooling[idx](x))
            out_feats += (x, )
        return out_feats