import torch
import torch.nn as nn
from model.TAD.module import Unit1D

import torch.nn.init as init

class Tower(nn.Module):
    def __init__(self, out_channels, layer):
        super().__init__()
        
        conf_towers = [] 
        for i in range(layer):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.conf_tower = nn.Sequential(*conf_towers)

    def forward(self, x):
        return self.conf_tower(x)


class conf_head(nn.Module):
    def __init__(self, out_channels=512, num_classes=34):
        super().__init__()
        self.conf = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

    def forward(self, x):
        x = self.conf(x)
        return x


class loc_head(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.loc =nn.Sequential(Unit1D(
                in_channels=out_channels,
                output_channels=out_channels,
                kernel_shape=3,
                stride=1,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(
                    in_channels=out_channels,
                    output_channels=2,
                    kernel_shape=3,
                    stride=1,
                    use_bias=True,
                    activation_fn=None
            )
        )

    def forward(self, x):
        x = self.loc(x)
        return x

def init_weights(module):
    """
    Initialize weights for different layers in the model.
    """
    if isinstance(module, nn.Conv1d):  # For Conv1d layers (Unit1D is likely based on Conv1d)
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # He initialization for ReLU
        if module.bias is not None:
            init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, nn.Linear):  # For Linear layers, if there are any
        init.xavier_normal_(module.weight)  # Xavier initialization for Linear layers
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):  # GroupNorm layers
        init.ones_(module.weight)  # Initialize scale to 1
        init.zeros_(module.bias)   # Initialize bias to 0

class PredictionHead(nn.Module):
    def __init__(self, in_channel=512, num_classes=34):
        super().__init__()
        self.loc_tower = Tower(in_channel, 3)
        self.conf_tower = Tower(in_channel, 3)
        
        self.loc_head = loc_head(out_channels=in_channel)
        self.conf_head = conf_head(out_channels=in_channel, num_classes=num_classes)

        # Apply the weight initialization to all layers in the module
        self.apply(init_weights)
        
    def forward(self, x):
        # 获取特征
        loc_feat = self.loc_tower(x)
        conf_feat = self.conf_tower(x)

        # 通过预测头
        loc_feat = self.loc_head(loc_feat)
        conf_feat = self.conf_head(conf_feat)

        return loc_feat, conf_feat