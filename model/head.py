import torch
import torch.nn as nn
from model.TAD.head import PredictionHead
from model.TAD.module import ScaleExp
import torch.nn.init as init

def init_weights(module):
    """
    Initialize weights for different layers in the model.
    """
    if isinstance(module, nn.Conv1d):  # For Conv1d layers (e.g., Unit1D is likely based on Conv1d)
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')  # He initialization
        if module.bias is not None:
            init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, nn.Linear):  # For Linear layers
        init.xavier_normal_(module.weight)  # Xavier initialization for Linear layers
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.GroupNorm):  # GroupNorm layers
        init.ones_(module.weight)  # Initialize scale to 1
        init.zeros_(module.bias)   # Initialize bias to 0

class ClsLocHead(nn.Module):
    def __init__(self, num_classes, head_layer, in_channel=512):
        super(ClsLocHead, self).__init__()
        self.num_classes = num_classes
        self.loc_heads = nn.ModuleList()
        for i in range(head_layer):
            self.loc_heads.append(ScaleExp())
        self.PredictionHead = PredictionHead(in_channel=in_channel, num_classes=num_classes)  # Assuming PredictionHead is defined elsewhere
        # Apply weight initialization to all submodules
        self.apply(init_weights)

    def forward(self, fpn_feats):
        out_offsets = []
        out_cls_logits = []

        # Process each feature map from the FPN
        for i, feat in enumerate(fpn_feats):
            assert not torch.isnan(feat).any(), "NaN detected in loc_logits before PredictionHead"
            loc_logits, conf_logits = self.PredictionHead(feat)
            assert not torch.isnan(loc_logits).any(), f"NaN detected in loc_logits at layer {i}"
            # Apply the corresponding loc_head to the loc_logits
            out_offsets.append(
                self.loc_heads[i](loc_logits)
                .view(feat.size(0), 2, -1)  # B, 2, N
                .permute(0, 2, 1)  # B, N, 2
                .contiguous()
            )

            # Append the classification logits
            out_cls_logits.append(
                conf_logits.view(feat.size(0), self.num_classes, -1)  # B, num_classes, N
                .permute(0, 2, 1)  # B, N, num_classes
                .contiguous()
            )

        # Return the processed outputs
        return out_offsets, out_cls_logits