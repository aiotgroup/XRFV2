import torch.nn as nn
from utils.basic_config import Config
from strategy.loss.loss import MultiSegmentLoss


class TAD(nn.Module):
    def __init__(self, backbone, loss):
        super(TAD, self).__init__()
        self.backbone = backbone
        self.loss = loss

    # def forward(self, input):
    #     output_dict = self.backbone(input)
    #     loss_l, loss_c = self.loss([output_dict['loc'], output_dict['conf'], output_dict["priors"][0]], targets)
