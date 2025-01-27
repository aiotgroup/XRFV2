import torch
import torch.nn as nn
import logging
from model.models import register_model, make_backbone, make_backbone_config
from model.head import ClsLocHead
from model.TAD.embedding import Embedding
from model.model_config import TAD_single_Config
from model.embedding import TADEmbedding, TADEmbedding_pure, NoneEmbedding
from model.fusion import GatedFusion, GatedFusionAdd, GatedFusionWeight, GatedFusionAdd2


logger = logging.getLogger(__name__)

@register_model('TAD_muti_pre')
class TAD_muti_pre(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_pre, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusion(hidden_size=config.out_channels)

        logger.info(f'load {config.embed_type} embedding')
        logger.info(f'load {config.backbone_name}')
        backbone_config = make_backbone_config(config.backbone_name, cfg=config.backbone_config)
        self.backbone = make_backbone(config.backbone_name, backbone_config)
        self.modality = config.modality
        self.head = ClsLocHead(num_classes=config.num_classes, head_layer=config.head_num)
        self.priors = []
        t = config.priors
        for i in range(config.head_num):
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2
        self.num_classes = config.num_classes

    def forward(self, input):
        x_imu = input['imu']
        x_wifi = input['wifi']
        B, C, L = x_imu.size()
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 30, 2048]) torch.Size([4, 270, 2048])
        x_imu = self.embedding_imu(x_imu)
        x_wifi = self.embedding_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = self.fusion(x_imu, x_wifi)
        x = self.embedding(x)
        # print(x.shape)        torch.Size([4, 512, 256])
        feats = self.backbone(x)

        # for f in feats:
        #     print(f.shape)

        out_offsets, out_cls_logits = self.head(feats)
        priors = torch.cat(self.priors, 0).to(x.device).unsqueeze(0)
        loc = torch.cat([o.view(B, -1, 2) for o in out_offsets], 1)
        conf = torch.cat([o.view(B, -1, self.num_classes) for o in out_cls_logits], 1)

        # print(priors.shape, loc.shape, conf.shape)

        return {
            'loc': loc,
            'conf': conf,
            'priors': priors # trainer ddp需要弄成priors[0]
        }

def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data
