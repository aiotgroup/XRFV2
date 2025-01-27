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

@register_model('TAD_muti_none')
class TAD_muti_none(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_none, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, out_channels=config.out_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, out_channels=config.out_channels, stride=1)

        assert config.embed_type in ['None', 'Down']

        if config.embed_type == 'None':
            self.embedding = NoneEmbedding()
        elif config.embed_type == 'Down':
            self.embedding = Embedding(config.out_channels, out_channels=config.out_channels, stride=2)

        self.fusion = GatedFusion(hidden_size=config.out_channels)

        logger.info(f'load {config.embed_type} embedding')
        logger.info(f'load {config.backbone_name}')
        backbone_config = make_backbone_config(config.backbone_name, cfg=config.backbone_config)
        self.backbone = make_backbone(config.backbone_name, backbone_config)
        self.modality = config.modality
        self.head = ClsLocHead(num_classes=config.num_classes, head_layer=config.head_num, in_channel=config.out_channels)
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

        x_imu = self.embedding_imu(x_imu)
        x_wifi = self.embedding_wifi(x_wifi)

        x = self.fusion(x_imu, x_wifi)
        x = self.embedding(x)

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

