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

@register_model('TAD_muti_imu')
class TAD_muti_imu(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_imu, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(in_channels=512, out_channels=512, layer=3, input_length=config.input_length)

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
        x = x_imu
        # x = self.fusion(x_imu, x_wifi)
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

@register_model('TAD_muti_wifi')
class TAD_muti_wifi(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_wifi, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(in_channels=512, out_channels=512, layer=3, input_length=config.input_length)

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
        # x_imu = input['imu']
        x_wifi = input['wifi']
        B, C, L = x_wifi.size()
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 30, 2048]) torch.Size([4, 270, 2048])
        # x_imu = self.embedding_imu(x_imu)
        x_wifi = self.embedding_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = x_wifi
        # x = self.fusion(x_imu, x_wifi)
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

@register_model('TAD_muti_wifi2')
class TAD_muti_wifi2(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_wifi2, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        # if config.embed_type == 'Norm':
        #     self.embedding = NoneEmbedding()
        # else:
        #     self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.embedding = TADEmbedding(config.wifi_in_channels, out_channels=512, layer=3, input_length=config.input_length)

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
        # x_imu = input['imu']
        x_wifi = input['wifi']
        B, C, L = x_wifi.size()
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 30, 2048]) torch.Size([4, 270, 2048])
        # x_imu = self.embedding_imu(x_imu)
        x_wifi = self.embedding(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = x_wifi
        # x = self.fusion(x_imu, x_wifi)
        # x = self.embedding(x)
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

@register_model('TAD_muti_add')
class TAD_muti_add(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_add, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusionAdd(hidden_size=config.out_channels)

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

@register_model('TAD_muti_add2')
class TAD_muti_add2(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_add2, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusionAdd2(hidden_size=config.input_length)

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

@register_model('TAD_muti_weight')
class TAD_muti_weight(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_weight, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        if config.embed_type == 'Norm':
            self.embedding = NoneEmbedding()
        else:
            self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusionWeight(hidden_size=config.out_channels)

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

@register_model('TAD_muti_tsse')
class TAD_muti_tsse(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_tsse, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        # if config.embed_type == 'Norm':
        #     self.embedding = NoneEmbedding()
        # else:
        #     self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.embedding_tsse_imu = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)
        self.embedding_tsse_wifi = TADEmbedding_pure(config.wifi_in_channels, out_channels=512, layer=3, input_length=config.input_length)

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

        x_imu = self.embedding_tsse_imu(x_imu)
        x_wifi = self.embedding_tsse_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = self.fusion(x_imu, x_wifi)
        # x = self.embedding(x)
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


@register_model('TAD_muti_tsse_add')
class TAD_muti_tsse_add(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_tsse_add, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        # if config.embed_type == 'Norm':
        #     self.embedding = NoneEmbedding()
        # else:
        #     self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.embedding_tsse_imu = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)
        self.embedding_tsse_wifi = TADEmbedding_pure(config.wifi_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusionAdd2(hidden_size=256)

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

        x_imu = self.embedding_tsse_imu(x_imu)
        x_wifi = self.embedding_tsse_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = self.fusion(x_imu, x_wifi)
        # x = self.embedding(x)
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

@register_model('TAD_muti_tsse_weight')
class TAD_muti_tsse_weight(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_muti_tsse_weight, self).__init__()
        self.config = config
        self.embedding_imu = Embedding(config.imu_in_channels, stride=1)
        self.embedding_wifi = Embedding(config.wifi_in_channels, stride=1)

        # if config.embed_type == 'Norm':
        #     self.embedding = NoneEmbedding()
        # else:
        #     self.embedding = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.embedding_tsse_imu = TADEmbedding_pure(config.imu_in_channels, out_channels=512, layer=3, input_length=config.input_length)
        self.embedding_tsse_wifi = TADEmbedding_pure(config.wifi_in_channels, out_channels=512, layer=3, input_length=config.input_length)

        self.fusion = GatedFusionWeight(hidden_size=config.out_channels)

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

        x_imu = self.embedding_tsse_imu(x_imu)
        x_wifi = self.embedding_tsse_wifi(x_wifi)
        # print(x_imu.shape, x_wifi.shape)  torch.Size([4, 512, 256]) torch.Size([4, 512, 256])
        x = self.fusion(x_imu, x_wifi)
        # x = self.embedding(x)
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

if __name__ == '__main__':

    cfg = {
        "model": {
            "name": "TAD_muti_pre",
            "backbone_name": "mamba",
            "modality": "imu",
            "in_channels": 30,
            "backbone_config": None
        }
    }
    from dataset.wwadl import WWADLDatasetSingle, detection_collate
    from dataset.wwadl_muti import WWADLDatasetMuti
    from torch.utils.data import DataLoader
    # train_dataset = WWADLDatasetSingle('/root/shared-nvme/dataset/all_30_3', split='train', modality='imu')
    train_dataset = WWADLDatasetMuti('/root/shared-nvme/dataset/all_30_3')
    batch_size = 4
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True
    )

    from model.models import make_model, make_model_config
    model_cfg = make_model_config(cfg['model']['backbone_name'], cfg['model'])
    model = make_model('TAD_muti_pre', model_cfg).to('cuda')

    print(model.config.get_dict())

    for i, (data_batch, label_batch) in enumerate(train_data_loader):
        print(f"Batch {i} labels: {len(label_batch)}")
        data_batch = _to_var(data_batch, 'cuda')
        output = model(data_batch)

        break


    # tad_config = TAD_single_Config(cfg)
    # print(tad_config.get_dict())
    # backbone_config = Mamba_config(tad_config.backbone_config)
    # print(backbone_config.get_dict())
