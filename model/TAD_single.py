import torch
import torch.nn as nn
import logging
from model.models import register_model, make_backbone, make_backbone_config
from model.head import ClsLocHead
from model.TAD.embedding import Embedding
from model.model_config import TAD_single_Config
from model.embedding import TADEmbedding

logger = logging.getLogger(__name__)

@register_model('TAD_single')
class TAD_single(nn.Module):
    def __init__(self, config: TAD_single_Config):
        super(TAD_single, self).__init__()
        self.config = config
        if config.embed_type == 'Norm':
            self.embedding = Embedding(config.in_channels, stride=config.embedding_stride)
        else:
            self.embedding = TADEmbedding(config.in_channels, out_channels=512, layer=3, input_length=config.input_length)

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
        x = input[self.modality]
        B, C, L = x.size()

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

def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data

if __name__ == '__main__':

    cfg = {
        "model": {
            "name": "TAD",
            "backbone_name": "mamba",
            "modality": "imu",
            "in_channels": 30,
            "backbone_config": None
        }
    }
    from dataset.wwadl import WWADLDatasetSingle, detection_collate
    from torch.utils.data import DataLoader
    train_dataset = WWADLDatasetSingle('/root/shared-nvme/dataset/all_30_3', split='train', modality='imu')

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
    model = make_model('TAD_single', model_cfg).to('cuda')

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
