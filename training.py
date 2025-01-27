
import torch
import logging
from init_utils import init_dataset, init_model
from torchinfo import summary
from pipeline.trainer_ddp import Trainer as Trainer_ddp
from pipeline.trainer_dp import Trainer as Trainer_dp
import model
from model.models import make_model, make_model_config
from utils.setting import write_setting
logger = logging.getLogger(__name__)

def count_gflops(model, input_size):
    batch_data = torch.randn(input_size)
    model_stats = summary(model, input_data=batch_data, verbose=0)
    return model_stats.total_mult_adds / 1e9  # 转换为 GFLOPs

def train(config, type = 'dp'):
    # 1. get_dataset
    train_dataset = init_dataset(config)

    model_cfg = make_model_config(config['model']['backbone_name'], config['model'])
    logger.info(f"Initializing model with backbone: {config['model']['backbone_name']} ...")
    model = make_model(config['model']['name'], model_cfg)
    logger.info(f"Model {config['model']['name']} initialized successfully.")
    config['model'] = model_cfg.get_dict()
    write_setting(config)
    log_info = 'model params: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad))

    logger.info(log_info)

    # backbone_gflops = count_gflops(strategy.backbone, (64, 90, 1000))
    # print(f'Backbone GFLOPs: {backbone_gflops}')
    if type == 'dp':
        trainer = Trainer_dp(config, train_dataset, model)
    else:
        trainer = Trainer_ddp(config, train_dataset, model)

    trainer.training()





