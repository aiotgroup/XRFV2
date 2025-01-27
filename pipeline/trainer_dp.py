import logging

import torch
import random
import os.path
import numpy as np
from tqdm import tqdm
from torch import nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from strategy.loss.loss import MultiSegmentLoss
import heapq  # 用于管理最小堆
from dataset.wwadl import detection_collate

GLOBAL_SEED = 42

logger = logging.getLogger(__name__)

# Worker 初始化函数
def worker_init_fn(worker_id):
    np.random.seed(GLOBAL_SEED + worker_id)

def register_hooks(model):
    for name, module in model.named_modules():
        # 仅对实际计算的模块注册钩子
        if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.Identity)):
            module.register_forward_hook(forward_hook(name))

def _to_var(data: dict, device):
    for key, value in data.items():
        data[key] = value.to(device)  # Directly move tensor to device
    return data


def forward_hook(module_name):
    """
    钩子函数，用于检查输出是否合法。
    """
    def hook(module, input, output):
        # 检查输出是否是 Tensor 类型
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                logging.error(f"NaN or Inf detected in module: {module_name}")
                raise RuntimeError(f"NaN or Inf detected in module: {module_name}")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        logging.error(f"NaN or Inf detected in module: {module_name}, output[{i}]")
                        raise RuntimeError(f"NaN or Inf detected in module: {module_name}, output[{i}]")
        # 如果输出是 tuple 或 list，逐个检查其中的元素
        elif isinstance(output, dict):
            for key, out in output.items():
                if isinstance(out, torch.Tensor):
                    if torch.isnan(out).any() or torch.isinf(out).any():
                        logging.error(f"NaN or Inf detected in module: {module_name}, key: {key}")
                        raise RuntimeError(f"NaN or Inf detected in module: {module_name}, key: {key}")
        else:
            logging.warning(f"Output of module: {module_name} is not a Tensor or tuple/list of Tensors. Skipping check.")
    return hook

class BestModelSaver:
    def __init__(self, check_point_path, max_models=10):
        self.check_point_path = check_point_path
        self.max_models = max_models
        # 使用最大堆保存模型信息 [(负的metric, model_path), ...]
        self.best_models = []

    def save_model(self, model_state_dict, model_name, metric, is_save=False):
        # 构造保存路径
        model_path = os.path.join(self.check_point_path, f"{model_name}.pt")

        if is_save:
            torch.save(model_state_dict, model_path)
            return

        # 如果队列未满，直接保存模型
        if len(self.best_models) < self.max_models:
            torch.save(model_state_dict, model_path)
            # 保存负的metric以构造最大堆
            heapq.heappush(self.best_models, (-metric, model_path))
            print(f"Model saved: {model_path} (Metric: {metric:.5f})")
        else:
            # 检查是否优于当前最差模型（堆顶是负的最大值，对应正的最小值）
            if metric < -self.best_models[0][0]:  # 假设指标是损失，越小越好
                # 删除最差模型
                _, worst_model_path = heapq.heappop(self.best_models)
                if os.path.exists(worst_model_path):
                    os.remove(worst_model_path)
                    print(f"Old model removed: {worst_model_path}")

                # 保存新模型
                torch.save(model_state_dict, model_path)
                heapq.heappush(self.best_models, (-metric, model_path))
                print(f"Model saved: {model_path} (Metric: {metric:.5f})")
            else:
                print(f"Model not saved. Metric: {metric:.5f} is worse than the top 10.")

    def get_best_models(self):
        # 返回按指标从小到大排序的模型列表（还原负的metric）
        return sorted([(-metric, path) for metric, path in self.best_models], key=lambda x: x[0])

class Trainer(object):
    def __init__(self,
                 config,
                 train_dataset,
                 model
                 ):
        super(Trainer, self).__init__()
        self.model = model
        self.train_dataset = train_dataset

        training_config = config['training']

        self.batch_size = training_config['train_batch_size']
        self.num_epoch = training_config['num_epoch']

        # loss setting -----------------------------------------------------------
        self.loss = MultiSegmentLoss(num_classes=config['model']['num_classes'], clip_length=config['dataset']['clip_length'])
        self.lw = config['loss']['lw']
        self.cw = config['loss']['cw']

        # learning config ---------------------------------------------------------
        self.opt_method = training_config['opt_method']
        self.lr_rate = training_config['lr_rate']
        self.lr_rate_adjust_epoch = training_config['lr_rate_adjust_epoch']
        self.lr_rate_adjust_factor = training_config['lr_rate_adjust_factor']
        self.weight_decay = training_config['weight_decay']


        self.check_point_path = config['path']['result_path']

        self.model_info = f'{config["model"]["backbone_name"]}_{config["model"]["model_set"]}'
        self.writer = SummaryWriter(os.path.join(self.check_point_path, f'tb_{self.model_info}'))

        # DDP setting -------------------------------------------------------------
        self.dist_url = 'env://'
        self.rank = 0
        self.world_size = 0
        self.gpu=0

        self.device = 'cuda'

    def _init_optimizer(self):

        params = self.model.parameters()

        if self.opt_method == 'adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr_rate,
                                              weight_decay=self.weight_decay)
        elif self.opt_method == 'adamw':
            self.optimizer = torch.optim.AdamW(params=params,
                                               lr=self.lr_rate,
                                               weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=params,
                                             lr=self.lr_rate,
                                             weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         self.lr_rate_adjust_epoch,
                                                         self.lr_rate_adjust_factor)

    def _train_one_step(self, data, targets):
        data = _to_var(data, self.device)
        # data = data.to(self.device)  # 确保输入数据在正确的设备上
        targets = [t.to(self.device) for t in targets]
        self.optimizer.zero_grad()


        try:
            output_dict = self.model(data)
            assert not torch.isnan(output_dict['loc']).any(), "NaN detected in output_dict['loc']"
        except AssertionError as e:
            logging.info("Error occurred during training, saving data and model parameters for debugging...")

            # 创建保存路径
            error_save_path = os.path.join(self.check_point_path, "debug")
            os.makedirs(error_save_path, exist_ok=True)

            # 保存导致问题的输入数据
            data_save_path = os.path.join(error_save_path, "error_data.pt")
            torch.save(data, data_save_path)
            logging.info(f"Input data saved to: {data_save_path}")

            # 保存模型参数
            model_save_path = os.path.join(error_save_path, "error_model.pth")
            torch.save(self.model.state_dict(), model_save_path)
            logging.info(f"Model parameters saved to: {model_save_path}")

            # 再次抛出异常以中断训练
            raise e

        loc_p = output_dict['loc'].clamp(min=0)
        loss_l, loss_c = self.loss([loc_p, output_dict['conf'], output_dict["priors"][0]], targets)

        assert not torch.isnan(loss_l).any(), "NaN detected in loss_l"
        assert not torch.isnan(loss_c).any(), "NaN detected in loss_c"

        loss_l = loss_l * self.lw * 100
        loss_c = loss_c * self.cw

        loss = loss_l + loss_c

        # 反向传播
        loss.backward()

        # 优化器更新权重
        self.optimizer.step()

        # 无需分布式同步，直接返回损失
        return loss.item(), loss_l.item(), loss_c.item()

    def training(self):
        # 给不同的进程分配不同的、固定的随机数种子
        self.set_seed(2024)
        register_hooks(self.model)
        device = torch.device(self.device)

        # dataset loader -------------------------------------------------------------------------------
        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        print('Using {} dataloader workers every process'.format(nw))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,  # DataParallel 不需要使用分布式采样
            pin_memory=True,
            num_workers=nw,  # 动态设置 workers
            collate_fn=detection_collate,  # 自定义 collate_fn
            worker_init_fn=worker_init_fn,  # 初始化每个 worker 的随机种子
            drop_last = True
        )

        # load model ------------------------------------------------------------------------------------

        # 转为DataParallel模型 ---------------------------------------------------------------------------
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))
        else:
            print("Using a single GPU for training")

        self.model = self.model.to(device=device)

        self._init_optimizer()

        mini_train_loss = float('inf')
        saver = BestModelSaver(self.check_point_path, max_models=1)  # 初始化最佳模型管理

        for epoch in range(self.num_epoch):
            np.random.seed(epoch)  # 设置随机种子
            self.model.train()

            tbar = tqdm(train_loader)

            iteration = 0
            loss_loc_val = 0
            loss_conf_val = 0
            cost_val = 0

            for clips, targets in tbar:
                iteration += 1
                loss, loss_l, loss_c = self._train_one_step(clips, targets)

                loss_loc_val += loss_l
                loss_conf_val += loss_c
                cost_val += loss

                tbar.set_description('Epoch: %d: ' % (epoch + 1))
                tbar.set_postfix(train_loss=loss)
                # 每次迭代清理显存
                torch.cuda.empty_cache()


            tbar.close()

            loss_loc_val /= (iteration + 1)
            loss_conf_val /= (iteration + 1)
            cost_val /= (iteration + 1)
            plog = 'Epoch-{} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}' \
                .format(epoch, cost_val, loss_loc_val, loss_conf_val)

            logging.info(plog)

            self.scheduler.step()

            if epoch == 49:
                saver.save_model(self.model.state_dict(), f"{self.model_info}_50-epoch-{epoch}", cost_val, is_save=True)
            
            if epoch == 64:
                saver.save_model(self.model.state_dict(), f"{self.model_info}_65-epoch-{epoch}", cost_val, is_save=True)

            # 保存当前模型
            saver.save_model(self.model.state_dict(), f"{self.model_info}-epoch-{epoch}", cost_val)

            self.writer.add_scalar("Train Loss", cost_val, epoch)
            self.writer.add_scalar("loss_loc_val Loss", loss_loc_val, epoch)
            self.writer.add_scalar("loss_conf_val Loss", loss_conf_val, epoch)



        # torch.save(self.model.state_dict(),
        #            os.path.join(self.check_point_path, '%s-final' % (self.model_info)))

        # if os.path.exists(os.path.join(self.check_point_path, "initial_weights.pt")) is True:
        #     os.remove(os.path.join(self.check_point_path, "initial_weights.pt"))


    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic =True