import json
import os
import h5py
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset.modality.wifi import WWADL_wifi
from dataset.modality.imu import WWADL_imu
from dataset.modality.airpods import WWADL_airpods
from dataset.modality.base import handle_nan_and_interpolate
from dataset.action import id_to_action


def load_file_list(dataset_path):
    # 读取 test.csv
    test_csv_path = os.path.join(dataset_path, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"{test_csv_path} does not exist.")

    print("Loading test.csv...")
    test_df = pd.read_csv(test_csv_path)
    file_name_list = test_df['file_name'].tolist()
    print(f"Loaded {len(file_name_list)} file names from test.csv.")

    return file_name_list

class WWADLDatasetTestSingle():

    def __init__(self, config, modality=None, device_keep_list=None):
        # 初始化路径配置
        self.dataset_dir = config['path']['dataset_path']
        dataset_root_path = config['path']['dataset_root_path']
        self.test_file_list = load_file_list(self.dataset_dir)

        # 读取info.json文件
        self.info_path = os.path.join(self.dataset_dir, "info.json")
        with open(self.info_path, 'r') as json_file:
            self.info = json.load(json_file)

        if modality is None:
            assert len(self.info['modality_list']) == 1, "single modality"
            self.modality = self.info['modality_list'][0]
        else:
            self.modality = modality
        # 构建测试文件路径列表
        if self.modality == 'airpods':
            self.file_path_list = [
                os.path.join(dataset_root_path, 'AirPodsPro', t)
                for t in self.test_file_list
            ]
        else:
            self.file_path_list = [
                os.path.join(dataset_root_path, self.modality, t)
                for t in self.test_file_list
            ]

        # 设置接收器过滤规则和新映射
        self.device_keep_list = device_keep_list
        print(f"device_keep_list: {self.device_keep_list}")
        self.new_mapping = self.info['segment_info'].get('new_mapping', None)

        # 定义模态数据集映射
        self.modality_dataset_map = {
            'imu': WWADL_imu,
            'wifi': WWADL_wifi,
            'airpods': WWADL_airpods
        }
        self.modality_dataset = self.modality_dataset_map[self.modality]

        # 加载分段和目标信息
        segment_info = self.info['segment_info']['train']
        self.clip_length = segment_info[modality]['window_len']
        self.stride = segment_info[modality]['window_step']
        self.target_len = self.info['segment_info']['target_len']

        # 加载评估标签路径
        self.eval_gt = os.path.join(self.dataset_dir, f'{self.modality}_annotations.json')

        # 加载全局均值和标准差
        self.global_mean, self.global_std = self.load_global_stats()

        # 初始化动作ID到动作映射
        self.id_to_action = self.info['segment_info'].get('id2action', id_to_action)

        self.normalize = True



    def load_global_stats(self):
        """
        从文件加载全局均值和方差。
        如果文件中不存在当前 modality，则计算并更新文件。
        """
        stats_path = os.path.join(self.dataset_dir, "global_stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Global stats file '{stats_path}' not found. Ensure it is generated during training.")

        with open(stats_path, 'r') as f:
            stats = json.load(f)
        # 如果当前 modality 不在文件中，计算并保存
        if self.modality not in stats:
            raise FileNotFoundError(
                f"Modality '{self.modality}' not found in stats file. Ensure it is generated during training.")

        # 从文件中加载当前 modality 的均值和方差
        self.global_mean = np.array(stats[self.modality]["global_mean"])
        self.global_std = np.array(stats[self.modality]["global_std"])

        return self.global_mean, self.global_std

    def get_data(self, file_path):
        sample = self.modality_dataset(file_path,
                                       receivers_to_keep=None,
                                       new_mapping=self.new_mapping)
        sample_count = len(sample.data)

        # 生成 offset 列表，用于分割视频片段
        if sample_count < self.clip_length:
            offsetlist = [0]  # 视频长度不足 clip_length，只取一个片段
        else:
            offsetlist = list(range(0, sample_count - self.clip_length + 1, self.stride))  # 根据步长划分片段
            if (sample_count - self.clip_length) % self.stride:
                offsetlist += [sample_count - self.clip_length]  # 确保最后一个片段不被遗漏

        for offset in offsetlist:
            clip = sample.data[offset: offset + self.clip_length]  # 获取当前的 clip

            # 调用封装的函数进行处理
            clip = handle_nan_and_interpolate(clip, self.clip_length, self.target_len)
            assert not np.any(np.isnan(clip)), "Data contains NaN values!"

            clip = torch.tensor(clip, dtype=torch.float32)

            # 根据模态处理数据
            if self.modality == 'imu':
                clip = self.process_imu(clip)
            elif self.modality == 'wifi':
                clip = self.process_wifi(clip)
            elif self.modality == 'airpods':
                clip = self.process_airpods(clip)

            data = {
                self.modality: clip
            }

            yield data, [offset, offset + self.clip_length]

    def process_imu(self, sample):
        sample = sample.permute(1, 2, 0)  # [5, 6, 2048]
        device_num = sample.shape[0]
        imu_channel = sample.shape[1]
        sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
            
        if self.device_keep_list:
            # 1. sample [5*6=30, 2048] -> [5, 6, 2048]
            sample = sample.reshape(device_num, imu_channel, -1)  # [5, 6, 2048]
            # 2. 保留设备 [5, 6, 2048] -> [2, 6, 2048] 保留设备 2, 3
            sample = sample[self.device_keep_list]  # [len(device_keep_list), 6, 2048]
            # 3. [2, 6, 2048] -> [2*6=12, 2048]
            sample = sample.reshape(-1, sample.shape[-1])  # [5*6=30, 2048]
        
        return sample

    def process_wifi(self, sample):
        # WiFi数据处理
        # [2048, 3, 3, 30]
        sample = sample.permute(1, 2, 3, 0)  # [3, 3, 30, 2048]
        sample = sample.reshape(-1, sample.shape[-1])  # [3*3*30, 2048]
        
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
        return sample

    def process_airpods(self, sample):
        # AirPods 数据处理：处理加速度和角速度
        acceleration = sample[:, 3:6]  # 加速度: X, Y, Z (列索引 3 到 5)
        rotation = sample[:, 6:9]      # 角速度: X, Y, Z (列索引 6 到 8)
        
        # 合并加速度和角速度数据 [2048, 6]
        sample = torch.cat((acceleration, rotation), dim=1)  # [2048, 6]
        
        # 转置为 [6, 2048]
        sample = sample.T  # 转置为 [6, 2048]
        
        # 全局归一化：使用序列维度的均值和标准差
        if self.normalize:
            sample = (sample - torch.tensor(self.global_mean, dtype=torch.float32)[:, None]) / \
                     (torch.tensor(self.global_std, dtype=torch.float32)[:, None] + 1e-6)
        return sample

    def dataset(self):
        for file_path, file_name in zip(self.file_path_list, self.test_file_list):
            yield file_name, self.get_data(file_path)



if __name__ == '__main__':
    dataset = WWADLDatasetTestSingle('/root/shared-nvme/WWADL', '/root/shared-nvme/dataset/imu_30_3')
    dataset.dataset()
