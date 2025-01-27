
import torch
from torch.utils.data import Dataset
from dataset.wwadl import WWADLDatasetSingle


imu_name_to_id = {
    'gl': 0,
    'lh': 1,
    'rh': 2,
    'lp': 3,
    'rp': 4,
}

class WWADLDatasetMutiAll(Dataset):
    def __init__(self, dataset_dir, split="train", receivers_to_keep=None):
        """
        初始化 WWADL 数据集。
        :param dataset_dir: 数据集所在目录路径。
        :param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        if receivers_to_keep is None:
            receivers_to_keep = {
                'imu': [0, 1, 2, 3, 4],
                'wifi': True,
                'airpods': True
            }
        else:
            receivers_to_keep = {
                'imu': [imu_name_to_id[receiver] for receiver in receivers_to_keep['imu']] if receivers_to_keep['imu'] else None,
                'wifi': receivers_to_keep['wifi'],
                'airpods': receivers_to_keep['airpods']
            }
        self.imu_dataset = None
        self.wifi_dataset = None
        self.airpods_dataset = None
        self.labels = None
        if receivers_to_keep['imu']:
            self.imu_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='imu', device_keep_list=receivers_to_keep['imu'])
            self.labels = self.imu_dataset.labels
        if receivers_to_keep['wifi']:
            self.wifi_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='wifi')
            if self.labels is None:
                self.labels = self.wifi_dataset.labels
        if receivers_to_keep['airpods']:
            self.airpods_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='airpods')
            if self.labels is None:
                self.labels = self.airpods_dataset.labels
        
        self.data_len = None

    def shape(self):
        shape_info = ''
        if self.imu_dataset is not None:
            self.data_len = self.imu_dataset.shape()[0]
            shape_info += f'{self.imu_dataset.shape()}_'
        if self.wifi_dataset is not None:
            self.data_len = self.wifi_dataset.shape()[0]
            shape_info += f'{self.wifi_dataset.shape()}_'
        if self.airpods_dataset is not None:
            self.data_len = self.airpods_dataset.shape()[0]
            shape_info += f'{self.airpods_dataset.shape()}'
        return self.data_len, shape_info

    def __len__(self):
        """
        返回数据集的样本数。
        """
        if self.data_len is None:
            self.shape()
        return self.data_len

    def __getitem__(self, idx):
        data = {}
        label = None
        if self.imu_dataset is not None:
            imu_data, imu_label = self.imu_dataset[idx]
            data['imu'] = imu_data['imu']
            label = imu_label
        if self.wifi_dataset is not None:
            wifi_data, wifi_label = self.wifi_dataset[idx]
            data['wifi'] = wifi_data['wifi']
            if label is None:
                label = wifi_label
        if self.airpods_dataset is not None:
            airpods_data, airpods_label = self.airpods_dataset[idx]
            if self.imu_dataset is not None:
                data['imu'] = torch.cat((data['imu'], airpods_data['airpods']), dim=0)
            else:
                data['imu'] = airpods_data['airpods']
            if label is None:
                label = airpods_label
        return data, label

