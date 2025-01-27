
import torch
from torch.utils.data import Dataset
from dataset.wwadl_test import WWADLDatasetTestSingle

imu_name_to_id = {
    'gl': 0,
    'lh': 1,
    'rh': 2,
    'lp': 3,
    'rp': 4,
}

class WWADLDatasetTestMutiALL():
    def __init__(self, config, receivers_to_keep=None):
        
        if receivers_to_keep is None:
            self.receivers_to_keep = {
                'imu': [0, 1, 2, 3, 4],
                'wifi': True,
                'airpods': False
            }
        else:
            self.receivers_to_keep = {
                'imu': [imu_name_to_id[receiver] for receiver in receivers_to_keep['imu']] if receivers_to_keep['imu'] else None,
                'wifi': receivers_to_keep['wifi'],
                'airpods': receivers_to_keep['airpods']
            }
        
        self.imu_dataset = WWADLDatasetTestSingle(config, 'imu', self.receivers_to_keep['imu'])
        self.wifi_dataset = WWADLDatasetTestSingle(config, 'wifi')
        self.airpods_dataset = WWADLDatasetTestSingle(config, 'airpods')

        self.eval_gt = self.imu_dataset.eval_gt
        self.id_to_action = self.imu_dataset.id_to_action
        print('WWADLDatasetTestMuti')

    def iter_data(self, **data_iters):
        """
        动态生成器：并行迭代选定的模态数据，输出合并后的数据字典和标签。
        """
        # 动态组合所有选择的模态迭代器
        for data_batch in zip(*data_iters.values()):
            data = {}
            label = None  # 用于保存统一的标签
            
            # data_batch 是一个包含多个元素的元组，按顺序包含了每个模态的 (data, label)
            for idx, modality in enumerate(data_iters.keys()):
                # 从 data_batch 中取出每个模态的数据
                modality_data, modality_label = data_batch[idx]
                
                if modality == 'airpods':
                    if data.get('imu') is not None:
                        data['imu'] = torch.cat((data['imu'], modality_data['airpods']), dim=0)
                    else:
                        data['imu'] = modality_data['airpods']

                if modality == 'imu':
                    if data.get('imu') is not None:
                        data['imu'] = torch.cat((data['imu'], modality_data['imu']), dim=0)
                    else:
                        data['imu'] = modality_data['imu']
                
                if modality == 'wifi':
                    data['wifi'] = modality_data['wifi']
                
                # print(f"modality: {modality}, {modality_label}")

                if label is None:  # 假设所有模态的标签相同，取任意模态的标签即可
                    label = modality_label

            yield data, label


    def dataset(self):
        """
        生成器：根据 receivers_to_keep 动态加载选定模态的数据文件，并返回文件名和对应的数据生成器。
        """
        # 动态构造模态文件路径
        file_iterators = {}
        if self.receivers_to_keep['imu']:
            imu_files = zip(self.imu_dataset.file_path_list, self.imu_dataset.test_file_list)
            file_iterators['imu'] = imu_files
        if self.receivers_to_keep['wifi']:
            wifi_files = zip(self.wifi_dataset.file_path_list, self.wifi_dataset.test_file_list)
            file_iterators['wifi'] = wifi_files
        if self.receivers_to_keep['airpods']:
            airpods_files = zip(self.airpods_dataset.file_path_list, self.airpods_dataset.test_file_list)
            file_iterators['airpods'] = airpods_files

        # 动态获取模态文件迭代器
        for file_groups in zip(*file_iterators.values()):
            # 检查文件名一致性
            file_names = [file_name for _, file_name in file_groups]
            assert all(name == file_names[0] for name in file_names), f"File name mismatch: {file_names}"

            # 构造数据迭代器
            data_iters = {}
            for modality, (file_path, _) in zip(file_iterators.keys(), file_groups):
                if modality == 'imu':
                    data_iters[modality] = self.imu_dataset.get_data(file_path)
                elif modality == 'wifi':
                    data_iters[modality] = self.wifi_dataset.get_data(file_path)
                elif modality == 'airpods':
                    data_iters[modality] = self.airpods_dataset.get_data(file_path)

            # 返回文件名和数据生成器
            yield file_names[0], self.iter_data(**data_iters)
            del data_iters  # 清理迭代器对象
