
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

    def iter_data(self, imu_data_iter, wifi_data_iter, airpods_data_iter):
        for imu_data, wifi_data, airpods_data in zip(imu_data_iter, wifi_data_iter, airpods_data_iter):
            data = {}
            if self.receivers_to_keep['imu']:
                data['imu'] = imu_data[0]['imu']
            if self.receivers_to_keep['wifi']:
                data['wifi'] = wifi_data[0]['wifi']
            if self.receivers_to_keep['airpods']:
                if data.get('imu') is not None:
                    data['imu'] = torch.cat((data['imu'], airpods_data[0]['airpods']), dim=0)
                else:
                    data['imu'] = airpods_data[0]['airpods']
            yield data, imu_data[1]


    def dataset(self):
        """
        生成器：根据 receivers_to_keep 动态加载选定模态的数据文件，并返回文件名和对应的数据生成器。
        """
        # 当其中一个生成器耗尽时，zip 会停止
        imu_files = zip(self.imu_dataset.file_path_list, self.imu_dataset.test_file_list)
        wifi_files = zip(self.wifi_dataset.file_path_list, self.wifi_dataset.test_file_list)
        airpods_files = zip(self.airpods_dataset.file_path_list, self.airpods_dataset.test_file_list)
        for (imu_path, imu_name), (wifi_path, wifi_name), (airpods_path, airpods_name) in zip(imu_files, wifi_files, airpods_files):
            # 打印文件路径和名称（便于调试）
            # print(imu_path, imu_name, wifi_path, wifi_name)
            
            # 确保文件名一致
            assert imu_name == wifi_name == airpods_name, f"File name mismatch: {imu_name} != {wifi_name} != {airpods_name}"
            
            # 返回文件名和数据迭代器
            yield imu_name, self.iter_data(
                imu_data_iter=self.imu_dataset.get_data(imu_path),
                wifi_data_iter=self.wifi_dataset.get_data(wifi_path),
                airpods_data_iter=self.airpods_dataset.get_data(airpods_path)
            )
