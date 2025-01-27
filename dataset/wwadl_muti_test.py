
from torch.utils.data import Dataset
from dataset.wwadl_test import WWADLDatasetTestSingle

class WWADLDatasetTestMuti():
    def __init__(self, config):
        self.imu_dataset = WWADLDatasetTestSingle(config, 'imu')
        self.wifi_dataset = WWADLDatasetTestSingle(config, 'wifi')
        self.eval_gt = self.imu_dataset.eval_gt
        self.id_to_action = self.imu_dataset.id_to_action
        print('WWADLDatasetTestMuti')

    def iter_data(self, imu_data_iter, wifi_data_iter):
        """
        生成器：并行迭代 IMU 数据和 WiFi 数据，输出合并后的数据字典和标签。
        """
        for imu_data, wifi_data in zip(imu_data_iter, wifi_data_iter):
            # print(imu_data[1], wifi_data[1])
            data = {
                'imu': imu_data[0]['imu'],
                'wifi': wifi_data[0]['wifi']
            }
            yield data, imu_data[1]


    def dataset(self):
        """
        生成器：遍历 IMU 和 WiFi 文件路径，检查文件名一致性，并返回文件名和对应的数据生成器。
        """
        # 当其中一个生成器耗尽时，zip 会停止
        imu_files = zip(self.imu_dataset.file_path_list, self.imu_dataset.test_file_list)
        wifi_files = zip(self.wifi_dataset.file_path_list, self.wifi_dataset.test_file_list)

        for (imu_path, imu_name), (wifi_path, wifi_name) in zip(imu_files, wifi_files):
            # 打印文件路径和名称（便于调试）
            # print(imu_path, imu_name, wifi_path, wifi_name)

            # 确保文件名一致
            assert imu_name == wifi_name, f"File name mismatch: {imu_name} != {wifi_name}"

            # 返回文件名和数据迭代器
            yield imu_name, self.iter_data(
                self.imu_dataset.get_data(imu_path),
                self.wifi_dataset.get_data(wifi_path)
            )




if __name__ == '__main__':
    from global_config import get_basic_config

    config = get_basic_config()

    config['path']['dataset_path'] = '/root/shared-nvme/dataset/all_30_3'

    dataset = WWADLDatasetTestMuti(config=config)

    for file_name, data in dataset.dataset():
        print(file_name)
        for d, segment in data:
            print(d['wifi'].shape, d['imu'].shape, segment)
            # break
        break

    # def a():
    #     for i in range(3):
    #         yield i
    #
    # def b():
    #     for i in range(4):
    #         yield i
    #
    # def c():
    #     for i, ii in zip(a(), b()):
    #         yield i, ii
    #
    # for (i, ii) in c():
    #     print(i, ii)