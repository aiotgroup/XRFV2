
from dataset.modality.base import WWADLBase
from utils.h5 import load_h5

class WWADL_wifi(WWADLBase):
    """
    数据维度说明:
    - 数据 shape: (2900, 3, 3, 30)
        - 第一个维度 (2900): 样本数量（例如时间序列的时间步）
        - 第二个维度 (3): 设备数量（例如接收设备或天线）
        - 第三个维度 (3): 通道数量（例如发送天线或数据通道）
        - 第四个维度 (30): 特征维度（如频率子载波或信号特征）
    """

    def __init__(self, file_path, receivers_to_keep=None, new_mapping=None):
        """
        初始化 WiFi 数据处理类，并保留指定的接收设备

        Args:
            file_path (str): 数据文件路径
            receivers_to_keep (list, optional): 要保留的接收设备索引列表（如 [0, 2]）。
        """
        super().__init__(file_path)
        self.load_data(file_path)

        # 如果提供了需要保留的接收设备列表，则过滤设备维度
        if receivers_to_keep:
            self.retain_receivers(receivers_to_keep)

        if new_mapping:
            self.mapping_label(new_mapping)


    def load_data(self, file_path):

        data = load_h5(file_path)

        self.data = data['amp']

        self.label = data['label']


    def retain_receivers(self, receivers_to_keep):
        """
        过滤并保留指定接收设备的维度

        Args:
            receivers_to_keep (list): 要保留的接收设备索引列表
        """
        # 检查接收设备索引范围是否有效
        max_receivers = self.data.shape[1]
        valid_receivers = [i for i in receivers_to_keep if 0 <= i < max_receivers]

        # 保留指定接收设备的维度
        self.data = self.data[:, valid_receivers, :, :]












