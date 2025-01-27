import numpy as np
from dataset.modality.base import WWADLBase
from utils.h5 import load_h5


class WWADL_imu(WWADLBase):
    """
    数据维度说明:
    - 数据 shape: (2900, 5, 6)
        - 第一个维度 (2900): 样本数量（例如时间序列的时间步）
        - 第二个维度 (5): 设备数量（5个IMU设备，对应位置见 name_to_id）
        - 第三个维度 (6): IMU数据的维度（例如加速度和陀螺仪的6个轴数据）

    name_to_id 映射设备位置到索引:
        - 'glasses': 0
        - 'left hand': 1
        - 'right hand': 2
        - 'left pocket': 3
        - 'right pocket': 4
    """
    def __init__(self, file_path, receivers_to_keep=None, new_mapping=None):
        """
        初始化 IMU 数据处理类，并保留指定设备的维度

        Args:
            file_path (str): 数据文件路径
            devices_to_keep (list, optional): 要保留的设备名称列表（如 ['glasses', 'left hand']）。
        """
        super().__init__(file_path)
        self.duration = 0
        self.load_data(file_path)

        # 如果提供了需要保留的设备列表，则过滤设备维度
        if receivers_to_keep:
            self.retain_devices(receivers_to_keep)

        if new_mapping:
            self.mapping_label(new_mapping)

    def load_data(self, file_path):
        """
        加载IMU数据并完成预处理

        Args:
            file_path (str): 数据文件的路径
        """
        data = load_h5(file_path)
        # 数据转置: (5, 2900, 6) 调整为 (2900, 5, 6)
        self.data = np.transpose(data['data'], (1, 0, 2))

        # 加载标签和持续时间
        self.label = data['label']
        self.duration = data['duration']

    def retain_devices(self, devices_to_keep):
        """
        过滤并保留指定设备的维度

        Args:
            devices_to_keep (list): 需要保留的设备名称列表
        """
        # 定义 name_to_id 映射
        name_to_id = {
            'glasses': 0,
            'left hand': 1,
            'right hand': 2,
            'left pocket': 3,
            'right pocket': 4,
        }

        # 根据名称映射找到对应的索引
        device_indices = [name_to_id[device] for device in devices_to_keep if device in name_to_id]

        # 保留指定设备维度
        self.data = self.data[:, device_indices, :]

