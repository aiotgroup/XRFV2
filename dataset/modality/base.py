import os
import numpy as np
from scipy.interpolate import interp1d

def handle_nan_and_interpolate(data, window_len, target_len):
    """
    插值并在插值前处理 NaN 值的通用函数。
    Args:
        data (np.ndarray): 输入数据，维度为 (window_len, ...)
        window_len (int): 原始时序长度。
        target_len (int): 目标时序长度。
    Returns:
        np.ndarray: 插值后的数据，时间维度变为 target_len，其他维度保持不变。
    """
    original_shape = data.shape  # 获取原始形状
    flattened_data = data.reshape(window_len, -1)  # 展平除时间维度以外的所有维度

    # 定义插值函数
    def interpolate_channel(channel_data):
        original_indices = np.linspace(0, window_len - 1, window_len)
        target_indices = np.linspace(0, window_len - 1, target_len)

        # 检查 NaN 并处理
        nan_mask = np.isnan(channel_data)
        if np.any(nan_mask):  # 如果存在 NaN 值
            valid_indices = np.where(~nan_mask)[0]
            valid_values = channel_data[~nan_mask]

            if len(valid_indices) > 1:  # 至少两个有效值
                interp_func = interp1d(valid_indices, valid_values, kind='linear', bounds_error=False, fill_value="extrapolate")
                channel_data = interp_func(np.arange(window_len))
            else:
                # 如果有效值不足，填充为 0
                channel_data = np.zeros_like(channel_data)

        # 插值到目标长度
        interp_func = interp1d(original_indices, channel_data, kind='linear', bounds_error=False, fill_value="extrapolate")
        return interp_func(target_indices)

    # 对所有通道进行插值处理
    interpolated_flattened_data = np.array([
        interpolate_channel(flattened_data[:, i])
        for i in range(flattened_data.shape[1])
    ]).T  # 转置回时间维度在前

    # 恢复原始形状，时间维度替换为 target_len
    reshaped_interpolated_data = interpolated_flattened_data.reshape(target_len, *original_shape[1:])
    return reshaped_interpolated_data

class WWADLBase():
    def __init__(self, file_path):
        self.data = None
        self.label = None
        self.file_name = os.path.basename(file_path)
        self.window_len = 0
        self.window_step = 0

    def load_data(self, file_path):
        pass

    def mapping_label(self, old_to_new_mapping):
        for i in range(len(self.label)):
            try:
                self.label[i][1] = old_to_new_mapping[str(self.label[i][1])]
            except:
                print(self.label[i][1], old_to_new_mapping)