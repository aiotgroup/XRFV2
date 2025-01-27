import os
import json
import h5py
import numpy as np
from tqdm import tqdm


def load_h5(filepath):
    def recursively_load_group_to_dict(h5file, path):
        """
        递归加载 HDF5 文件中的组和数据集为嵌套字典
        """
        result = {}
        group = h5file[path]

        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # 如果是组，则递归加载
                result[key] = recursively_load_group_to_dict(h5file, f"{path}/{key}")
            elif isinstance(item, h5py.Dataset):
                # 如果是数据集，则加载为 NumPy 数组
                result[key] = item[()]

        return result

    with h5py.File(filepath, 'r') as h5file:
        return recursively_load_group_to_dict(h5file, '/')



def save_h5(filepath, data):
    """
    将嵌套字典保存为 HDF5 文件，带进度条显示
    """

    def recursively_save_dict_to_group(h5file, path, dictionary, progress):
        """
        递归保存嵌套字典到 HDF5 文件，更新进度条
        """
        for key, value in dictionary.items():
            # 处理键的路径
            full_path = f"{path}/{key}"
            progress.update(1)

            if isinstance(value, dict):
                # 如果是字典，则递归保存
                recursively_save_dict_to_group(h5file, full_path, value, progress)
            elif isinstance(value, np.ndarray):
                # 如果是 NumPy 数组，直接保存
                h5file.create_dataset(full_path, data=value)
            else:
                # 如果是其他类型，转换为 NumPy 数组后保存
                h5file.create_dataset(full_path, data=np.array(value))

    # 统计总保存项的数量
    def count_items(dictionary):
        count = 0
        for key, value in dictionary.items():
            if isinstance(value, dict):
                count += count_items(value)
            else:
                count += 1
        return count

    total_items = count_items(data)

    # 创建 HDF5 文件并显示进度条
    with h5py.File(filepath, 'w') as h5file, tqdm(total=total_items, desc="Saving HDF5 data") as progress:
        recursively_save_dict_to_group(h5file, '/', data, progress)


def save_data_in_batches(data, labels, batch_size, h5_file_path, json_file_path):
    """
    分批次保存数据到 H5 和 JSON 文件中。
    :param data: np.array, 形状为 (n, 3000, 5) 的数据
    :param labels: list, 长度为 n 的标签，每个标签可能长度不一
    :param batch_size: 每批保存的样本数量
    :param h5_file_path: 保存 H5 文件的路径
    :param json_file_path: 保存 JSON 文件的路径
    """
    n = data.shape[0]
    label_dict = {}

    # 创建 HDF5 文件并预分配数据集
    with h5py.File(h5_file_path, 'w') as h5_file:
        data_shape = data.shape[1:]
        h5_dataset = h5_file.create_dataset('data', shape=(n, *data_shape), dtype=data.dtype)

        # 分批写入数据
        for i in range(0, n, batch_size):
            start = i
            end = min(i + batch_size, n)
            h5_dataset[start:end] = data[start:end]

            # 同时保存对应的标签
            for j in range(start, end):
                label_dict[j] = labels[j]

    # 保存标签到 JSON 文件
    with open(json_file_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=4)


def read_data_by_id(h5_file_path, json_file_path, sample_id):
    """
    根据指定的 ID 读取数据和标签。
    :param h5_file_path: H5 文件的路径
    :param json_file_path: JSON 文件的路径
    :param sample_id: 要读取的样本 ID
    :return: 对应 ID 的数据和标签
    """
    # 读取标签
    with open(json_file_path, 'r') as json_file:
        labels = json.load(json_file)

    if str(sample_id) not in labels:
        raise ValueError(f"ID {sample_id} 不存在！")

    # 读取 HDF5 数据
    with h5py.File(h5_file_path, 'r') as h5_file:
        data = h5_file['data'][sample_id]  # 按索引读取数据

    label = labels[str(sample_id)]
    return data, label