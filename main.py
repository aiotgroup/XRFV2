import os
import sys
import torch

# project_path = '/home/lanbo/WWADL/WWADL_code'
# project_path = '/root/shared-nvme/code/WWADL_code'
# sys.path.append(project_path)

import argparse
import torch
import json
from training import train
from testing import test

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data

import logging

def setup_logging(log_file_path):
    """
    设置日志配置，将日志内容输出到控制台和文件。
    Args:
        log_file_path (str): 日志文件的路径，例如 'output.log'。
    """
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()  # 获取根日志记录器
    logger.setLevel(logging.INFO)

    # 检查是否已经设置了处理器，避免重复
    if not logger.handlers:
        # 文件处理器
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

def init_configs():
    parser = argparse.ArgumentParser(description="WiVio study")
    parser.add_argument('--is_train', dest="is_train", required=False, type=bool, default=False,
                        help="是否训练")
    parser.add_argument('--config_path', dest="config_path", required=True, type=str, help="config 的地址")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = init_configs()
    config = load_setting(args.config_path)
    if args.is_train:
        setup_logging(config['path']['log_path']['train'])
        train(config=config)
    else:
        setup_logging(config['path']['log_path']['test'])
        test(config=config)