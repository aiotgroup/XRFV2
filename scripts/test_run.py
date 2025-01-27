import os
import sys
import json

# 定义路径
project_path = '/root/shared-nvme/XRFV2'
dataset_root_path = '/root/shared-nvme/dataset'
causal_conv1d_path = '/root/shared-nvme/video-mamba-suite/causal-conv1d'
mamba_path = '/root/shared-nvme/video-mamba-suite/mamba'
sys.path.append(project_path)

os.environ["PYTHONPATH"] = f"{project_path}:{causal_conv1d_path}:{mamba_path}:" + os.environ.get("PYTHONPATH", "")
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path, Run_config

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data

test_model_list = [
    '/root/shared-nvme/code_result/result/WWADLDatasetMutiAll_XRFV2_mamba_layer_8_i_1-all'
]


for test_model_path in test_model_list:
    config = load_setting(os.path.join(test_model_path, 'setting.json'))

    config['path']['dataset_root_path'] = '/root/shared-nvme/WWADL'
    config['path']['dataset_path'] = '/root/shared-nvme/dataset/open_dataset'
    config['path']['basic_path']['python_path'] = '/root/.conda/envs/t1/bin/python'
    run = Run_config(config, 'train')

    test_gpu = 1

    # config['testing']['pt_file_name'] = 'Transformer_layer_8_-final'
    # config['model']['backbone_name'] = 'Transformer'

    write_setting(config)

    print(run.config_path)

    run.python_path = '/root/.conda/envs/mamba/bin/python'

    os.system(
        f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
        f"{run.main_path} --config_path {run.config_path} "
    )