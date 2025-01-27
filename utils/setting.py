import json
import os
import datetime

class Run_config():
    def __init__(self, config:dict, type):
        self.python_path = config["path"]["basic_path"]["python_path"]
        self.main_path = config["path"]["basic_path"]["project_path"]+'/main.py'
        if type == 'test':
            self.ddp_devices, self.master_port, self.nproc_per_node = self.get_ddp_config(config["training"]["DDP"]["devices"][:1])
        else:
            self.ddp_devices, self.master_port, self.nproc_per_node = self.get_ddp_config(config["training"]["DDP"]["devices"])
        self.log_path = config["path"]["log_path"][type]
        self.config_path = config['path']['result_path'] + '/setting.json'

    def get_ddp_config(self, devices: list):
        nproc_per_node = len(devices)
        master_port = '29501'
        ddp_devices = ''
        for i in range(nproc_per_node - 1):
            ddp_devices += f'{devices[i]},'
        ddp_devices += f'{devices[nproc_per_node - 1]}'

        return ddp_devices, master_port, nproc_per_node

def load_setting(url: str)->dict:
    with open(url, 'r') as f:
        data = json.load(f)
        return data

def write_setting(data):
    # path = os.path.join(save_path, 'setting.json')
    save_path = os.path.join(data['path']['result_path'], 'setting.json')
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def update_time(config: dict, type='datetime')->dict:
    now = datetime.datetime.now()
    current_month = now.strftime('%m')  # 月份（带前导零）
    current_day = now.strftime('%d')  # 日（带前导零）
    current_hour = now.strftime('%H')  # 小时（24小时制）
    current_minute = now.strftime('%M')
    config[type] = f"{current_month}-{current_day}-{current_hour}-{current_minute}"
    return config

def get_day()->str:
    now = datetime.datetime.now()
    current_month = now.strftime('%m')  # 月份（带前导零）
    current_day = now.strftime('%d')  # 日（带前导零）
    return f"25_{current_month}-{current_day}"

def get_time()->str:
    """
    获取当前时间并格式化为 'YYYY_MM-DD_HH:MM:SS' 格式
    Returns:
        str: 当前时间的字符串表示
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m-%d_%H:%M:%S")
    return formatted_time

def get_log_path(config: dict, day: str, dataset_set: str, model_set: str, tag: str=None)->dict:
    log_path = os.path.join(config['path']['basic_path']['log_path'], day)
    if tag is not None:
        log_path = os.path.join(log_path, f"{tag}")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # if config["training"]["pretrain"]["enable"]:
    #     pretrain_ratio = config["training"]["pretrain"]["ratio"]
    #     train_log_path = os.path.join(log_path, f"{dataset_set}_{model_set}_{pretrain_ratio}-TRAIN.log")
    #     test_log_path = os.path.join(log_path, f"{dataset_set}_{model_set}_{pretrain_ratio}-TEST.log")
    # else:
    train_log_path  = os.path.join(log_path, f"{dataset_set}_{model_set}-TRAIN.log")
    test_log_path   = os.path.join(log_path, f"{dataset_set}_{model_set}-TEST.log")
    log_path = {
        'train': train_log_path,
        'test':  test_log_path
    }
    return log_path

def get_result_path(config: dict, day: str, dataset_set: str, model_set: str, tag: str=None)->os.path:
    result_path = os.path.join(config['path']['basic_path']['result_path'], day)
    if tag is not None:
        result_path = os.path.join(result_path, f"{tag}")
    result_path = os.path.join(result_path, f"{dataset_set}_{model_set}")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    return result_path

class Setting():
    def __init__(self, url:str):
        self.url = url
        self.config = load_setting(url)
        self.config = update_time(self.config)
        self.datetime = self.config['datetime']

    def get_config(self):
        return self.config

    def update(self, config:dict):
        self.config = config
        write_setting(config)