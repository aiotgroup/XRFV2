import os
import json


def get_basic_config(basic_config_path = 'basic_config.json'):
    with open(basic_config_path, 'r') as json_file:
        config = json.load(json_file)
    return config

def write_setting(config):
    # path = os.path.join(save_path, 'setting.json')
    save_path = os.path.join(config['path']['result_path'], 'setting.json')
    with open(save_path, "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
