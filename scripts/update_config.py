import os
from utils.setting import get_day, get_time, write_setting, get_result_path, get_log_path


def prepare_config(model_arc_name, dataset_str, model_str, config, gpu, day, dataset_root_path, python_path, tag, gpus=None, receivers_to_keep=None):
    dataset_name, dataset, channel, modality = dataset_str
    model_name, batch_size, epoch, model_config = model_str
    model_set = model_name
    for k, v in model_config.items():
        model_set += f'_{k}_{v}'
    
    config['datetime'] = get_time()
    config["training"]["DDP"]["enable"] = True
    if gpus:
        config["training"]["DDP"]["devices"] = gpus
    else:
        config["training"]["DDP"]["devices"] = [gpu]
    config["model"]['num_classes'] = 30
    config["dataset"]['num_classes'] = 30
    config["model"]['name'] = model_arc_name
    config["model"]["backbone_config"] = model_config
    config["model"]["backbone_name"] = model_name
    
    if receivers_to_keep:
        channel = receivers_to_keep['channel']
        modality = receivers_to_keep['modality']
        model_set = f"{model_set}-{receivers_to_keep['tag']}"

    if isinstance(channel, tuple):
        config["model"]['imu_in_channels'] = channel[0]
        config["model"]['wifi_in_channels'] = channel[1]
    else:
        config["model"]["in_channels"] = channel
    
    config["model"]["model_set"] = model_set
    config["model"]["modality"] = modality
    config["training"]["lr_rate"] = 4e-05
    
    # TAG =====================================================================================
    config['path']['dataset_path'] = os.path.join(dataset_root_path, dataset)
    config['path']['log_path'] = get_log_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)
    config['path']['result_path'] = get_result_path(config, day, f'{dataset_name}_{dataset}', model_set, tag)
    config['path']['basic_path']['python_path'] = python_path
    
    config['dataset']['dataset_name'] = dataset_name
    config['dataset']['clip_length'] = 1500
    if receivers_to_keep:
        config['dataset']['receivers_to_keep'] = receivers_to_keep
    config["training"]['num_epoch'] = epoch
    config["training"]['train_batch_size'] = batch_size
    
    write_setting(config)
    return config
