import logging
import importlib
from dataset.wwadl import WWADLDatasetSingle
from dataset.wwadl_test import WWADLDatasetTestSingle
from dataset.wwadl_muti import WWADLDatasetMuti
from dataset.wwadl_muti_test import WWADLDatasetTestMuti
from dataset.wwadl_muti_all import WWADLDatasetMutiAll
from dataset.wwadl_muti_all_test import WWADLDatasetTestMutiALL

logger = logging.getLogger(__name__)

def init_dataset(config: dict):
    if config['dataset']['dataset_name'] == 'WWADLDatasetSingle':
        dataset = WWADLDatasetSingle(config['path']['dataset_path'], split='train', modality=config["model"]["modality"])
    elif config['dataset']['dataset_name'] == 'WWADLDatasetMuti':
        dataset = WWADLDatasetMuti(config['path']['dataset_path'])
    elif config['dataset']['dataset_name'] == 'WWADLDatasetMutiAll':
        dataset = WWADLDatasetMutiAll(config['path']['dataset_path'], split='train', receivers_to_keep=config['dataset']['receivers_to_keep'])
    else:
        raise ValueError(f"Unsupported dataset name: {config['dataset']['dataset_name']}. "
                         "Please check the configuration.")

    # print(f"Number of samples in data: {dataset.data.shape[0]}")
    # print(f"Number of labels: {len(dataset.labels)}")
    # Log dataset information
    logger.info(f"Dataset '{config['dataset']['dataset_name']}' loaded successfully.")
    logger.info(f"Number of samples: {dataset.shape()[0]} \t shape: {dataset.shape()}")
    logger.info(f"Number of labels: {len(set(dataset.labels))}")

    return dataset

def init_test_dataset(config: dict):
    if config['dataset']['dataset_name'] == 'WWADLDatasetSingle':
        # dataset = WWADLDatasetTestSingle(config, modality=config["model"]["modality"])
        dataset = WWADLDatasetTestSingle(config, modality=config["model"]["modality"])
    elif config['dataset']['dataset_name'] == 'WWADLDatasetMuti':
        dataset = WWADLDatasetTestMuti(config)
    elif config['dataset']['dataset_name'] == 'WWADLDatasetMutiAll':
        dataset = WWADLDatasetTestMutiALL(config, receivers_to_keep=config['dataset']['receivers_to_keep'])
    else:
        raise ValueError(f"Unsupported dataset name: {config['dataset']['dataset_name']}. "
                         "Please check the configuration.")

    # print(f"Number of samples in data: {dataset.data.shape[0]}")
    # print(f"Number of labels: {len(dataset.labels)}")
    # Log dataset information
    logger.info(f"Dataset '{config['dataset']['dataset_name']}' loaded successfully.")
    return dataset

def init_model(config: dict):
    logger.info(f"Initializing model with backbone: {config['model']['backbone_name']} and model set: {config['model']['model_set']}...")
    Model = importlib.import_module("model")
    Backbone_config = getattr(Model, f"{config['model']['backbone_name']}_config")
    Backbone = getattr(Model, config['model']['backbone_name'])
    backbone_config = Backbone_config(config['model']['model_set'])
    model = Backbone(backbone_config)
    logger.info(f"Model {config['model']['backbone_name']} initialized successfully.")
    return model


if __name__ == '__main__':
    # dataset = init_dataset(config)
    from global_config import config
    model = init_model(config)

