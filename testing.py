
from init_utils import init_dataset, init_model, init_test_dataset
from model.models import make_model, make_model_config
from pipeline.tester import Tester

def test(config):
    test_dataset = init_test_dataset(config)

    model_cfg = make_model_config(config['model']['backbone_name'], config['model'])
    model = make_model(config['model']['name'], model_cfg)

    pt_file_name = config['testing'].get('pt_file_name', None)
    tester = Tester(config, test_dataset=test_dataset, model=model, pt_file_name=pt_file_name)
    tester.testing()