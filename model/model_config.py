from utils.basic_config import Config
from model.models import register_model_config, register_model, make_backbone, make_backbone_config

class TAD_single_Config(Config):
    def __init__(self):
        self.name = ''
        self.model_set = ''
        self.num_classes = 34
        self.input_length = 2048
        self.in_channels = 30
        self.out_channels = 512
        self.wifi_in_channels = 270
        self.imu_in_channels = 30
        self.priors = 128
        self.backbone_name = 'mamba'
        self.backbone_config = None
        self.modality = 'wifi'
        self.embedding_stride=1
        self.head_num = 3
        self.embed_type = 'Norm'

    def init_model_config(self):
        backbone_config = make_backbone_config(self.backbone_name, cfg=self.backbone_config)
        self.backbone_config = backbone_config.get_dict()
        return backbone_config

@register_model_config('mamba')
class Mamba_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'TAD'
        self.update(cfg)
        backbone_config = self.init_model_config()
        # self.backbone_config.input_length = self.input_length
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('ActionMamba')
class ActionMamba_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'Down'
        self.update(cfg)
        self.backbone_name = 'mamba'
        backbone_config = self.init_model_config()
        # self.backbone_config.input_length = self.input_length
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('wifiTAD')
class WifiTAD_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 128
        self.embedding_stride=1
        self.embed_type = 'Norm'

        self.update(cfg)
        backbone_config = self.init_model_config()

        self.backbone_config['input_length'] = self.input_length
        self.head_num = backbone_config.layer_num

@register_model_config('Transformer')
class Transformer_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'TAD'
        self.update(cfg)
        backbone_config = self.init_model_config()
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('ActionFormer')
class ActionFormer_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'Down'
        self.update(cfg)
        self.backbone_name = 'Transformer'
        backbone_config = self.init_model_config()
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('TriDet')
class TriDet_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'Down'
        self.update(cfg)
        self.backbone_name = 'TriDet'
        backbone_config = self.init_model_config()
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('TemporalMaxer')
class TemporalMaxer_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'Down'
        self.update(cfg)
        self.backbone_name = 'TemporalMaxer'
        backbone_config = self.init_model_config()
        self.head_num = backbone_config.arch[-1] + 1

@register_model_config('Ushape')
class Ushape_config(TAD_single_Config):
    def __init__(self, cfg=None):
        super().__init__()
        self.priors = 256
        self.embedding_stride=2
        self.embed_type = 'Down'
        self.out_channels = 128
        self.update(cfg)
        self.backbone_name = 'Ushape'
        backbone_config = self.init_model_config()
        self.head_num = 5
        self.out_channels = backbone_config.in_channels