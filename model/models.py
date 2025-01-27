

backbones = {}
def register_backbone(name):
    def decorator(cls):
        backbones[name] = cls
        return cls
    return decorator

backbones_config = {}
def register_backbone_config(name):
    def decorator(cls):
        backbones_config[name] = cls
        return cls
    return decorator

def make_backbone_config(name, cfg = None):
    backbone_config = backbones_config[name](cfg)
    return backbone_config

# builder functions
def make_backbone(name, cfg = None):
    backbone = backbones[name](cfg)
    return backbone

# meta arch (the actual implementation of each model)
models = {
    'mamba': None
}
models_config = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def register_model_config(name):
    def decorator(cls):
        models_config[name] = cls
        return cls
    return decorator

def make_model_config(name, cfg=None):
    meta_arch_config = models_config[name](cfg)
    return meta_arch_config

def make_model(name, cfg=None):
    meta_arch = models[name](cfg)
    return meta_arch