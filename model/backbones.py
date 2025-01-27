import torch
import torch.nn as nn
import torch.nn.init as init
from utils.basic_config import Config

from model.mamba.backbones import MambaBackbone
from model.transformer.backbones import ConvTransformerBackbone
from model.TAD.backbone import TSSE, LSREF
from model.mamba.necks import FPNIdentity
from model.TriDet.backbones import SGPBackbone
from model.TemporalMaxer.backbones import ConvPoolerBackbone
from model.Ushape.backbones import UNetBackbone, UNetBackbone2
from model.models import register_backbone_config, register_backbone

@register_backbone_config('mamba')
class Mamba_config(Config):
    def __init__(self, cfg = None):
        self.layer = 4
        self.n_embd = 512
        self.n_embd_ks = 3  # 卷积核大小
        self.scale_factor = 2  # 下采样率
        self.with_ln = True  # 使用 LayerNorm
        self.mamba_type = 'dbm'

        self.update(cfg)    # update ---------------------------------------------
        self.arch = (2, self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积

@register_backbone('mamba')
class Mamba(nn.Module):
    def __init__(self, config: Mamba_config):
        super(Mamba, self).__init__()
        # Mamba Backbone
        self.mamba_model = MambaBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            mamba_type=config.mamba_type
        )
        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        feats, masks = self.mamba_model(x, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        return fpn_feats

@register_backbone_config('Transformer')
class Transformer_config(Config):
    def __init__(self, cfg = None):
        # Backbone 配置
        self.n_embd = 512
        self.n_head = 8
        self.n_embd_ks = 3  # 卷积核大小
        self.max_len = 256
        # window size for self attention; <=1 to use full seq (ie global attention)
        n_mha_win_size = -1
        self.scale_factor = 2
        self.with_ln= True
        self.attn_pdrop = 0.0
        self.proj_pdrop = 0.4
        self.path_pdrop = 0.1
        self.use_abs_pe = False
        self.use_rel_pe = False

        self.priors = 256  # 初始特征点数量
        self.layer_skip = 3
        self.layer = 4


        self.update(cfg)    # update ---------------------------------------------
        self.arch = (2, self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        self.mha_win_size = [n_mha_win_size] * (1 + self.arch[-1])
        print(f'self.arch: {self.arch}')

@register_backbone('Transformer')
class Transformer(nn.Module):
    def __init__(self, config: Transformer_config):
        super(Transformer, self).__init__()

        # Transformer Backbone
        self.backbone = ConvTransformerBackbone(
            n_in=512,  # Must match the output of the embedding layer
            n_embd=config.n_embd,
            n_head=config.n_head,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            max_len=config.max_len,
            mha_win_size=config.mha_win_size,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            attn_pdrop=config.attn_pdrop,
            proj_pdrop=config.proj_pdrop,
            path_pdrop=config.path_pdrop,
            use_abs_pe=config.use_abs_pe,
            use_rel_pe=config.use_rel_pe
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        feats, masks = self.backbone(x, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        return fpn_feats


@register_backbone_config('wifiTAD')
class WifiTAD_config(Config):
    def __init__(self, cfg = None):
        self.layer_num = 3
        self.input_length = 2048
        self.skip_ds_layer = 3
        self.priors = 128

        self.update(cfg)    # update ---------------------------------------------

@register_backbone('wifiTAD')
class WifiTAD(nn.Module):
    def __init__(self, config: WifiTAD_config):
        super(WifiTAD, self).__init__()

        self.layer_skip = config.skip_ds_layer
        self.skip_tsse = nn.ModuleList()

        self.layer_num = config.layer_num

        self.PyTSSE = nn.ModuleList()
        self.PyLSRE = nn.ModuleList()

        for i in range(self.layer_skip):
            self.skip_tsse.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2,
                                       length=(config.input_length // 2) // (2 ** i)))

        for i in range(self.layer_num):
            self.PyTSSE.append(TSSE(in_channels=512, out_channels=256, kernel_size=3, stride=2, length=config.priors//(2**i)))
            self.PyLSRE.append(LSREF(len=config.priors//(2**i),r=((config.input_length // 2)//config.priors)*(2**i)))


    def forward(self, embedd):

        deep_feat = embedd
        global_feat = embedd.detach()

        for i in range(len(self.skip_tsse)):
            deep_feat = self.skip_tsse[i](deep_feat)

        out_feats = []
        for i in range(self.layer_num):
            deep_feat = self.PyTSSE[i](deep_feat)
            out = self.PyLSRE[i](deep_feat, global_feat)
            out_feats.append(out)

        return out_feats
    
@register_backbone_config('TriDet')
class TriDet_config(Config):
    def __init__(self, cfg = None):
        # Backbone 配置
        self.n_embd = 512
        self.n_head = 8
        self.n_embd_ks = 3  # 卷积核大小
        self.max_len = 256
        # window size for self attention; <=1 to use full seq (ie global attention)
        n_sgp_win_size = 1
        self.scale_factor = 2
        self.with_ln= True
        self.attn_pdrop = 0.0
        self.proj_pdrop = 0.4
        self.path_pdrop = 0.1
        self.use_abs_pe = False
        self.use_rel_pe = False

        self.priors = 256  # 初始特征点数量
        self.layer_skip = 3
        self.layer = 4
        self.sgp_mlp_dim = 768
        self.downsample_type = 'max'
        self.init_conv_vars = 0
        self.k = 4

        self.update(cfg)    # update ---------------------------------------------
        self.arch = (2, self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        self.sgp_win_size = [n_sgp_win_size] * (1 + self.arch[-1])
        print(f'self.arch: {self.arch}')

@register_backbone('TriDet')
class TriDet(nn.Module):
    def __init__(self, config: TriDet_config):
        super(TriDet, self).__init__()

        # Transformer Backbone
        self.backbone = SGPBackbone(
            n_in=512,
            n_embd=config.n_embd,
            sgp_mlp_dim=config.sgp_mlp_dim,
            n_embd_ks=config.n_embd_ks,
            max_len=config.max_len,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            path_pdrop=config.path_pdrop,
            downsample_type=config.downsample_type,
            sgp_win_size=config.sgp_win_size,
            use_abs_pe=config.use_abs_pe,
            k=config.k,
            init_conv_vars=config.init_conv_vars
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        feats, masks = self.backbone(x, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        return fpn_feats

@register_backbone_config('TemporalMaxer')
class TemporalMaxer_config(Config):
    def __init__(self, cfg = None):
        # Backbone 配置
        self.n_embd = 512
        self.n_embd_ks = 3  # 卷积核大小
        self.max_len = 256
        self.scale_factor = 2
        self.with_ln= True

        self.priors = 256  # 初始特征点数量
        self.layer = 2

        self.update(cfg)    # update ---------------------------------------------
        self.arch = (self.layer, 4)  # 卷积层结构：基础卷积、stem 卷积、branch 卷积
        print(f'self.arch: {self.arch}')


@register_backbone('TemporalMaxer')
class TemporalMaxer(nn.Module):
    def __init__(self, config: TemporalMaxer_config):
        super(TemporalMaxer, self).__init__()

        # Transformer Backbone
        self.backbone = ConvPoolerBackbone(
            n_in=512,
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            max_len=config.max_len,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
        )

        # Neck: FPNIdentity
        self.neck = FPNIdentity(
            in_channels=[config.n_embd] * (config.arch[-1] + 1),  # 输入特征通道，假设每一层的输出特征通道一致
            out_channel=config.n_embd,  # 输出特征通道数
            scale_factor=config.scale_factor,  # 下采样倍率
            with_ln=config.with_ln  # 是否使用 LayerNorm
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize neck
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        batched_masks = torch.ones(B, 1, L, dtype=torch.bool).to(x.device)
        feats, masks = self.backbone(x, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)

        return fpn_feats
    
@register_backbone_config('Ushape')
class Ushape_config(Config):
    def __init__(self, cfg = None):
        # Backbone 配置
        self.in_channels = 128
        self.filters = [128, 256, 512, 1024, 2048, 4096]
        self.layers=3
        self.branch_layer=2
        self.update(cfg)    # update ---------------------------------------------


@register_backbone('Ushape')
class Ushape(nn.Module):
    def __init__(self, config: Ushape_config):
        super(Ushape, self).__init__()

        # Transformer Backbone in_channels = 64, branch_layer=4, layers=3, unet_branch_layers=2
        self.backbone = UNetBackbone2(
            in_channels=config.in_channels,
            layers=config.layers,
            branch_layer=4,
            unet_branch_layers=config.branch_layer
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize LayerNorm
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        # ([4, 512, 256])
        feats = self.backbone(x)
        # [torch.Size([4, 64, 256]), torch.Size([4, 64, 256]), torch.Size([4, 64, 256]), torch.Size([4, 64, 256]), torch.Size([4, 64, 256])]
        return feats