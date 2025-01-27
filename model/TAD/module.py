import torch
import torch.nn as nn
import torch.nn.functional as F
from model.TAD.atten import FullAttention, AttentionLayer, FullAttention_new
from model.TAD.encoder import Encoder2, EncoderLayer2, Encoder3, EncoderLayer3

class ScaleExp(nn.Module):
    '''
    Different layers regression to different size range
    Learn a trainable scalar to automatically adjust the base of exp(si * x)
    '''
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)

class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input to Unit1D")
            raise ValueError("NaN detected in input to Unit1D")

        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            if pad_t < 0:
                print(f"Negative padding detected: {pad_t}")
                raise ValueError(f"Negative padding detected: {pad_t}")
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
            if torch.isnan(x).any():
                print("NaN detected after padding")
                raise ValueError("NaN detected after padding")

        weights = self.conv1d.weight
        if torch.isnan(weights).any():
            print("NaN detected in Conv1d weights")
            raise ValueError("NaN detected in Conv1d weights")

        x = self.conv1d(x)
        if torch.isnan(x).any():
            print("NaN detected after Conv1d")
            raise ValueError("NaN detected after Conv1d")

        if self._activation_fn is not None:
            x = self._activation_fn(x)
            if torch.isnan(x).any():
                print("NaN detected after activation function")
                raise ValueError("NaN detected after activation function")

        return x


class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x


class PoolConv(nn.Module):
    def __init__(self, in_channels):
        super(PoolConv, self).__init__()
        self.dwconv1 = nn.Sequential(
            Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None),
            nn.ReLU(inplace=True))
        self.max = nn.Sequential(nn.MaxPool1d(3,2,1), nn.Sigmoid())
        self.conv = Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None)
        self.conv2 = Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None)
        self.norm = nn.GroupNorm(32, in_channels)
        self.lu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.dwconv1(x)
        y = self.norm(self.max(y)*self.conv(y))
        y = self.conv2(y)

        return  y

 
class ds(nn.Module):
    def __init__(self, in_channels):
        super(ds, self).__init__()
        self.dwconv1 = nn.Sequential(Unit1D(in_channels=in_channels,
                        output_channels=in_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None),
                        nn.GroupNorm(32, in_channels),
                        nn.ReLU(inplace=True)
                        )

    def forward(self, x):
        x = self.dwconv1(x)
        return  x


class Cat_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Cat_Fusion, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, 1, bias=True),
                        Unit1D(in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=False,
                        activation_fn=None),
                        nn.PReLU())

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-2)
        x = self.conv(x)
        return  x


class joint_attention(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, length, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(joint_attention, self).__init__()
        if attn == "full":
            self.cross_atten = Encoder2(
                [
                    EncoderLayer2(
                        AttentionLayer(FullAttention_new(len=length, n_heads=n_heads, attention_dropout=dropout, output_attention=output_attention), 
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                norm_layer=None
            )
        else:
            self.cross_atten = Encoder2(
                [
                    EncoderLayer2(
                        AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                       d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(e_layers)
                ],
                norm_layer=None
            )
    
    def forward(self, q, k, v):
        out, _ = self.cross_atten(q, k, v, attn_mask=None)
        out = out.permute(0, 2, 1)
        return out
  