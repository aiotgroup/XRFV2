import torch
import torch.nn as nn
from model.TAD.module import Unit1D
from model.TAD.atten import FullAttention, AttentionLayer, FullAttention_new
from model.TAD.encoder import Encoder2, EncoderLayer2, Encoder3, EncoderLayer3
from model.TAD.module import ContraNorm, PoolConv, Cat_Fusion, joint_attention, ds


class TSSE(nn.Module):
    def __init__(self, in_channels, out_channels, length, kernel_size=3, stride=2):
        super(TSSE, self).__init__()
        self.Downscale = ds(512)

        self.pconv = PoolConv(512)
        self.self_attention = joint_attention(enc_in=in_channels, d_model=in_channels, n_heads=16,length=length, d_ff=in_channels*4, e_layers=1, factor=3, dropout=0.01, output_attention=False, attn='full')
        
        self.c1 = Cat_Fusion(1024, 1024)
        self.c2 = Cat_Fusion(1024, 1024)
        self.c3 = Cat_Fusion(2048, 512)
        self.contra_norm = ContraNorm(dim=length, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)

    def forward(self, time):
        high = self.pconv(time)
        time2 = self.Downscale(time)
        low = self.self_attention(time2, time2, time2)
        high2 = self.c1(low, high)
        low2 = self.c2(high, low)
        out = self.c3(high2, low2)
        out = self.contra_norm(out)
        return out
   
    
class LSREF(nn.Module):
    def __init__(self, len,r):
        super(LSREF, self).__init__()
        self.r = r
        self.len = len
        self.conv = nn.Conv1d(512,512,1,1)
        self.beta = nn.Parameter(torch.ones(1))
        self.crs = CrossPyramidFusion(length=1,enc_in=512, d_model=512, n_heads=16, d_ff=512*4, e_layers=1, factor=3, dropout=0.01, output_attention=False, attn='full2')


    def forward(self, x, global_feat):
        return self.sliding_window_gaussian(x, global_feat, self.r)
    
    def sliding_window_gaussian(self, feat, global_feat, r):
        batch_size, channels, length = global_feat.shape
        step = 2 * r
        window_size = 2 * r
        
        num_windows = (length - window_size) // step + 1
        result = torch.zeros(batch_size, channels, num_windows, dtype=torch.float32).to(feat.device)
        
        for i in range(0, length - window_size + 1, step):
            window = global_feat[:, :, i:i + window_size]  # Extract window directly
            
            # Calculate maximum and minimum values in the window
            window_max = window.max(dim=-1, keepdim=True)[0]
            window_min = window.min(dim=-1, keepdim=True)[0]
            
            # Calculate Euclidean distance between max and min values
            euclidean_dist = torch.norm(window_max - window_min, p=2, dim=-1) + 1e-6
            
            # Determine corresponding index in result tensor
            idx = i // step
            
            # Store the Euclidean distance values in result tensor
            result[:, :, idx] = euclidean_dist
        
        # Apply your subsequent operations
        result = torch.relu(self.conv(result))

        # print(f'global_feat.shape {global_feat.shape}, r: {r}')
        # print(f'feat.shape: {feat.shape}', f'result.shape: {result.shape}')

        new_feat = self.crs(feat, result)
        return new_feat


class CrossPyramidFusion(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, d_ff, e_layers, length, factor=3, dropout=0.1, output_attention=False, attn='full', activation='gelu', distil=False):
        super(CrossPyramidFusion, self).__init__()
        self.cross_atten = Encoder3(
            [
                EncoderLayer3(
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
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
    
    def forward(self, feat1, feat2):
        out, _ = self.cross_atten(feat1, feat2, attn_mask=None)
        out = out.permute(0, 2, 1)
        return out
