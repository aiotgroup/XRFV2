import torch
import torch.nn as nn
import numpy as np
from math import sqrt

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import time

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask, plot_scores=False):  # Removed save_path parameter
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        assert not torch.isnan(scores).any(), "NaN detected in scores after einsum"

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        scores = torch.clamp(scores, min=-1e6, max=1e6)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)


        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
    
    def plot_attention_scores(self, scores, prefix):
        # Take the first batch and first head for simplicity
        score_matrix = scores[0, 0].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(score_matrix, aspect='auto', cmap='viridis', extent=[0, score_matrix.shape[1], score_matrix.shape[0], 0])
        plt.colorbar()
        plt.title('Attention Scores Heatmap')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Use a unique filename based on the prefix and current time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f'{prefix}_attention_scores_heatmap_{timestamp}.png'
        plt.savefig(filename)  # Save the figure with a unique name
        plt.close()


class CrossAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CrossAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        assert not torch.isnan(scores).any(), "NaN detected in scores after einsum"
        assert not torch.isinf(scores).any(), "Inf detected in scores after einsum"
        scores = torch.clamp(scores, min=-1e3, max=1e3)
        scores = torch.pow(scores, 3)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)
        scores = scores - scores.max(dim=-1, keepdim=True).values  # Normalize for stability
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class AttentionLayer2(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer2, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.query_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, queries2, keys2, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        queries2 = self.query_projection2(queries2).view(B, L, H, -1)
        keys2 = self.key_projection2(keys2).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            queries2,
            keys2,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class CrossAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(CrossAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.max = nn.MaxPool1d(3,1,1)
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries)
        queries = queries.view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention2(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention2, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys,queries2, keys2, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores2 = torch.sigmoid(torch.einsum("blhe,bshe->bhls", queries2, keys2))
        scores = scores
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A1 = torch.softmax(scale * scores*scale * scores2, dim=-1)
        # A2 = torch.sigmoid()
        A = self.dropout(A1)
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)



class Cross_Attention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(Cross_Attention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, q1, k1, v1, q2, k2, v2, attn_mask):
        B, L, H, E = q1.shape
        _, S, _, D = v1.shape
        scale = self.scale or 1./sqrt(E)

        scores1 = torch.einsum("blhe,bshe->bhls", q1, k1)
        scores2 = torch.einsum("blhe,bshe->bhls", q2, k2)

        A1 = self.dropout(torch.tanh(scores1 * scale))
        A2 = self.dropout(torch.sigmoid(scores2 * scale))
        A = A1*A2
        Value1 = torch.einsum("bhls,bshd->blhd", A, v1)
        # Value2 = torch.einsum("bhls,bshd->blhd", A2, v2)
        # V = self.theta*Value1 + Value2
        
        return (Value1.contiguous(), None)


class AttentionLayer3(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer3, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection1 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection1 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection1 = nn.Linear(d_model, d_values * n_heads)
        self.query_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection2 = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection2 = nn.Linear(d_model, d_values * n_heads)
        self.out_projection1 = nn.Linear(d_values * n_heads, d_model)
        self.out_projection2 = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, q1, k1, v1, q2, k2, v2, attn_mask):
        B, L, _ = q1.shape
        _, S, _ = k1.shape
        H = self.n_heads
        q1 = self.query_projection1(q1).view(B, L, H, -1)
        k1 = self.key_projection1(k1).view(B, S, H, -1)
        v1 = self.value_projection1(v1).view(B, S, H, -1)

        q2 = self.query_projection2(q2).view(B, L, H, -1)
        k2 = self.key_projection2(k2).view(B, S, H, -1)
        v2 = self.value_projection2(v2).view(B, S, H, -1)

        out1, attn = self.inner_attention(
            q1,k1,v1,q2,k2,v2,attn_mask
        )
        out1 = out1.view(B, L, -1)
        out = self.out_projection1(out1)

        return out, attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
    

class FullAttention_new(nn.Module):
    def __init__(self, len, n_heads=16, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention_new, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        # self.beta = nn.Parameter(torch.ones(1))
        self.w = nn.Parameter(torch.empty(1, len, n_heads, n_heads*2))  # Define w as a learnable parameter
        nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain('sigmoid'))

        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        combined = torch.stack((queries, keys), dim=-1)  
        fro_norm = torch.norm(combined, p=1, dim=-1)
        scores2 = torch.einsum("blhe,bshe->bhls", fro_norm, self.w)

        # print(f'queries shape: {queries.shape}')
        # print(f'key shape: {keys.shape}')
        # print(f"fro_norm shape: {fro_norm.shape}")  # [B, L, H]
        # print(f"self.w shape: {self.w.shape}")  # 检查 len 和 n_heads 的一致性
        # print(f"scores shape: {scores.shape}")  # 可能是 [B, H, L, S]
        # print(f"scores2 shape: {scores2.shape}")  # 可能是 [B, H, L, D]

        # Gaussian attention scores
        scores = torch.clamp(scores, min=-1e3, max=1e3)
        scores2 = torch.clamp(scores2, min=-1e3, max=1e3)

        scores_gaussian = torch.tanh(scale*scores)*torch.sigmoid(scale * scores2)

        # Attention weights and values
        scores_gaussian = scores_gaussian - scores_gaussian.max(dim=-1, keepdim=True).values

        A = self.dropout(torch.softmax(scores_gaussian, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

