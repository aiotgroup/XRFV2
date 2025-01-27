import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pad):
        super(TokenEmbedding, self).__init__()
        # padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        padding = pad
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pad, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, pad=pad)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
    
    
class DatawithoutPOSEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pad, dropout=0.1):
        super(DatawithoutPOSEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, pad=pad)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        return self.dropout(x)
    
class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels=512, stride=1):
        super(Embedding, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, time):
        time = self.embedding(time)
        return time
