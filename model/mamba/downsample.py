import torch
import torch.nn as nn
import torch.nn.functional as F


class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_size=1,
                 stride=1,
                 padding='same',
                 activation_fn=None,
                 use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_size,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_size = kernel_size

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_size - self._stride, 0)
        else:
            return max(self._kernel_size - (t % self._stride), 0)

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

        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)

        return x


class ds(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2, activation_fn=None):
        super(ds, self).__init__()

        # If activation_fn is not provided, default to None
        if activation_fn is None:
            self.activation_fn = None
        else:
            self.activation_fn = activation_fn

        # Adjust GroupNorm based on input channels
        self.dwconv1 = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=in_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   activation_fn=None),  # No activation in Unit1D
            nn.GroupNorm(num_groups=min(32, in_channels), num_channels=in_channels),
        )

    def forward(self, x):
        x = self.dwconv1(x)

        # Apply activation function if provided
        if self.activation_fn is not None:
            x = self.activation_fn(x)

        return x


class Downsample(nn.Module):
    def __init__(
            self,
            in_channels,  # 输入特征通道数，列表形式，表示每一层的特征通道
            out_channel,  # 输出特征通道数，保持不变
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步幅为2
            num_steps=3,  # 降采样步数，每个特征图经历 num_steps 次卷积
            activation_fn=F.relu  # 激活函数
    ):
        super(Downsample, self).__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_steps = num_steps
        self.activation_fn = activation_fn

        # 为每个输入特征图创建降采样层
        self.downsample_layers = nn.ModuleList()

        for _ in range(len(in_channels)):  # 对每个输入特征图生成降采样层
            # 构建每层的降采样步骤
            downsample_block = []
            for _ in range(num_steps):  # num_steps 步骤的卷积层
                downsample_block.append(
                    nn.Conv1d(
                        in_channels=in_channels[0],  # 输入通道固定，假设每个层的输入通道相同
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=kernel_size // 2  # 保持序列长度
                    )
                )
                downsample_block.append(
                    nn.GroupNorm(num_groups=min(32, out_channel), num_channels=out_channel)
                )
                if self.activation_fn is not None:
                    downsample_block.append(nn.ReLU(inplace=True))  # 使用 nn.ReLU 而非 F.relu

            # 将每个 downsample_block 放入 ModuleList
            self.downsample_layers.append(nn.Sequential(*downsample_block))

    def forward(self, feats, masks):
        """
        Args:
            feats (List[Tensor]): List of feature maps (B, C, T) at each level.
            masks (List[Tensor]): List of masks (B, 1, T) at each level.
        """
        downsampled_feats = []
        downsampled_masks = []

        for i, layer in enumerate(self.downsample_layers):
            # Apply downsampling block to each feature map
            feat = feats[i]
            mask = masks[i]

            # Apply the downsampling layers to the feature map
            downsampled_feat = layer(feat)

            # Adjust the mask's length after downsampling, by using max pooling or similar operation
            for _ in range(self.num_steps):  # Apply max_pool1d `num_steps` times
                mask = F.max_pool1d(mask.float(), kernel_size=self.stride, stride=self.stride)

            mask = mask.bool()
            downsampled_feats.append(downsampled_feat)
            downsampled_masks.append(mask)

        return downsampled_feats, downsampled_masks