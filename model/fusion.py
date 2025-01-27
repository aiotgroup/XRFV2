import torch
import torch.nn as nn

class GatedFusion(nn.Module):
    def __init__(self, hidden_size: int):
        """
        使用门控机制的模态融合模块。
        Args:
            hidden_size (int): 特征的隐藏维度 (C)。
        """
        super(GatedFusion, self).__init__()
        self.gate_linear = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)  # 1D 卷积等价于线性层
        self.gate = nn.Sigmoid()  # 门控激活函数
        self.fc = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1)  # 融合后特征映射回 hidden_size

    def forward(self, imu_features: torch.Tensor, wifi_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。
        Args:
            imu_features (torch.Tensor): IMU 模态特征，形状 (B, C, L)。
            wifi_features (torch.Tensor): WiFi 模态特征，形状 (B, C, L)。
        Returns:
            torch.Tensor: 融合后的特征，形状 (B, C, L)。
        """
        # 计算门控值
        gate_imu = self.gate(self.gate_linear(imu_features))  # (B, C, L)
        gate_wifi = self.gate(self.gate_linear(wifi_features))  # (B, C, L)

        # 元素级融合
        gated_imu = gate_imu * imu_features  # (B, C, L)
        gated_wifi = gate_wifi * wifi_features  # (B, C, L)

        # 拼接特征
        combined_features = torch.cat([gated_imu, gated_wifi], dim=1)  # (B, C * 2, L)

        # 映射到输出空间
        output = self.fc(combined_features)  # (B, C, L)
        return output


class GatedFusionAdd(nn.Module):
    def __init__(self, hidden_size: int):
        """
        使用门控机制的模态融合模块。
        Args:
            hidden_size (int): 特征的隐藏维度 (C)。
        """
        super(GatedFusionAdd, self).__init__()
        self.gate_linear = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)  # 1D 卷积等价于线性层
        self.gate = nn.Sigmoid()  # 门控激活函数
        self.fc = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1)  # 融合后特征映射回 hidden_size

    def forward(self, imu_features: torch.Tensor, wifi_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。
        Args:
            imu_features (torch.Tensor): IMU 模态特征，形状 (B, C, L)。
            wifi_features (torch.Tensor): WiFi 模态特征，形状 (B, C, L)。
        Returns:
            torch.Tensor: 融合后的特征，形状 (B, C, L)。
        """
        # 计算门控值
        gate_imu = self.gate(self.gate_linear(imu_features))  # (B, C, L)
        gate_wifi = self.gate(self.gate_linear(wifi_features))  # (B, C, L)

        # 元素级融合
        gated_imu = gate_imu * imu_features  # (B, C, L)
        gated_wifi = gate_wifi * wifi_features  # (B, C, L)

        # 拼接特征
        # combined_features = torch.cat([gated_imu, gated_wifi], dim=1)  # (B, C * 2, L)
        combined_features = gated_imu + gated_wifi

        # 映射到输出空间
        # output = self.fc(combined_features)  # (B, C, L)
        return combined_features

class GatedFusionAdd2(nn.Module):
    def __init__(self, hidden_size: int):
        """
        使用门控机制的模态融合模块。
        Args:
            hidden_size (int): 特征的隐藏维度 (C)。
        """
        super(GatedFusionAdd2, self).__init__()
        # print('hidden_size', hidden_size)
        self.gate_linear = nn.Linear(hidden_size, hidden_size * 2)  # 1D 卷积等价于线性层
        self.gate = nn.Sigmoid()  # 门控激活函数
        self.fc = nn.Linear(hidden_size * 2, hidden_size)  # 融合后特征映射回 hidden_size

    def forward(self, imu_features: torch.Tensor, wifi_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。
        Args:
            imu_features (torch.Tensor): IMU 模态特征，形状 (B, C, L)。
            wifi_features (torch.Tensor): WiFi 模态特征，形状 (B, C, L)。
        Returns:
            torch.Tensor: 融合后的特征，形状 (B, C, L)。
        """
        # 计算门控值
        # print(imu_features.shape, wifi_features.shape)
        # print(combined_features.shape)
        gate = self.gate(self.gate_linear(imu_features + wifi_features))
        combined = torch.cat([imu_features, wifi_features], dim=-1)

        fused_features = gate * combined

        output = self.fc(fused_features)

        return output

class GatedFusionWeight(nn.Module):
    def __init__(self, hidden_size: int):
        """
        使用门控机制的模态融合模块。
        Args:
            hidden_size (int): 特征的隐藏维度 (C)。
        """
        super(GatedFusionWeight, self).__init__()
        # self.gate_linear = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)  # 1D 卷积等价于线性层
        # self.gate = nn.Sigmoid()  # 门控激活函数
        # self.fc = nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1)  # 融合后特征映射回 hidden_size

    def forward(self, imu_features: torch.Tensor, wifi_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑。
        Args:
            imu_features (torch.Tensor): IMU 模态特征，形状 (B, C, L)。
            wifi_features (torch.Tensor): WiFi 模态特征，形状 (B, C, L)。
        Returns:
            torch.Tensor: 融合后的特征，形状 (B, C, L)。
        """
        # 计算门控值
        # gate_imu = self.gate(self.gate_linear(imu_features))  # (B, C, L)
        # gate_wifi = self.gate(self.gate_linear(wifi_features))  # (B, C, L)

        # 元素级融合
        gated_imu = 0.8 * imu_features  # (B, C, L)
        gated_wifi = 0.2 * wifi_features  # (B, C, L)

        # 拼接特征
        # combined_features = torch.cat([gated_imu, gated_wifi], dim=1)  # (B, C * 2, L)
        combined_features = gated_imu + gated_wifi

        # 映射到输出空间
        # output = self.fc(combined_features)  # (B, C, L)
        return combined_features