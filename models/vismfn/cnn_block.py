import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Shallow Feature Representation Block (SFRB) ---
class SFRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) # 增加BN层
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) # 增加BN层
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x_concat):
        x = self.relu1(self.bn1(self.conv1(x_concat)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

# --- Multi-Scale Fusion Block (MSFB) ---
class DilatedConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(channels) # 增加BN层
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MSFB(nn.Module):
    def __init__(self, channels): # channels 是输入M_{n-1}的通道数，也是C1, C2, C3的通道数
        super().__init__()
        # C1 part (Equation 8)
        self.dil_conv1_c1 = DilatedConvBlock(channels, dilation=1)
        self.dil_conv2_c1 = DilatedConvBlock(channels, dilation=2)
        self.dil_conv3_c1 = DilatedConvBlock(channels, dilation=3)
        # 输入是 channels*3, 输出是 channels
        self.fc1_c1 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.bn_c1_reduce = nn.BatchNorm2d(channels)

        # C2 part (Equation 9) - input is C1 (channels)
        self.dil_conv1_c2 = DilatedConvBlock(channels, dilation=1)
        self.dil_conv2_c2 = DilatedConvBlock(channels, dilation=2)
        self.dil_conv3_c2 = DilatedConvBlock(channels, dilation=3)
        self.fc1_c2 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.bn_c2_reduce = nn.BatchNorm2d(channels)

        # C3 part (Equation 10) - input is concat(C1, C2) (channels*2)
        self.fc1_c3 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bn_c3_reduce = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, M_n_minus_1):
        # C1 (Eq 8)
        fd1_mn1 = self.dil_conv1_c1(M_n_minus_1)
        fd2_mn1 = self.dil_conv2_c1(M_n_minus_1)
        fd3_mn1 = self.dil_conv3_c1(M_n_minus_1)
        concat1 = torch.cat([fd1_mn1, fd2_mn1, fd3_mn1], dim=1)
        C1 = self.relu(self.bn_c1_reduce(self.fc1_c1(concat1)))

        # C2 (Eq 9)
        fd1_c1 = self.dil_conv1_c2(C1)
        fd2_c1 = self.dil_conv2_c2(C1)
        fd3_c1 = self.dil_conv3_c2(C1)
        concat2 = torch.cat([fd1_c1, fd2_c1, fd3_c1], dim=1)
        # 增加残差连接：C2 = F(C1 + F(concat2))
        C2_residual = self.fc1_c2(concat2)
        C2 = self.relu(self.bn_c2_reduce(C2_residual + C1))

        # C3 (Eq 10)
        concat3 = torch.cat([C1, C2], dim=1)
        C3 = self.relu(self.bn_c3_reduce(self.fc1_c3(concat3)))

        # Output M_n (Eq 11)
        M_n = M_n_minus_1 + C3 # Residual connection
        return M_n

# --- Global Feature Fusion Block (GFFB) ---
class GFFB(nn.Module):
    def __init__(self, num_msfb_outputs, channels_per_msfb, out_channels):
        super().__init__()
        self.conv_fuse = nn.Conv2d(num_msfb_outputs * channels_per_msfb, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels) # 增加BN层
        self.relu = nn.ReLU(inplace=True)

    def forward(self, msfb_outputs_list): # list of M1, M2, ... Mn
        concatenated_features = torch.cat(msfb_outputs_list, dim=1)
        fused_features = self.relu(self.bn(self.conv_fuse(concatenated_features)))
        return fused_features