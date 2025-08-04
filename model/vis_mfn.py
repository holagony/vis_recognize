import torch
import torch.nn as nn
from model.branches.depth_branch import DPTSceneDepthBranch, SimpleSceneDepthBranch
from model.branches.transmission_branch import TransmissionBranch
from model.branches.spectral_branch import SpectralBranch
from model.branches.detail_branch import DetailBranch
from model.blocks import SFRB, MSFB, GFFB
from utils import config

class NNClassifier(nn.Module):
    """
    固定2层的分类器模块
    """
    def __init__(self, in_features, num_classes):
        super(NNClassifier, self).__init__()
        
        # 基于512输入优化分类器：512 -> 256 -> 128 -> 5
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes))
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(x)

class VisMFN(nn.Module):
    """
    VisMFN模型
    """
    def __init__(self,
                 num_visibility_levels=5,
                 sfrb_out_channels=config.SFRB_OUT_CHANNELS,
                 num_msfb_blocks=config.NUM_MSFB_BLOCKS,
                 gffb_out_channels=config.GFFB_OUT_CHANNELS,
                 img_size_tuple=config.TARGET_INPUT_SIZE,
                 device=torch.device("cpu"),
                 use_simple_depth=config.USE_SIMPLE_DEPTH
                 ):
        super().__init__()
        
        self.img_h, self.img_w = img_size_tuple
        self.img_size_tuple = img_size_tuple
        self.device = device
        self.use_simple_depth = use_simple_depth
        
        # 特征提取分支
        self._init_feature_branches()
        
        # 特征融合模块
        self._init_fusion_modules(sfrb_out_channels, num_msfb_blocks, gffb_out_channels)
        
        # 分类器（固定2层）
        self._init_classifier(gffb_out_channels, num_visibility_levels)
        
        # 移动到指定设备
        self.to(self.device)

    def _init_feature_branches(self):
        """
        初始化特征提取分支
        """
        # 场景深度分支 - 根据配置选择
        if self.use_simple_depth:
            self.scene_depth_branch = SimpleSceneDepthBranch(model_weight_path=config.SIMPLE_DEPTH_MODEL_PATH, device=self.device)
        else:
            self.scene_depth_branch = DPTSceneDepthBranch(dpt_model_type="dpt_hybrid", device=self.device)

        # 其他分支
        self.transmission_branch = TransmissionBranch(
            omega=config.TRANSMISSION_OMEGA, 
            patch_size=config.TRANSMISSION_PATCH_SIZE,
            guided_radius=config.TRANSMISSION_GUIDED_RADIUS, 
            guided_eps=config.TRANSMISSION_GUIDED_EPS).to(self.device)
        
        self.spectral_branch = SpectralBranch(
            enhancement_factor=config.SPECTRAL_ENHANCEMENT_FACTOR).to(self.device)
        
        self.detail_branch = DetailBranch(
            guided_radius=config.DETAIL_GUIDED_RADIUS, 
            guided_eps=config.DETAIL_GUIDED_EPS).to(self.device)

    def _init_fusion_modules(self, sfrb_out_channels, num_msfb_blocks, gffb_out_channels):
        """
        初始化特征融合模块
        """
        # 各分支通道数：深度(1) + 透射图(1) + 光谱(3) + 细节(3) + RGB(3) = 11
        sfrb_in_channels = 1 + 1 + 3 + 3 + 3
        
        # SFRB: 浅层特征表示块
        self.sfrb = SFRB(sfrb_in_channels, sfrb_out_channels).to(self.device) # out目前64
        
        # MSFB: 多尺度融合块 - 渐进式通道增长
        self.msfb_blocks = nn.ModuleList()
        current_channels = sfrb_out_channels  # 起始通道数64
        self.msfb_output_channels = []
        
        for i in range(num_msfb_blocks):
            # 计算当前块的输出通道数
            multiplier = config.MSFB_CHANNEL_MULTIPLIERS[i] if i < len(config.MSFB_CHANNEL_MULTIPLIERS) else config.MSFB_CHANNEL_MULTIPLIERS[-1]
            out_channels = int(sfrb_out_channels * multiplier)
            
            self.msfb_blocks.append(MSFB(current_channels, out_channels).to(self.device))
            self.msfb_output_channels.append(out_channels)
            current_channels = out_channels  # 更新下一块的输入通道数
        
        # GFFB: 全局特征融合块 - 输入为所有MSFB输出的拼接
        gffb_input_channels = sum(self.msfb_output_channels)  # MSFB1(128) + MSFB2(256) = 384
        self.gffb = GFFB(gffb_input_channels, gffb_out_channels).to(self.device)
        print(f"GFFB: {gffb_input_channels} → {gffb_out_channels} 通道")

    def _init_classifier(self, gffb_out_channels, num_visibility_levels):
        """
        初始化分类器（固定2层）
        """
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = NNClassifier(in_features=gffb_out_channels, num_classes=num_visibility_levels)

    def extract_modal_features(self, x_rgb_tensor):
        """
        提取各模态特征
        """
        detail_feat = self.detail_branch(x_rgb_tensor) # 细节特征
        spectral_feat = self.spectral_branch(x_rgb_tensor) # 光谱特征
        trans_feat = self.transmission_branch(x_rgb_tensor) # 传输图特征
        depth_feat = self.scene_depth_branch(x_rgb_tensor) # 深度特征

        return {'detail': detail_feat,
                'spectral': spectral_feat,
                'transmission': trans_feat,
                'depth': depth_feat,
                'rgb': x_rgb_tensor,}

    def forward(self, x_rgb_tensor):
        modal_features = self.extract_modal_features(x_rgb_tensor)
        
        balanced_features = [modal_features['detail'],
                             modal_features['spectral'],
                             modal_features['transmission'],
                             modal_features['depth'],
                             modal_features['rgb']]
        
        concatenated_features = torch.cat(balanced_features, dim=1) # 平衡后特征拼接
        sfrb_out = self.sfrb(concatenated_features) # 浅层特征表示

        # 多尺度特征融合
        msfb_intermediate_outputs = []
        current_msfb_input = sfrb_out
        for msfb_block in self.msfb_blocks:
            current_msfb_input = msfb_block(current_msfb_input)
            msfb_intermediate_outputs.append(current_msfb_input)

        # 拼接所有MSFB输出进行全局特征融合
        concatenated_msfb_features = torch.cat(msfb_intermediate_outputs, dim=1)
        gffb_out = self.gffb(concatenated_msfb_features) # 全局特征融合
        pooled_out = self.global_pool(gffb_out)
        pooled_out_flat = torch.flatten(pooled_out, 1) # 全局平均池化和分类
        output = self.classifier(pooled_out_flat) # 分类预测

        return output

    def get_model_info(self):
        """
        获取模型信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "VisMFN_Enhanced",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_msfb_blocks": len(self.msfb_blocks),
            "msfb_channels": " → ".join(map(str, self.msfb_output_channels)),
            "max_channels": max(self.msfb_output_channels + [config.GFFB_OUT_CHANNELS]),
            "gffb_out_channels": config.GFFB_OUT_CHANNELS,
            "img_size": self.img_size_tuple,
            "num_feature_branches": 5}