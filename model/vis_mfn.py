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
        
        # 固定为2层：输入 -> 512 -> 256 -> 输出
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, num_classes))
        
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
            self.depth_channel_expansion = None  # SimpleSceneDepthBranch 已经输出16通道
            print("使用轻量级深度分支 16通道输出")
        else:
            self.scene_depth_branch = DPTSceneDepthBranch(dpt_model_type="dpt_hybrid", device=self.device)
            # 为DPT添加1x1卷积将1通道扩展到16通道
            self.depth_channel_expansion = nn.Conv2d(1, 16, kernel_size=1, bias=False).to(self.device)
            print("使用DPT深度分支 1通道 + 1x1卷积扩展到16通道")
        
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
        # 计算SFRB输入通道数: 深度(16) + 透射图(1) + 光谱(3) + 细节(3) + RGB(3) = 74
        sfrb_in_channels = 16 + 1 + 3 + 3 + 3
        
        # SFRB: 浅层特征表示块
        self.sfrb = SFRB(sfrb_in_channels, sfrb_out_channels).to(self.device) # out目前128
        
        # MSFB: 多尺度融合块
        self.msfb_blocks = nn.ModuleList()
        for _ in range(num_msfb_blocks):
            self.msfb_blocks.append(MSFB(sfrb_out_channels).to(self.device)) # out目前128
        
        # GFFB: 全局特征融合块（保留通道注意力）
        self.gffb = GFFB(num_msfb_blocks, sfrb_out_channels, gffb_out_channels).to(self.device) # out目前512

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
        depth_feat_raw = self.scene_depth_branch(x_rgb_tensor)
        
        # 如果使用DPT分支，需要通过1x1卷积扩展通道数
        if self.depth_channel_expansion is not None:
            depth_feat = self.depth_channel_expansion(depth_feat_raw)
        else:
            depth_feat = depth_feat_raw
        
        return {'detail': detail_feat,
                'spectral': spectral_feat,
                'depth': depth_feat,
                'rgb': x_rgb_tensor,
                'transmission': trans_feat}

    def forward(self, x_rgb_tensor):
        modal_features = self.extract_modal_features(x_rgb_tensor) # 提取各模态特征 + 原始图片
        concatenated_features = torch.cat([modal_features['detail'],
                                           modal_features['spectral'], 
                                           modal_features['depth'],
                                           modal_features['rgb'],
                                           modal_features['transmission']], dim=1) # 特征拼接
        sfrb_out = self.sfrb(concatenated_features) # 浅层特征表示

        # 多尺度特征融合
        msfb_intermediate_outputs = []
        current_msfb_input = sfrb_out
        for msfb_block in self.msfb_blocks:
            current_msfb_input = msfb_block(current_msfb_input)
            msfb_intermediate_outputs.append(current_msfb_input)

        gffb_out = self.gffb(msfb_intermediate_outputs) # 全局特征融合
        pooled_out = self.global_pool(gffb_out)
        pooled_out_flat = torch.flatten(pooled_out, 1) # 全局平均池化和分类
        output = self.classifier(pooled_out_flat) # 分类预测

        return output

    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "VisMFN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_msfb_blocks": len(self.msfb_blocks),
            "gffb_out_channels": self.gffb.feature_enhance[0].out_channels,
            "img_size": self.img_size_tuple,
            "num_feature_branches": 5,  # 深度、透射图、光谱、细节、RGB
            "use_attention": f"每个MSFB块使用CBAM + GFFB中使用CBAM（共{len(self.msfb_blocks)+1}个CBAM）"
        }

    def extract_features(self, x_rgb_tensor):
        """
        提取中间特征用于可视化和分析
        """
        modal_features = self.extract_modal_features(x_rgb_tensor)
        
        concatenated_features = torch.cat([
            modal_features['detail'],
            modal_features['spectral'], 
            modal_features['depth'],
            modal_features['rgb'],
            modal_features['transmission']], dim=1)
        
        sfrb_out = self.sfrb(concatenated_features)
        
        msfb_outputs = []
        current_input = sfrb_out
        for msfb_block in self.msfb_blocks:
            current_input = msfb_block(current_input)
            msfb_outputs.append(current_input)
        
        gffb_out = self.gffb(msfb_outputs)
        
        return {'modal_features': modal_features,
                'sfrb_output': sfrb_out,
                'msfb_outputs': msfb_outputs,
                'gffb_output': gffb_out}