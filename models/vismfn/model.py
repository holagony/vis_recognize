import torch
import torch.nn as nn
from datasets.feature_extraction import feature_extraction_block
from models.vismfn.cnn_block import SFRB, MSFB, GFFB
from models.vismfn.attention_block import CBAM
from utils import config

class VisMFN(nn.Module):
    def __init__(self,
                 scene_depth_weights_path,
                 num_visibility_levels=5,
                 sfrb_out_channels=config.SFRB_OUT_CHANNELS,
                 num_msfb_blocks=config.NUM_MSFB_BLOCKS,
                 gffb_out_channels=config.GFFB_OUT_CHANNELS,
                 img_size=config.TARGET_INPUT_SIZE,
                 device=config.DEVICE):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.device = device

        sfrb_in_channels = 3 + 3 + 16 + 3 + 1
        self.sfrb = SFRB(sfrb_in_channels, sfrb_out_channels).to(self.device)

        self.msfb_blocks = nn.ModuleList()
        for _ in range(num_msfb_blocks): 
            self.msfb_blocks.append(MSFB(sfrb_out_channels).to(self.device))

        self.gffb = GFFB(num_msfb_blocks, sfrb_out_channels, gffb_out_channels).to(self.device)

        self.attention_module = CBAM(gffb_out_channels).to(self.device) 
   
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        fc_in_features = gffb_out_channels
        
        fc_layers_list = []

        fc_layers_list.extend([
                nn.Linear(fc_in_features, fc_in_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(fc_in_features, fc_in_features // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(fc_in_features // 2, num_visibility_levels)])

        self.fc_layers = nn.Sequential(*fc_layers_list).to(self.device)
        
    def forward(self, x_ori, x_aug):
        init_features, num_channels = feature_extraction_block(x_ori, x_aug) # 提取特征
        sfrb_out = self.sfrb(init_features)
        
        msfb_outputs = []
        msfb_input = sfrb_out
        for msfb_block in self.msfb_blocks:
            msfb_input = msfb_block(msfb_input)
            msfb_outputs.append(msfb_input)
            
        gffb_out = self.gffb(msfb_outputs)

        # 应用空间注意力
        if self.spatial_attention_module is not None: # 检查模块是否存在
            attended_features = self.spatial_attention_module(gffb_out)
        else:
            attended_features = gffb_out
            
        pooled_out = self.global_avg_pool(attended_features)
        pooled_out_flat = torch.flatten(pooled_out, 1)
        output_logits = self.fc_layers(pooled_out_flat)
        
        return output_logits