import torch.nn as nn
from models.wuhan.psa import SequentialPolarizedSelfAttention

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEModule(nn.Module):
    '''
    SE-Net 注意力模块
    '''

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_se=False, se_reduction=16, use_psa=False):
        super(BasicBlock, self).__init__()
        self.use_se = use_se
        self.use_psa = use_psa

        # 支持空洞卷积
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)

        # SE注意力模块
        if use_se:
            self.se = SEModule(planes, se_reduction)
        if use_psa:
            self.psa = SequentialPolarizedSelfAttention(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用SE注意力
        if self.use_se:
            out = self.se(out)

        if self.use_psa:
            out = self.psa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_se=False, se_reduction=16, use_psa=False):
        super(Bottleneck, self).__init__()
        self.use_se = use_se
        self.use_psa = use_psa

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 支持空洞卷积
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        # SE注意力模块
        if use_se:
            self.se = SEModule(planes * 4, se_reduction)
        if use_psa:
            self.psa = SequentialPolarizedSelfAttention(planes * 4)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 应用SE注意力
        if self.use_se:
            out = self.se(out)

        if self.use_psa:
            out = self.psa(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5, in_channels=11, use_dilation=True, dilation_rates=None, use_se=False, se_reduction=16, use_psa=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.use_dilation = use_dilation
        self.use_se = use_se
        self.se_reduction = se_reduction
        self.use_psa = use_psa
        # 设置默认的空洞率
        if dilation_rates is None:
            if use_dilation:
                # 标准空洞卷积设置：layer3和layer4使用空洞卷积
                self.dilation_rates = [1, 1, 2, 4]  # [layer1, layer2, layer3, layer4]
            else:
                self.dilation_rates = [1, 1, 1, 1]  # 不使用空洞卷积
        else:
            self.dilation_rates = dilation_rates

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 使用空洞卷积创建层
        self.layer1 = self._make_layer(block, 64, layers[0], dilation=self.dilation_rates[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilation=self.dilation_rates[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilation=self.dilation_rates[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilation=self.dilation_rates[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        '''
        权重初始化方法
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化（He初始化）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm层权重初始化为1，偏置初始化为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 线性层使用Xavier初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))

        layers = []
        # 第一个block处理stride和dilation
        layers.append(block(self.inplanes, planes, stride, downsample, dilation, use_se=self.use_se, se_reduction=self.se_reduction, use_psa=self.use_psa))
        self.inplanes = planes * block.expansion

        # 后续block保持相同的dilation
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, use_se=self.use_se, se_reduction=self.se_reduction, use_psa=self.use_psa))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def extract_features(self, x):
        '''
        提取特征表示，不包含最后的分类层
        用于对比学习
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # 不经过最后的分类层，直接返回特征
        return x


class JointModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128, num_classes=5):
        super(JointModel, self).__init__()
        self.encoder = base_encoder
        # 投影头：用于对比学习
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim))
        
        # 分类头：用于交叉熵损失
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 提取特征 - 获取ResNet的特征表示，而不是分类输出
        h = self.encoder.extract_features(x)
        # 投影到对比学习空间
        z = self.projector(h)
        # 分类输出
        logits = self.classifier(h)
        return h, z, logits
    

def resnet18(in_channels=11, use_dilation=True, use_se=False, se_reduction=16, use_psa=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, use_dilation=use_dilation, use_se=use_se, se_reduction=se_reduction, use_psa=use_psa, **kwargs)

    return model


def resnet34(in_channels=11, use_dilation=True, use_se=False, se_reduction=16, use_psa=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, use_dilation=use_dilation, use_se=use_se, se_reduction=se_reduction, use_psa=use_psa, **kwargs)
    return model


def resnet50(in_channels=11, use_dilation=True, use_se=False, se_reduction=16, use_psa=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, use_dilation=use_dilation, use_se=use_se, se_reduction=se_reduction, use_psa=use_psa, **kwargs)
    return model


def resnet101(in_channels=11, use_dilation=True, use_se=False, se_reduction=16, use_psa=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], in_channels=in_channels, use_dilation=use_dilation, use_se=use_se, se_reduction=se_reduction, use_psa=use_psa, **kwargs)
    return model


def resnet152(in_channels=11, use_dilation=True, use_se=False, se_reduction=16, use_psa=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], in_channels=in_channels, use_dilation=use_dilation, use_se=use_se, se_reduction=se_reduction, use_psa=use_psa, **kwargs)
    return model
