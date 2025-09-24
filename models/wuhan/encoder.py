import sys

sys.path.extend(['.', '..'])

import torch
import torch.nn as nn
from models.wuhan.psa import SequentialPolarizedSelfAttention


class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out, use_dropout=False, dropout_rate=None, plus_psa=False, psa_channels=None):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True), nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True), nn.BatchNorm2d(ch_out), nn.ReLU(inplace=True))
        self.dropout = use_dropout
        self.dropout_rate = nn.Dropout2d(dropout_rate, inplace=False)
        self.psa = plus_psa
        self.psa_attention = SequentialPolarizedSelfAttention(psa_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.psa:
            x = self.psa_attention(x)
        if self.dropout:
            x = self.dropout_rate(x)
        return x


class Encoder(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, use_dropout=False):
        super(Encoder, self).__init__()
        self.use_dropout = use_dropout
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=32, use_dropout=self.use_dropout, dropout_rate=0.1, plus_psa=True, psa_channels=32)
        self.Conv2 = conv_block(ch_in=32, ch_out=64, use_dropout=self.use_dropout, dropout_rate=0.1, plus_psa=True, psa_channels=64)
        self.Conv3 = conv_block(ch_in=64, ch_out=128, use_dropout=self.use_dropout, dropout_rate=0.1, plus_psa=True, psa_channels=128)
        self.Conv4 = conv_block(ch_in=128, ch_out=256, use_dropout=self.use_dropout, dropout_rate=0.1, plus_psa=True, psa_channels=256)
        self.Conv5 = conv_block(ch_in=256, ch_out=256, use_dropout=self.use_dropout, dropout_rate=0.1, plus_psa=True, psa_channels=256)

        self.fc = nn.Sequential(nn.Linear(14 * 14 * 256, 256), nn.Linear(256, output_ch))

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        # print(x4.shape)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x5 = torch.flatten(x5, 1)

        out = self.fc(x5)

        return out


if __name__ == "__main__":
    net = Encoder(3, 1)
    a = torch.randn(3, 3, 224, 224)
    x = net(a)
    print(x.shape)
