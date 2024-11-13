from model.pvtv2 import *
import torch.nn.functional as F
#from model.decoder import Decoder
#from model.Groupmamba_decoder import Decoder
# from model.Lightmamba_decoder import Decoder
from model.VMmamba_decoder import Decoder
from model.MambaAttention import EfficientMambaAttention
from torch import nn
import torch


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, high_feat, low_feat):
        x = self.up(high_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x








class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Encoder
        backbone = pvt_v2_b2()
        model_dict = backbone.state_dict()
        path = 'Net_MSA_Decoder/pretrain/pvt_v2_b2.pth'
        save_model = torch.load(path, map_location='cpu')
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}

        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)  # 64, 128 , 320 ,512
        self.encoder = backbone
        print('Pretrained encoder loaded.')

        # neck模块  和VM搭配效果不好
        # self.mamba_att1=EfficientMambaAttention(64)
        # self.mamba_att2=EfficientMambaAttention(128)
        # self.mamba_att3=EfficientMambaAttention(320)
        # self.mamba_att4=EfficientMambaAttention(512)

        # Decoder模块

        self.decoder = Decoder(128)

        # sal head
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x, shape):
        return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # backbone2
        x4, x3, x2, x1 = self.encoder(x)

        # neck
        # x1=self.mamba_att1(x1)
        # x2=self.mamba_att2(x2)
        # x3=self.mamba_att3(x3)
        # x4=self.mamba_att4(x4)

        # Decoder
        shape = x.size()[2:]
        P5, P4, P3, P2, P1 = self.decoder(x4, x3, x2, x1, shape)
        return P5, self.sigmoid(P5), P4, self.sigmoid(P4), P3, self.sigmoid(P3), P2, self.sigmoid(P2), P1, self.sigmoid(P1)

