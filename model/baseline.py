from model.pvtv2 import *
import torch.nn.functional as F

from torch import nn
import torch
from .Groupmamba_decoder import GroupMambaLayer

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
        self.seg_out = nn.Conv2d(out_channels, 1, 1)
        
    def forward(self, low_feat, high_feat):
        x = self.up(high_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        pred=self.seg_out(x)
        return x,pred
    


class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels * 2)

        self.conv3 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

        
        

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Encoder
        backbone = pvt_v2_b2()
        model_dict = backbone.state_dict()
        path = 'Net_Uncertainty/pretrain/pvt_v2_b2.pth'
        save_model = torch.load(path, map_location='cpu')
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        backbone.load_state_dict(model_dict)  # 64, 128 , 320 ,512
        self.encoder = backbone
        print('Pretrained encoder loaded.')

        # neck模块
        self.conv4 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(320, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        self.conv_block = Conv_Block(128)
        
        self.groupmamba_att = GroupMambaLayer(128)
        
        self.seg_out = nn.Conv2d(128, 1, 1)
        
        
        # Decoder模块

        self.decoder3 = DecoderBlock(512, 320, 320)
        self.decoder2 = DecoderBlock(320, 128, 128)
        self.decoder1 = DecoderBlock(128, 64, 64)
        
        # sal head
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x, shape):
        return F.interpolate(x, size=shape, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        # backbone2
        x4, x3, x2, x1 = self.encoder(x)

        # neck
        E4, E3, E2 = self.conv4(x4), self.conv3(x3), self.conv2(x2)

        if E4.size()[2:] != E3.size()[2:]:
            E4 = F.interpolate(E4, size=E3.size()[2:], mode='bilinear')
        if E2.size()[2:] != E3.size()[2:]:
            E2 = F.interpolate(E2, size=E3.size()[2:], mode='bilinear')

        E5 = self.conv_block(E4, E3, E2)

        # #  yzq  add
        E5 = E5 + self.groupmamba_att(E5)
        sal4 = self.seg_out(E5)
        sal4 = self.upsample(sal4, size)
        sal_sig4 = self.sigmoid(sal4)
        
        # Decoder   
        fusion, predict3 = self.decoder3(x3, x4)
        sal3 = self.upsample(predict3, size)
        sal_sig3 = self.sigmoid(sal3)

        fusion, predict2 = self.decoder2(x2, fusion)
        sal2 = self.upsample(predict2, size)
        sal_sig2 = self.sigmoid(sal2)

        fusion, predict1 = self.decoder1(x1, fusion)
        sal1 = self.upsample(predict1, size)
        sal_sig1 = self.sigmoid(sal1)
   
        
        
        return sal4, sal_sig4, sal3, sal_sig3, sal2, sal_sig2, sal1, sal_sig1
