import sys
import torch
import torch.nn as nn

from einops import rearrange
import os

sys.path.append(os.getcwd())

from .Groupmamba_decoder import GroupMambaLayer
import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:  # drop_prob废弃率=0，或者不是训练的时候，就保持原来不变
        return x
    keep_prob = 1 - drop_prob  # 保持率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (b, 1, 1, 1) 元组  ndim 表示几维，图像为4维
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 0-1之间的均匀分布[2,1,1,1]
    random_tensor.floor_()  # 下取整从而确定保存哪些样本 总共有batch个数
    output = x.div(keep_prob) * random_tensor  # 除以 keep_prob 是为了让训练和测试时的期望保持一致
    # 如果keep，则特征值除以 keep_prob；如果drop，则特征值为0
    return output  # 与x的shape保持不变


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


## Hybrid attention Enhancement  (MAE)
# class MAE(nn.Module):
#     """
#     MAE类：一个基于卷积和注意力机制的模块
#     """
#     def __init__(self, dim, num_heads=8, bias=False):
#         """
#         初始化函数
        
#         参数:
#             dim (int): 输入特征的维度
#             num_heads (int): 注意力机制中的头数
#             bias (bool): 是否在卷积层中添加偏置项
#         """
#         super(MAE, self).__init__()
#         self.num_heads = num_heads
#         self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
#         # 定义一个深度可分离卷积层，对q, k, v进行进一步处理
#         # groups参数设置为dim * 3，意味着每个通道进行独立的卷积操作
#         self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
#         # 定义一个卷积层，用于对注意力权重进行进一步处理（假设是某种注意力权重的转换）
#         # 注意这里输出的通道数是9，可能是为了与后续的dep_conv配合使用
#         self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
#         # 定义一个深度可分离卷积层，用于对注意力权重处理后的特征进行进一步处理
#         # groups参数设置为dim // num_heads，意味着每num_heads个通道进行一组独立的卷积操作
#         self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
#                                   groups=dim // self.num_heads, padding=1)
#         # 定义一个DropPath层，用于在训练过程中随机丢弃部分路径以防止过拟合
#         self.drop = DropPath(drop_prob=0.3)
#         # 定义一个GELU激活函数层
#         self.gelu = nn.GELU()
        
#         self.mamba = GroupMambaLayer(dim)
        
#     def forward(self, x):  
#         # 保存原始输入x，用于后续可能的残差连接  
#         x_idx = x  
#         # 获取输入x的形状信息  
#         b, c, h, w = x.shape  
#         # 增加一个维度，方便后续处理  
#         x = x.unsqueeze(2)  
      
#         # 通过卷积层生成q, k, v的输入  
#         qkv = self.qkv_dwconv(self.qkv(x))  
#         # 移除之前增加的维度  
#         qkv = qkv.squeeze(2)  
      
#         # 对qkv进行排列和重塑，准备进行注意力计算  
#         f_conv = qkv.permute(0, 2, 3, 1)  
#         f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)  
#         # 对f_all进行进一步的卷积处理  
#         f_all = self.fc(f_all.unsqueeze(2))  
#         # 移除之前增加的维度  
#         f_all = f_all.squeeze(2)  
      
#         # 局部卷积部分  
#         # 对f_all进行排列和重塑，准备进行局部卷积  
#         f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)  
#         # 增加一个维度，以匹配深度可分离卷积的输入  
#         f_conv = f_conv.unsqueeze(2)  
#         # 进行深度可分离卷积处理  
#         out_conv = self.dep_conv(f_conv)  
#         # 移除之前增加的维度  
#         out_conv = out_conv.squeeze(2)  
      
#         # 全局自注意力部分  
#         out = x_idx + self.mamba(x_idx)
        
#         # 将局部卷积和全局自注意力的结果相加，并应用DropPath和GELU激活函数  
#         output = x_idx + self.drop(self.gelu(out + out_conv))  
      
#         return output


class MAE(nn.Module):
    """
    MAE类：一个基于卷积和注意力机制的模块
    """
    def __init__(self, dim, num_heads=8, bias=False):
        """
        初始化函数
        
        参数:
            dim (int): 输入特征的维度
            num_heads (int): 注意力机制中的头数
            bias (bool): 是否在卷积层中添加偏置项
        """
        super(MAE, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        # 定义一个深度可分离卷积层，对q, k, v进行进一步处理
        # groups参数设置为dim * 3，意味着每个通道进行独立的卷积操作
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        # 定义一个卷积层，用于对注意力权重进行进一步处理（假设是某种注意力权重的转换）
        # 注意这里输出的通道数是9，可能是为了与后续的dep_conv配合使用
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)
        # 定义一个深度可分离卷积层，用于对注意力权重处理后的特征进行进一步处理
        # groups参数设置为dim // num_heads，意味着每num_heads个通道进行一组独立的卷积操作
        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)
        
        self.mamba = GroupMambaLayer(dim)
        
    def forward(self, x):  
        # 保存原始输入x，用于后续可能的残差连接  
        x_idx = x  
        # 获取输入x的形状信息  
        b, c, h, w = x.shape  
        # 增加一个维度，方便后续处理  
        x = x.unsqueeze(2)  
      
        # 通过卷积层生成q, k, v的输入  
        qkv = self.qkv_dwconv(self.qkv(x))  
        # 移除之前增加的维度  
        qkv = qkv.squeeze(2)  
      
        # 对qkv进行排列和重塑，准备进行注意力计算  
        f_conv = qkv.permute(0, 2, 3, 1)  
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)  
        # 对f_all进行进一步的卷积处理  
        f_all = self.fc(f_all.unsqueeze(2))  
        # 移除之前增加的维度  
        f_all = f_all.squeeze(2)  
      
        # 局部卷积部分  
        # 对f_all进行排列和重塑，准备进行局部卷积  
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)  
        # 增加一个维度，以匹配深度可分离卷积的输入  
        f_conv = f_conv.unsqueeze(2)  
        # 进行深度可分离卷积处理  
        out_conv = self.dep_conv(f_conv)  
        # 移除之前增加的维度  
        out_conv = out_conv.squeeze(2)  
      
        # 全局自注意力部分  
        out = x_idx + self.mamba(x_idx)
        
        # 将局部卷积和全局自注意力的结果相加 
        output = out + out_conv
      
        return output


if __name__ == "__main__":
    model = MAE(64, 8, False).cuda()
    inputs = torch.ones([2, 64, 128, 128]).cuda()  # [b,c,h,w]
    outputs = model(inputs)
    print(outputs.size())