import torch  
import torch.nn as nn  
  
# 导入自定义的Mamba模块  
from mamba_ssm import Mamba  
  
class EfficientMambaAttention(nn.Module):  
    """  
    使用Mamba的注意力模块  
    """  
    def __init__(self, channel, kernel_size=7):  
        """  
        初始化函数  
  
        参数:  
            channel (int): 输入的通道数  
            kernel_size (int, optional): 核大小，默认为7  
        """  
        super(EfficientMambaAttention, self).__init__()  
  
        # 定义sigmoid激活函数  
        self.sigmoid_x = nn.Sigmoid()  
  
        # 沿着宽度方向进行自适应平均池化  
        self.avg_x = nn.AdaptiveAvgPool2d((None, 1))  
  
        # 沿着高度方向进行自适应平均池化  
        self.avg_y = nn.AdaptiveAvgPool2d((1, None))  
  
        # 定义另一个sigmoid激活函数  
        self.sigmoid_y = nn.Sigmoid()  
  
        # 初始化Mamba模块，用于处理x方向  
        self.mamba_x = Mamba(  
            d_model=channel,  # 模型维度  
            d_state=16,  # SSM状态扩展因子  
            d_conv=4,  # 局部卷积宽度  
            expand=2,  # 块扩展因子  
        )  
  
        # 初始化Mamba模块，用于处理y方向  
        self.mamba_y = Mamba(  
            d_model=channel,  # 模型维度  
            d_state=16,  # SSM状态扩展因子  
            d_conv=4,  # 局部卷积宽度  
            expand=2,  # 块扩展因子  
        )  
  
    def forward(self, x):  
        """  
        前向传播函数  
  
        参数:  
            x (torch.Tensor): 输入张量  
  
        返回:  
            torch.Tensor: 输出张量  
        """  
  
        # 获取输入张量的尺寸  
        b, c, h, w = x.size()  
  
        # 沿着宽度方向进行平均池化，并转置  
        x_x = self.avg_x(x).squeeze(3).transpose(-1, -2)  
  
        # 通过Mamba模块处理x_x  
        x_ma = self.mamba_x(x_x).transpose(-1, -2)  
  
        # 应用sigmoid激活函数并重塑为原始尺寸  
        x_x = self.sigmoid_x(x_ma).view(b, c, h, 1)  
  
        # 沿着高度方向进行平均池化，并转置  
        x_y = self.avg_y(x).squeeze(2).transpose(-1, -2)  
  
        # 通过Mamba模块处理x_y，应用sigmoid激活函数，并重塑为原始尺寸  
        x_y = self.sigmoid_y(self.mamba_y(x_y).transpose(-1, -2)).view(b, c, 1, w)  
  
        # 返回x与x_x和x_y的逐元素乘积  
        return x * x_x.expand_as(x) * x_y.expand_as(x)