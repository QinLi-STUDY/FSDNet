import torch
import torch.nn as nn
import numbers
from einops import rearrange
def to_3d(x):  # 定义一个函数，将输入的4D张量转换为3D张量
    return rearrange(x, 'b c h w -> b (h w) c')  # 使用einops的rearrange函数，将4D张量转换为3D，改变h,w维度为一个长向量
def to_4d(x, h, w):  # 定义一个函数，将3D张量转换为4D张量，恢复h, w维度
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # 将3D张量转回4D张量，恢复原本的高度和宽度h, w
class BiasFree_LayerNorm(nn.Module):  # 定义一个无偏置的层归一化类，继承自nn.Module
    def __init__(self, normalized_shape):  # 初始化函数，normalized_shape是输入的维度
        super(BiasFree_LayerNorm, self).__init__()  # 调用父类的构造函数
        if isinstance(normalized_shape, numbers.Integral):  # 如果normalized_shape是整数，转换为元组形式
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # 将normalized_shape转为torch.Size格式

        assert len(normalized_shape) == 1  # 确保normalized_shape的长度为1（即只对最后一维做归一化）

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化归一化的权重参数，初始为1
        self.normalized_shape = normalized_shape  # 保存归一化维度

    def forward(self, x):  # 前向传播函数
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算输入x的方差，沿着最后一维计算
        return x / torch.sqrt(sigma + 1e-5) * self.weight  # 将x归一化后乘以权重



class WithBias_LayerNorm(nn.Module):  # 定义一个带偏置的层归一化类，继承自nn.Module
    def __init__(self, normalized_shape):  # 初始化函数，normalized_shape是输入的维度
        super(WithBias_LayerNorm, self).__init__()  # 调用父类的构造函数
        if isinstance(normalized_shape, numbers.Integral):  # 如果normalized_shape是整数，转换为元组形式
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)  # 转换为torch.Size格式

        assert len(normalized_shape) == 1  # 确保normalized_shape的长度为1（即只对最后一维做归一化）

        self.weight = nn.Parameter(torch.ones(normalized_shape))  # 初始化归一化的权重参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # 初始化归一化的偏置参数
        self.normalized_shape = normalized_shape  # 保存归一化维度

    def forward(self, x):  # 前向传播函数
        mu = x.mean(-1, keepdim=True)  # 计算输入x的均值，沿着最后一维计算
        sigma = x.var(-1, keepdim=True, unbiased=False)  # 计算输入x的方差
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias  # 进行标准化，并加上偏置



class LayerNorm(nn.Module):  # 定义一个LayerNorm类，支持两种类型的归一化
    def __init__(self, dim, LayerNorm_type= 'WithBias'):  # 初始化函数，dim是维度，LayerNorm_type指定归一化类型
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':  # 如果指定使用无偏置归一化
            self.body = BiasFree_LayerNorm(dim)  # 使用无偏置的归一化
        else:
            self.body = WithBias_LayerNorm(dim)  # 默认使用带偏置的归一化

    def forward(self, x):  # 前向传播函数
        h, w = x.shape[-2:]  # 获取输入的h,w尺寸（最后两个维度）
        return to_4d(self.body(to_3d(x)), h, w)  # 将输入x转换为3D，经过归一化后再转回4D

class DWConv(nn.Module):  # 定义一个深度可分离卷积类
    def __init__(self, dim=768):  # 初始化函数，dim是输入通道数，默认值为768
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)  # 定义一个深度可分离卷积层

    def forward(self, x, H, W):  # 前向传播函数，x是输入，H和W是目标的高度和宽度
        B, N, C = x.shape  # 获取输入的batch大小B，特征数N，通道数C
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()  # 将输入的维度重排为适合卷积操作的4D张量
        x = self.dwconv(x)  # 进行深度可分离卷积
        x = x.flatten(2).transpose(1, 2)  # 将卷积结果展平，恢复回原来的维度
        return x

class GatedConvolutionalLinearModule(nn.Module):  # 定义一个门控卷积线性模块类
    def __init__(self, in_channels, drop=0.1):  # 初始化函数，in_channels是输入通道数，drop是Dropout的比例
        super(GatedConvolutionalLinearModule, self).__init__()

        hidden_channels = int(2 * in_channels)  # 定义隐藏通道数为2倍的输入通道数
        self.fc1 = nn.Linear(in_channels, hidden_channels * 2)  # 定义一个全连接层fc1，将输入映射到隐藏空间
        self.dwconv = DWConv(hidden_channels)  # 定义一个深度可分离卷积层，输入通道数为hidden_channels
        self.act = nn.GELU()  # 定义一个GELU激活函数
        self.fc2 = nn.Linear(hidden_channels, in_channels)  # 定义一个全连接层fc2，将隐藏层映射回输入通道数
        self.drop = nn.Dropout(drop)  # 定义一个Dropout层，用于防止过拟合

    def forward(self, x):  # 前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  # 将输入张量重排为3D张量，形状为(B, H*W, C)
        x1, x2 = self.fc1(x).chunk(2, dim=-1)  # 将输入传入fc1层，并分成两个部分x1和x2
        x = self.act(self.dwconv(x1, h, w)) * self.dwconv(x2, h, w)  # 经过深度可分离卷积并计算门控输出
        x = self.drop(x)  # 使用Dropout
        x = self.fc2(x)  # 经过第二个全连接层
        x = self.drop(x)  # 使用Dropout
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # 将输出张量重排回4D张量
        return x

class ConvAttention(nn.Module):  # 定义一个卷积注意力机制类
    def __init__(self, dim, num_heads=4, bias=False):  # 初始化函数，dim是输入通道数，num_heads是注意力头数，bias指定是否使用偏置
        super(ConvAttention, self).__init__()
        self.num_heads = num_heads  # 设置注意力头数
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 初始化参数变量，用于控制注意力的大小

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 定义1x1卷积层，用于计算查询、键、值（QKV）
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)  # 深度可分离卷积，用于QKV的进一步处理
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 用于输出的卷积层

    def forward(self, x):  # 前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小

        qkv = self.qkv_dwconv(self.qkv(x))  # 计算QKV的深度可分离卷积
        q, k, v = qkv.chunk(3, dim=1)  # 将QKV分割为查询、键、值

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重排q为适应多头注意力的形状
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重排k
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # 重排v

        q = torch.nn.functional.normalize(q, dim=-1)  # 归一化q
        k = torch.nn.functional.normalize(k, dim=-1)  # 归一化k

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # 计算注意力矩阵
        attn = attn.softmax(dim=-1)  # 对注意力矩阵进行softmax操作

        out = (attn @ v)  # 根据注意力权重对v进行加权求和

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)  # 将输出重新排列成适合的形状
        out = self.project_out(out)  # 输出通过1x1卷积进行投影
        return out

class GCAM(nn.Module):  # 定义GCAM类，作为特征增强模块
    def __init__(self, in_channels):  # 初始化函数，in_channels是输入通道数
        super(GCAM, self).__init__()
        self.norm1 = LayerNorm(in_channels, LayerNorm_type='WithBias')  # 第一层归一化
        self.norm2 = LayerNorm(in_channels, LayerNorm_type='WithBias')  # 第二层归一化
        self.GCLM = GatedConvolutionalLinearModule(in_channels)  # 定义门控卷积线性模块
        self.convattention = ConvAttention(in_channels)  # 定义卷积注意力模块

    def forward(self, x):  # 前向传播函数
        x = x + self.convattention(self.norm1(x))  # 先经过卷积注意力模块，增强特征
        x = x + self.GCLM(self.norm1(x))  # 然后经过门控卷积模块，进一步增强特征
        return x  # 返回增强后的特征图


# 输入 B C H W,  输出 B C H W
if __name__ == '__main__':
    # 生成随机输入张量
    input = torch.randn(1, 32, 64, 64)
    # 实例化模型对象
    model = GCAM(in_channels=32)
    # 执行 MGDB 前向传播
    output = model(input)
    print('input_size:', input.size())
    print('output_size:', output.size())