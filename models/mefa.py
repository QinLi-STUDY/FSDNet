import torch
from torch import nn

# 通道注意力机制模块 (Channel Attention)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):  # 初始化函数，in_planes为输入特征图的通道数，ratio为降维比率
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化，将每个通道的特征压缩成1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化，将每个通道的特征压缩成1x1

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 1x1卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 1x1卷积层，恢复维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成注意力权重

    def forward(self, x):  # 前向传播函数
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化后，经过卷积和激活
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化后，经过卷积和激活

        out = avg_out + max_out  # 将平均池化和最大池化的结果相加
        return self.sigmoid(out)  # 使用Sigmoid函数生成注意力权重，并返回

# 空间注意力机制模块 (Spatial Attention)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # 初始化函数，kernel_size为卷积核大小
        super(SpatialAttention, self).__init__()  # 调用父类初始化函数

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 检查卷积核大小是否是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据卷积核大小决定填充的大小

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 卷积操作，输入2个通道，输出1个通道
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数，用于生成空间注意力权重

    def forward(self, x):  # 前向传播函数
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图在通道维度进行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图在通道维度进行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将平均池化和最大池化的结果拼接在一起
        x = self.conv1(x)  # 通过卷积操作生成空间注意力特征图
        return self.sigmoid(x)  # 使用Sigmoid函数生成注意力权重，并返回

# CBAM模块：结合通道注意力和空间注意力
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):  # 初始化函数，in_planes为输入特征图的通道数，ratio为降维比率，kernel_size为空间注意力的卷积核大小
        super(CBAM, self).__init__()  # 调用父类初始化函数
        self.ca = ChannelAttention(in_planes, ratio)  # 初始化通道注意力模块
        self.sa = SpatialAttention(kernel_size)  # 初始化空间注意力模块

    def forward(self, x):  # 前向传播函数
        out = x * self.ca(x)  # 通道维度的加权，使用通道注意力生成的权重对输入特征图进行加权
        result = out * self.sa(out)  # 空间维度的加权，使用空间注意力生成的权重对加权后的特征图进行加权
        return result  # 返回最终加权后的特征图

# 多尺度有效特征聚合模块 (MEFA)
class MEFA(nn.Module):  # 定义多尺度有效特征聚合模块
    def __init__(self, dim,channel):  # 初始化函数，dim表示输入特征图的通道数
        super(MEFA, self).__init__()  # 调用父类初始化函数
        self.h_c = dim // 4  # 计算隐藏通道数，减少计算量和参数量
        # 定义一个1x1卷积层，用于初始特征图的降维
        self.conv_init = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),  # 输入通道是2 * dim，输出是dim
        )
        # 定义3种不同大小的卷积核（3x3, 5x5, 7x7）来提取不同尺度的特征
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.h_c, self.h_c, kernel_size=3, padding=1, groups=self.h_c),  # 3x3卷积
            nn.BatchNorm2d(self.h_c),  # 批标准化
            nn.ReLU()  # 激活函数ReLU
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.h_c, self.h_c, kernel_size=5, padding=2, groups=self.h_c),  # 5x5卷积
            nn.BatchNorm2d(self.h_c),  # 批标准化
            nn.ReLU()  # 激活函数ReLU
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.h_c, self.h_c, kernel_size=7, padding=3, groups=self.h_c),  # 7x7卷积
            nn.BatchNorm2d(self.h_c),  # 批标准化
            nn.ReLU()  # 激活函数ReLU
        )
        # 定义最终的卷积层，聚合不同尺度的特征
        self.conv_fina1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),  # 1x1卷积，用于通道间融合
            nn.BatchNorm2d(dim),  # 批标准化
            nn.ReLU()  # 激活函数ReLU
        )
        # 定义最后的卷积层，通过3个卷积操作融合通道
        self.conv_fina2 = nn.Sequential(
            nn.Conv2d(dim,  dim, 3, 1, 1),  # 3x3卷积
            nn.BatchNorm2d(dim),  # 批标准化
            nn.ReLU(),  # 激活函数ReLU
        )
        self.sigmod=nn.Sigmoid()  # Sigmoid激活函数，用于生成最终输出的权重
        self.cbam = CBAM(dim, ratio=16, kernel_size=7)  # 初始化CBAM模块，输入通道数为dim，比例为16，卷积核大小为7
        self.conv = nn.Conv2d(channel, dim, 1)

    def forward(self, y1, y2):  # 前向传播函数，获取输入数据的两个部分
        y2 = self.conv(y1)
        f1 = torch.cat([y1, y2], dim=1)  # 将两个输入特征图在通道维度上拼接
        x = self.conv_init(f1)  # 通过conv_init进行降维
        x1, x2, x3, x4 = torch.split(x, self.h_c, dim=1)  # 将x划分为4个部分，每个部分的通道数为h_c
        x1 = self.conv1_1(x1)  # 对每个划分的部分应用卷积操作
        x2 = self.conv1_2(x2)  # 对每个划分的部分应用卷积操作
        x3 = self.conv1_3(x3)  # 对每个划分的部分应用卷积操作
        x = torch.cat([x1, x2, x3, x4], dim=1)  # 将卷积后的特征图在通道维度上拼接
        x = self.conv_fina1(x)  # 通过1x1卷积进一步聚合特征
        a = self.sigmod(x)  # 使用Sigmoid激活函数得到加权因子
        x1 =  self.cbam(y1)  # 对输入y1应用CBAM注意力机制
        x2 =  self.cbam(y2)  # 对输入y2应用CBAM注意力机制
        out = a*x1 + a*x2 + y1 + y2  # 将加权后的特征图与原始输入进行融合
        out = self.conv_fina2(out)  # 通过3x3卷积进一步融合特征
        return out  # 返回最终的输出特征图

if __name__ == '__main__':
    input1 = torch.randn(1, 512, 16, 16).cuda()  # 随机生成输入数据，形状为(1, 32, 64, 64)，并将其转移到GPU
    input2 = torch.randn(1, 256, 16, 16).cuda()  # 随机生成输入数据，形状为(1, 32, 64, 64)，并将其转移到GPU
    model = MEFA(512,256).cuda()  # 实例化MEFA模块，输入通道数为32，并将模型转移到GPU
    output = model(input1, input2)  # 将输入通过模块进行处理，注意这里传入的是两个相同的输入
    print(output.shape)  # 打印输出特征图的形状
