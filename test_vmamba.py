import torch
import torch.nn as nn
from models.vmamba import SS2D

if __name__=='__main__':
    ss2d = SS2D(d_model=256).cuda()
    x = torch.randn(16,256,64,64)
    x = x.cuda()
    y = ss2d(x)
    print(y.shape)