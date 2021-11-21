import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from utils import *
import torchvision.models as models








class chan_attention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(chan_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
#         print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
#         print(y.shape)
        return x * y.expand_as(x)