import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from utils import *
import torchvision.models as models
# from math import log,sqrt
# import numpy as np



class Inception_Net(nn.Module):
    def __init__(self,n=1024):
        super(Inception_Net, self).__init__()
        self.conv11=nn.Conv2d(n,192,(1,1))
        self.conv12=nn.Conv2d(192,224,(1,3),padding=(0,1))
        self.conv13=nn.Conv2d(224,256,(3,1),padding=(1,0))
        ###############################################
        self.conv21=nn.Conv2d(n,192,(1,1))
        self.conv22=nn.Conv2d(192,192,(5,1),padding=(2,0))
        self.conv23=nn.Conv2d(192,224,(1,5),padding=(0,2))
        self.conv24=nn.Conv2d(224,224,(7,1),padding=(3,0))             
        self.conv25=nn.Conv2d(224,256,(1,7),padding=(0,3))      
        ############################################################
        self.avg31=nn.AvgPool2d((3,3), stride=1, padding=(1,1))
        self.conv32=nn.Conv2d(n,128,(1,1))
        ##########################################################
        self.conv41=nn.Conv2d(n,384,(1,1))
        for l in [self.conv11, self.conv12, self.conv12, self.conv21, self.conv22, self.conv23, self.conv24,
                 self.conv25, self.conv32, self.conv41]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
        
          
    def forward(self, x):
        #######1st Branch#######
        x1 = F.relu(self.conv11(x))
#         print(x1.shape)
        x1 = F.relu(self.conv12(x1))
#         print(x1.shape)
        x1 = F.relu(self.conv13(x1))
# #         print(x1.shape)
        #######2nd Branch######
        x2 = F.relu(self.conv21(x))
#         print(x2.shape)
        x2 = F.relu(self.conv22(x2))
#         print(x2.shape)
        x2 = F.relu(self.conv23(x2))
#         print(x2.shape)
        x2 = F.relu(self.conv24(x2))
# #         print(x2.shape)
        x2 = F.relu(self.conv25(x2))
#         print(x2.shape)
        #######3rd Branch##########
        x3 = self.avg31(x)
        x3 = F.relu(self.conv32(x3))
#         print(x3.shape)
        #########4th Branch###############
        x4 = F.relu(self.conv41(x))
#         print(x4.shape)
       ##################################################
        result = torch.cat((x1, x2, x3, x4), 1)
#         print(result.shape)
        return result

#########Backbone###########
class Res_Net_101(nn.Module):
    def __init__(self,cfg):
        super(Res_Net_101, self).__init__()
        self.out_channels = 1024
        self.incep = Inception_Net(512)
        self.pretrained_model = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-5])  
        self.c3 = models.resnet101(pretrained=True).layer2
        self.c4 = models.resnet101(pretrained=True).layer3 
        self.m4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        # print(x.shape)
        x2 = self.pretrained_model(x)
        x3 = self.c3(x2)
        x4 = self.c4(x3)
        x3_1 = self.incep(x3)
        n4 = self.m4(x4)
        # print(x3_1.shape,n4.shape)
        feat=torch.add(x3_1, n4) 
        # print("feat shape",feat.shape)
        return feat

        


# def pix_atten(train_mode,features,mask,n):
#     net=Pixel_atten(n)
#     net=net.cuda()
#     sal_map=net(features)
#     upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
#     sal_map1=upsample(sal_map)
#     # print(sal_map1.shape)
#     criterion = nn.NLLLoss2d()
#     if train_mode:
#         loss=criterion(sal_map1,mask)
#         return sal_map,loss
#     else:
#         return sal_map,None



