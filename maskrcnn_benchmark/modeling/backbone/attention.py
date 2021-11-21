import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# from utils import *
import torchvision.models as models
from math import log,sqrt
import numpy as np



class Inception_Net(nn.Module):
    def __init__(self,n=768):
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
        for l in [self.conv11, self.conv12, self.conv13, self.conv21, self.conv22, self.conv23, self.conv24,
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

#########Pixel Attention###########
class Pixel_atten(nn.Module):
    def __init__(self,n):
        super(Pixel_atten, self).__init__()
        self.inception = Inception_Net(n)
        # self.conv = nn.Sequential(nn.Conv2d(1024,1,(1,1)),nn.ReLU())
        self.conv = nn.Conv2d(1024,2,(1,1))
        self.upsample=nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)   
        self.criterion = nn.NLLLoss2d() 
        self.criterion_2 = nn.MSELoss()
        # self.criterion = nn.CrossEntropyLoss()    
        # self.softmax = nn.Softmax2d()#F.log_softmax
        # self.softmax = F.log_softmax()
        for l in [self.conv]:
            torch.nn.init.normal_(l.weight, std=0.01)
            torch.nn.init.constant_(l.bias, 0)
           
    def forward(self, train_mode, x, mask):
        x = self.inception (x)
        x = self.conv(x)
        # x = self.softmax(x)
        # print(x.shape)
        # x1 = self.upsample(x)
        # print(x.sh F
        # sal_map= self.upsample(x)
        
        
        # sal_map1= self.upsample(sal_map)
        # print(torch.sum(mask))
        # sal_map1 = F.softmax(sal_map1, dim=1)
        # print(sal_map.sha, dim=1pe)
        # print(mask.shape)
        if train_mode:
            sal_map = self.upsample(x)
            sal_map = F.log_softmax(sal_map,dim=1)
            mask = torch.reshape(mask,(mask.shape[0],mask.shape[2],mask.shape[3]))
            # print(sal_map.shape,mask.shape)
            loss=self.criterion(sal_map,mask)
            return x, loss#F.softmax(x,dim=1),loss
        else:
            sal_map = self.upsample(x)
            # sal_map_1 = F.log_softmax(sal_map,dim=1)
            sal_map_2 = F.softmax(sal_map,dim=1)
            return x,sal_map_2 #F.softmax(x,dim=1), None

        


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



