#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import sys
import os
import pprint
#from torchsummary import summary
from torchinfo import summary
import h5py
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
import torch.autograd as autograd
from torch import optim
import pandas as pd
import scipy.io


# In[2]:


class netG(nn.Module):
    def __init__(self, nc = 1, nz = 1, ngf = 64, gfs = 5, ngpu = 1):
        super(netG, self).__init__()
        self.ngpu = ngpu

        self.main = nn.Sequential(

                nn.ConvTranspose2d(     nz, ngf * 8, gfs, 2, gfs//2, bias=False), 
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 8),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 4),

                nn.ConvTranspose2d(ngf * 4, ngf * 2, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf * 2),

                nn.ConvTranspose2d(ngf * 2,     ngf, gfs, 2, gfs//2, bias=False),
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf),
               
                nn.ConvTranspose2d(    ngf,      nc, 4, 2, 2, bias=False),
                nn.ReLU(True),
                
                ### Start dilations ###
                nn.ConvTranspose2d(     nc,ngf, gfs, 1, 6, output_padding=0,bias=False,dilation=3), 
                nn.ReLU(True),
                nn.InstanceNorm2d(ngf),
               
                nn.ConvTranspose2d(    ngf,  nc, gfs, 1, 10, output_padding=0, bias=False,dilation=5),
                
                nn.Tanh()
                
            )

    def forward(self, input):
        input = input.view(input.size(0), -1, 5, 5) # (*, 100, 1, 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output


# In[3]:


G = netG().cuda()
G.load_state_dict(torch.load('netG_epoch_27.pth'))


# In[7]:


noise = torch.rand(1, 25)*2-1
z =noise.cuda()
fake = G(z)
plt.contourf(fake[0][0].detach().cpu().numpy())
plt.colorbar()


# In[ ]:




