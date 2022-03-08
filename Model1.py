# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 21:31:23 2022

@author: Klaus
"""

import torch
from torch import nn

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

class Model(nn.Module): # subclass of Module class
    def __init__(self, hyperpars):
        super().__init__() # First, initiate the parent class (in Python 2, use super(Model,self))
        # Next, specify the layers, which will be available in this model.
        self.hyperpars=hyperpars
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.hyperpars.numberofchannels, 32, kernel_size=5, stride=1, padding='same'), # 3 input channels (RGB) (or 1 for b/w), 32 output channels (feature maps, created automatically); padding automatically chosen according to W_out=(W_in-kernel_size+2*padding)/stride+1 to keep dimensions of the image constant
            nn.ReLU(), # type of activation
            nn.MaxPool2d(kernel_size=2, stride=2)) # reduce image size to half in each dimension
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'), # have to input channel number from last output
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding='same'), # have to input channel number from last output
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout() # extra step: sets some elements to 0 randomly; avoids overfitting?
        self.fc1 = nn.Linear(int(self.hyperpars.image_size[0]/8) * int(self.hyperpars.image_size[1]/8) * 128, 1000) # Linear step; Args: Length of input array, length of output array
        self.fc2 = nn.Linear(1000, self.hyperpars.numberoflabels)# Second linear step
    
    # Override the existing forward method with custom layers
    def forward(self, x):
        x=x.float()
        out = self.layer1(x)
        out = self.layer2(out)
        out=self.layer3(out)
        out = torch.flatten(out, start_dim=1) # Convert to 1-D array; keeps all elements; keep dim 0 =batch dim
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def weight_reset(self):
        self.apply(weight_reset)
