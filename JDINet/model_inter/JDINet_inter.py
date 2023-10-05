import math
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_inter.UNet_3D_3D import UNet_3D_3D
from model_inter.UNet_2D_2D import UNet_2D_2D


def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1



class UNet_3D_2D(nn.Module):
    def __init__(self, block , n_inputs=4, n_outputs=3, batchnorm=False , joinType="concat" , upmode="transpose"):
        super().__init__()

        self.UNet_3D=UNet_3D_3D(block,n_inputs,n_outputs,batchnorm=False , joinType="concat" , upmode="transpose")
        self.UNet_2D=UNet_2D_2D(block,n_inputs,n_outputs,batchnorm=False , joinType="concat" , upmode="transpose")

        nf = [512 , 256 , 128 , 64]        
        out_channels = 1*n_outputs


        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels , kernel_size=7 , stride=1, padding=0) 
        )       



    def forward(self, images):

        out_3d,dx_3_3d,dx_2_3d,dx_1_3d,dx_0_3d=self.UNet_3D(images)   # [1,256,512,512]

        # print(out_3d.shape,dx_3_3d.shape,dx_2_3d.shape,dx_1_3d.shape,dx_0_3d.shape)
        # exit()
        
        out_2d=self.UNet_2D(out_3d,dx_3_3d,dx_2_3d,dx_1_3d,dx_0_3d)   # [1,64,512,512]
        
        out = self.outconv(out_2d)                            # [1,15,512,512]
        out = torch.split(out, dim=1, split_size_or_sections=1) 
        out = torch.cat(out)
        h0=972
        out=out[:,:,0:h0,:]

        return out

