import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_2D import SEGating

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d1(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D1(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


class UNet_2D_2D(nn.Module):
    def __init__(self, block , n_inputs=4, n_outputs=3, batchnorm=False , joinType="concat" , upmode="transpose"):
        super().__init__()

        nf = [512 , 256 , 128 , 64]        
        out_channels = 1*n_outputs
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module(".resnet_2D" , "model_denoi")
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm)  

        self.decoder = nn.Sequential(
            Conv_2d(nf[0], nf[1] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv2D(nf[1]*growth, nf[2], kernel_size=4, stride=2, padding=1 , upmode=upmode, batchnorm=batchnorm),
            upConv2D(nf[2]*growth, nf[3], kernel_size=4, stride=2, padding=1 , upmode=upmode, batchnorm=batchnorm),
            Conv_2d(nf[3]*growth, nf[3] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm),
            upConv2D(nf[3]*growth , 1, kernel_size=4, stride=2, padding=1 , upmode=upmode, batchnorm=batchnorm)
        )
        self.Tanh=nn.Tanh()

        self.feature_fuse = Conv_2d(nf[3]*n_inputs , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels , kernel_size=7 , stride=1, padding=0) 
        ) 

                

    def forward(self, images):

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images)
        dx_3 = self.lrelu(self.decoder[0](x_4))              # [1,256,64,64]
        
        dx_3 = joinTensors(dx_3 , x_3 , type=self.joinType)  # [1,512,64,64]
        # dx_3 = joinTensors(dx_3,dx_3_3d_reshape1,type='add')
        

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        dx_2 = joinTensors(dx_2 , x_2 , type=self.joinType)
        # dx_2 = joinTensors(dx_2,dx_2_3d_reshape1,type='add')

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        dx_1 = joinTensors(dx_1 , x_1 , type=self.joinType)
        # dx_1 = joinTensors(dx_1,dx_1_3d_reshape1,type='add')

        dx_0 = self.lrelu(self.decoder[3](dx_1))
        dx_0 = joinTensors(dx_0 , x_0 , type=self.joinType)   # [1,128,256,256]
        # dx_0 = joinTensors(dx_0,dx_0_3d_reshape1,type='add')

        dx_out = self.lrelu(self.decoder[4](dx_0))     # [1,64,512,512]

        return dx_out

