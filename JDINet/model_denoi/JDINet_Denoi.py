import math
import numpy as np
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_denoi.UNet_2D import UNet_2D_2D


def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1



class UNet_2D(nn.Module):
    def __init__(self, block , n_inputs=4, n_outputs=1, batchnorm=False , joinType="concat" , upmode="transpose"):
        super().__init__()
        self.UNet_2D=UNet_2D_2D(block,1,1,batchnorm=False , joinType="concat" , upmode="transpose")

    def forward(self, images):
        out_2d=self.UNet_2D(images) 

        return torch.clamp(out_2d, 0.0, 1.0)

