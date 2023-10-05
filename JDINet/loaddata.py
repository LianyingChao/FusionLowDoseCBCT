import imageio
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import os
from torch.utils.data import Dataset, DataLoader




dataPath='/mnt/data2/fusion_wal/data/'
batch_size=1
pairData=np.load('./pairdata.npy')

train_folder = ["1","2","3","4","5","6","7","8","9","10","11", "12","13","14","15","16","17"]


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def readData():
    input=np.zeros((1,640,640))
    gt=np.zeros((1,640,640))
    patientId=np.random.randint(0,17)
    patient_folder=train_folder[patientId]
    projId=np.random.randint(500)

    # load input frames
    for k in range(1):
        index=int(pairData[projId,k+1])
        # print(index)
        input[k,:,:] = imageio.imread(dataPath+patient_folder+'/ld_proj/'+str(index)+'.tif')
        gt[k,:,:]=imageio.imread(dataPath+patient_folder+'/nd_proj/'+str(index)+'.tif')

    input=torch.tensor(input)
    gt=torch.tensor(gt)
    # input=input.unsqueeze(0)
    # gt=gt.unsqueeze(0)
    input=input.unsqueeze(1)
    gt=gt.unsqueeze(1)
    input=input.float()
    gt=gt.float()
    input=to_variable(input)
    gt=to_variable(gt)

    return input, gt


def readTestData():
    input=np.zeros((1,640,640))
    gt=np.zeros((1,640,640))
    patientId=np.random.randint(0,17)
    patient_folder=train_folder[patientId]
    projId=np.random.randint(500)

    # load input frames
    for k in range(1):
        index=int(pairData[projId,k+1])
        # print(index)
        input[k,:,:] = imageio.imread(dataPath+patient_folder+'/ld_proj/'+str(index)+'.tif')
        gt[k,:,:]=imageio.imread(dataPath+patient_folder+'/nd_proj/'+str(index)+'.tif')

    input=torch.tensor(input)
    gt=torch.tensor(gt)
    # input=input.unsqueeze(0)
    # gt=gt.unsqueeze(0)
    input=input.unsqueeze(1)
    gt=gt.unsqueeze(1)
    input=input.float()
    gt=gt.float()
    input=to_variable(input)
    gt=to_variable(gt)

    return input, gt
