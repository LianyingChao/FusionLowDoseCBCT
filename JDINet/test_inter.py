import os
import torch
import numpy as np
import config
import imageio
from torch.autograd import Variable
from torch.nn import functional as F

from model_inter.JDINet_inter import UNet_3D_2D


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

args, unparsed = config.get_args()
device = torch.device('cuda' if args.cuda else 'cpu')

model = UNet_3D_2D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
model=model.to(device)

def test(args):
    model_path='./saved_model/JDINet_inter.pth'
    model_dict=model.state_dict()
    model.load_state_dict(torch.load(model_path)["state_dict"] , strict=True)
    model.eval()
    dataPath='../full_clean_proj/'
    pairData=np.load('./pairdata.npy')


    #test_folder = ["19"]                    
    with torch.no_grad():
        for walnutId in range(0,1):
            #walnut=test_folder[walnutId]
            #print(walnut)
            saveDir='../full_clean_proj//'
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            for idx in range(250):
                projId=idx*2

                rawFrameInput=np.zeros((4,640,640))
                rawFrameGt=np.zeros((1,640,640))

                # load input frames
                for k in range(4):
                    index=int(pairData[projId,k+1])
                    print(index)
                    img=imageio.imread(dataPath+str(index)+'.tif')
                    imageio.imsave(saveDir+str(index)+'.tif',img)
                    rawFrameInput[k,:,:]=img              
                rawFrameInput=torch.tensor(rawFrameInput)
                rawFrameGt=torch.tensor(rawFrameGt)

                rawFrameInput=rawFrameInput.unsqueeze(0)

                # padding
                h0=rawFrameInput.shape[-2]
                if h0%32!=0:
                    pad_h=32-(h0%32)
                    rawFrameInput=F.pad(rawFrameInput,(0,0,0,pad_h),mode='reflect')
                
                rawFrameGt=rawFrameGt.unsqueeze(1)
                rawFrameInput=rawFrameInput.unsqueeze(1)
                rawFrameInput=rawFrameInput.float()
                rawFrameGt=rawFrameGt.float()

                rawFrameInput=to_variable(rawFrameInput)
                rawFrameGt=to_variable(rawFrameGt)

                out = model(rawFrameInput)
                img=out.detach().cpu().numpy()
                for j in range(1):
                    index=int(pairData[projId,j+5])
                    print(index)
                    imageio.imsave(saveDir+str(index)+'.tif',img[j,0,:,:])

if __name__ == "__main__":
    test(args)
