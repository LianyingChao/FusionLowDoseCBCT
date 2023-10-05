import os
import torch
import numpy as np
import config
import imageio
from torch.autograd import Variable
from torch.nn import functional as F
from model_denoi.JDINet_Denoi import UNet_2D



def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

args, unparsed = config.get_args()
device = torch.device('cuda' if args.cuda else 'cpu')

model = UNet_2D(args.model.lower() , n_inputs=args.nbr_frame, n_outputs=args.n_outputs, joinType=args.joinType, upmode=args.upmode)
model=model.to(device)


def test(args):
    model_path='./saved_model/JDINet_denoi.pth'
    model_dict=model.state_dict()
    model.load_state_dict(torch.load(model_path)["state_dict"] , strict=True)
    model.eval()
    dataPath='../ld_proj/'
    pairData=np.load('./pairdata.npy')

    #test_folder=["19"]
    with torch.no_grad():
        for walnutId in range(0,1): #33-38
            #walnut=test_folder[walnutId]
            saveDir='../full_clean_proj/'
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            for idx in range(63):
                projId=idx*8    
                rawFrameInput=np.zeros((4,1,640,640))

                for k in range(4):
                    index=int(pairData[projId,k+1])
                    img=imageio.imread(dataPath+str(index)+'.tif')
                    rawFrameInput[k,0,:,:]=img

                rawFrameInput=torch.tensor(rawFrameInput)
                h0=rawFrameInput.shape[-2]
                if h0%32!=0:
                    pad_h=32-(h0%32)
                    rawFrameInput=F.pad(rawFrameInput,(0,0,0,pad_h),mode='reflect')
                rawFrameInput=rawFrameInput.float()
                rawFrameInput=to_variable(rawFrameInput)
                output = model(rawFrameInput) 
                output = output.detach().cpu().numpy()
                for j in range(4):
                    index=int(pairData[projId,j+1])
                    print(index)
                    imageio.imsave(saveDir+str(index)+'.tif',output[j,0,:,:])
   

if __name__ == "__main__":
    test(args)
