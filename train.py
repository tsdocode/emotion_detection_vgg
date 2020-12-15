from emotion.data.dataLoader import *
from emotion.model.vgg19 import vgg19
from emotion.trainer.helper import train_model



import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn




# import the necessary packages
import argparse
import os
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--device", required=True,
	help="Using CUDA or CPU")
args = vars(ap.parse_args())






#Load data
print('Loading Data', end = '\r')
dataset = EmotionData()
loader = EmotionDataloader(dataset.data_dict, 32 , 0)

dataloaders = loader.loader_dict
datasizes = loader.datasize_dict

print('Complete loading data')
#Load model
vgg = vgg19(1,7)

#Preparing training process
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(vgg.parameters() , lr=0.0001)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if (args['device'] == 'cuda'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = 'cpu'

vgg.to(device)

#Traning
vgg = train_model(vgg,dataloaders, datasizes , criterion, 
                    optimizer, exp_lr_scheduler, device,num_epochs=25)


save_dir = os.path.join(os.getcwd(),'emotion', 'data', 'saved', 'emotion.pth')
torch.save(vgg, save_dir)


