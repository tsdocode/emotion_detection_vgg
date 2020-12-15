from emotion.model.vgg19 import vgg19
import torch
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


LABELS = {
    0 : 'Angry',
    1 : 'Disgust',
    2 : 'Fear', 
    3 : 'Happy', 
    4 : 'Sad', 
    5 : 'Surprise',
    6 : 'Neutral'
}


import os

model_dir = os.path.join(os.getcwd(),'emotion', 'data', 'saved', 'emotion.pth')
net = vgg19(1,7)
net.load_state_dict(torch.load(model_dir))
net = net.to(device)
net.eval()

# from emotion.data.dataLoader import *


# print('Loading Data', end = '\r')
# dataset = EmotionData()
# loader = EmotionDataloader(dataset.data_dict, 32 , 0)

# dataloaders = loader.loader_dict
# datasizes = loader.datasize_dict

# import matplotlib.pyplot as plt


def predict(img):
    #print(img.shape)
    img = cv2.resize(img, (48,48))
    img = torch.Tensor(img)
    img = img.view(-1,48,48)
    img = img.view(-1,1,48,48)
    img  = img.to(device)
    print(img.shape)
    o = net(img)
    pre = torch.max(o, 1)[1]
    return LABELS[pre.item()]


import cv2

# Load the cascade
mtcnn = MTCNN(thresholds = [0.7, 0.5, 0.5], device = 'cuda', margin=20)
# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

def predict(img):
    #print(img.shape)
    img = cv2.resize(img, (48,48))
    img = torch.Tensor(img)
    img = img.view(-1,48,48)
    img = img.view(-1,1,48,48)
    img  = img.to(device)
    print(img.shape)
#     print(img.shape)
    o = net(img)
    print(o)
    pre = torch.max(o, 1)[1]
    print(f'predict {pre}')
    return LABELS[pre.item()]


def text(t, img, p):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = p
    fontScale              = 1
    fontColor              = (0,0,255)
    lineType               = 2
    org                    = (0,100)

    cv2.putText(img,t, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

while True:
    # Read the frame
    _, img = cap.read()
    # Detect the faces
   
    faces = mtcnn.detect(img)
    print(faces)
    
    
    # Draw the rectangle around each face
    if (type(faces[0]) != type(None)):
        for (x, y, x1, y1) in faces[0]:
            x1 = int(x1)
            y1 = int(y1)
            x = int(x)
            y = int(y)
            
            print(x , y , x1 , y1 , sep = ' ')
            
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
            face = img[x:x1, y:y1]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            t = predict(gray)
            text(t,img, (x,y))
            

    cv2.imshow('img', img) 
                                  

    # Display
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()







