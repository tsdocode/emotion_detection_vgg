import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd



class EmotionDataset(Dataset):
    def __init__(self, img, label):
        self.img = torch.from_numpy(img).permute(0,3,1,2)
        self.label = torch.from_numpy(label)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]

class EmotionData():
    def __init__(self):
        data_dict = self.__load_data__()
        self.train_data = data_dict['train']
        self.test_data = data_dict['test']
        self.valid_data = data_dict['valid']

    def __load_data__(self):
        #Reading file
        df = pd.read_csv('fer2013.csv')

        groups = [i for _, i in df.groupby('Usage')]
        training_data = groups[2]
        validation_data = groups[1]
        testing_data = groups[0]
        
        X_train = np.array(list(map(str.split, training_data.pixels)), np.float32) 
        X_val = np.array(list(map(str.split, validation_data.pixels)), np.float32) 
        X_test = np.array(list(map(str.split, testing_data.pixels)), np.float32) 
        X_train = X_train.reshape(X_train.shape[0], 48, 48, 1) 
        X_val = X_val.reshape(X_val.shape[0], 48, 48, 1)
        X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

        y_train = np.array(training_data.emotion)
        y_val = np.array(validation_data.emotion)
        y_test = np.array(testing_data.emotion)

        return {
            'train' : EmotionDataset(X_train, y_train),
            'valid' : EmotionDataset(X_val, y_val),
            'test' : EmotionDataset(X_test, y_test)
        }

