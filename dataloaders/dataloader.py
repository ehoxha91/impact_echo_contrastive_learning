from cProfile import label
from random import shuffle
import torch
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import numpy as np


class ImpactEchoDatasetClassifier(Dataset):
    
    def __init__(self, 
            X_path,
            y_path,
            sr = [200000, 104200],
            array_size = 1519,
            shuffle=False
    ):
        self.sr = sr
        self.X = np.array([])
        self.array_size = array_size
        self.shuffle = shuffle
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))

        # labels for DS1
        self.labels1 =  np.load(y_path[0])
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        self.labels1[self.labels1 < 1] = 0
        self.labels1[self.labels1 > 0] = 1
        self.labels1 = [int(label) for label in self.labels1]
        self.dataset1_size = len(self.labels1)
        if len(y_path) == 2:
            self.labels2 =  np.load(y_path[1])
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels2)} data points")
            self.y = np.append(self.labels1, self.labels2)

        elif len(y_path) > 2:
            self.labels3 =  np.load(y_path[1])
            self.labels3[self.labels3 < 1] = 0
            self.labels3[self.labels3 > 0] = 1
            self.labels3 = [int(label) for label in self.labels3]
            self.y = np.append(self.labels1, self.labels3)
            # import matplotlib.pyplot as plt
            # plt.imshow(np.reshape(self.labels3[0:1178], (31,38)))
            # plt.savefig('test.png')
            self.dataset1_size = len(self.y)
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels3)} data points")
            
            # labels for SDNET
            # self.labels2 =  np.load(y_path[2])
            # self.labels2[self.labels2 ==1] = 0
            # self.labels2[self.labels2 >= 2] = 1
            # self.y = np.append(self.y, self.labels2)

            self.labels2 =  np.load(y_path[2])
            self.labels2[self.labels2 < 1] = 0
            self.labels2[self.labels2 > 0] = 1
            self.labels2 = [int(label) for label in self.labels2]
            print(f"Loaded dataset {y_path[2]}, with {len(self.labels2)} data points")
            self.y = np.append(self.y, self.labels2)
        else:
            self.y = self.labels1

        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]

    def __getitem__(self, index):
        return torch.tensor(self.__normalize_data__(self.X[index][:self.array_size]), dtype=torch.double), torch.tensor(self.y[index], dtype=torch.int8)
        
    def __normalize_data__(self, signal):
        return (signal - np.min(signal))/(np.max(signal)-np.min(signal))
    
    def __len__(self):
        return len(self.X)

    def get_labels(self):
        return self.labels1, self.labels2

    def __gettest__(self):
        return torch.tensor(self.X), self.y


class ImpactEchoDatasetCL(Dataset):
    
    def __init__(self, 
                X_path,
                y_path,
                sr = [200000, 104200],
                array_size = 1519,
                shuffle=False,
                augment_data = True):
        self.sr = sr
        self.X = np.array([])
        self.augment_data = augment_data
        self.array_size = array_size
        self.shuffle = shuffle
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))

        # labels for DS1
        self.labels1 =  np.load(y_path[0])
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        self.labels1[self.labels1 < 1] = 0
        self.labels1[self.labels1 > 0] = 1
        self.labels1 = [int(label) for label in self.labels1]
        self.dataset1_size = len(self.labels1)
        if len(y_path) == 2:
            # labels for SDNET
            self.labels2 =  np.load(y_path[1])
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels2)} data points")
            self.y = np.append(self.labels1, self.labels2)

        elif len(y_path) > 2:

            self.labels3 =  np.load(y_path[1])
            self.labels3[self.labels3 < 1] = 0
            self.labels3[self.labels3 > 0] = 1
            self.labels3 = [int(label) for label in self.labels3]
            self.y = np.append(self.labels1, self.labels3)

            self.dataset1_size = len(self.y)
            print(f"Loaded dataset {y_path[1]}, with {len(self.labels3)} data points")
            
            self.labels2 =  np.load(y_path[2])
            self.labels2[self.labels2 < 1] = 0
            self.labels2[self.labels2 > 0] = 1
            self.labels2 = [int(label) for label in self.labels2]
            print(f"Loaded dataset {y_path[2]}, with {len(self.labels2)} data points")
            self.y = np.append(self.y, self.labels2)
        else:
            self.y = self.labels1

        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]

    def __getitem__(self, index):
        if self.augment_data:
            if index < self.dataset1_size or self.shuffle:
                X_datapoint1 = augment_impact_echo_data(self.X[index], self.sr[0])[:self.array_size]
                X_datapoint2 = augment_impact_echo_data(self.X[index], self.sr[0])[:self.array_size]
            else:
                X_datapoint1 = augment_impact_echo_data(self.X[index], self.sr[1])[:self.array_size]
                X_datapoint2 = augment_impact_echo_data(self.X[index], self.sr[1])[:self.array_size]
            return  [
                        torch.tensor(self.__normalize_data__(X_datapoint1), dtype=torch.float), 
                        torch.tensor(self.__normalize_data__(X_datapoint2), dtype=torch.float), 
                        torch.tensor(self.y[index], dtype=torch.int8),
                        torch.tensor(self.__normalize_data__(self.X[index,:self.array_size]), dtype=torch.float)
                ]
        else:
            return torch.tensor(self.__normalize_data__(self.X[index][:self.array_size]), dtype=torch.double), \
                   torch.tensor(self.y[index], dtype=torch.int8)
        
    def __normalize_data__(self, signal):
        return (signal - np.min(signal))/(np.max(signal)-np.min(signal))
    
    def __len__(self):
        return len(self.X)

    def get_labels(self):
        return self.labels1, self.labels2

    def __gettest__(self):
        return torch.tensor(self.X), self.y


from dataloaders.augmentations import *

class ImpactEchoDatasetBetter(Dataset):
    
    def __init__(self, augment_data = True):
        self.augment_data = augment_data
        self.array_size = 860
        self.sr = 500000
        self.X = np.array([])
        self.shuffle = shuffle
        X_train = np.load('data/data2/X_train_860.npy')
        y_train = np.load('data/data2/y_train_2cl.npy')
        X_train = np.column_stack((X_train, y_train))
        defectData = X_train[X_train[:,-1] != 0]
        goodData = X_train[X_train[:,-1] == 0]
        X_val = goodData[800:-1]
        self.X_val = np.row_stack((X_val, defectData[0:130]))
        Xy_train = np.row_stack((goodData[0:800], defectData[130:-1]))
        np.random.shuffle(Xy_train)
        self.X_train = Xy_train[:,:-1]
        self.y_train = Xy_train[:,-1]
        self.y_val = self.X_val[:, -1]


    def __getitem__(self, index):
        if self.augment_data:
            X_datapoint1 = augment_impact_echo_data(self.X_train[index], self.sr)[:self.array_size]
            X_datapoint2 = augment_impact_echo_data(self.X_train[index], self.sr)[:self.array_size]
            return  [
                        torch.tensor(self.__normalize_data__(X_datapoint1), dtype=torch.float), 
                        torch.tensor(self.__normalize_data__(X_datapoint2), dtype=torch.float), 
                        torch.tensor(self.y_train[index], dtype=torch.int8),
                        torch.tensor(self.__normalize_data__(self.X_train[index,:self.array_size]), dtype=torch.float)
                ]
        else:
            return torch.tensor(self.__normalize_data__(self.X_train[index][:self.array_size]), dtype=torch.double), \
                   torch.tensor(self.y_train[index], dtype=torch.int8)
        
    def __get_validation_data__(self):
        return torch.tensor(self.__normalize_data__(self.X_val[:,:-1]),  dtype=torch.float), \
                torch.tensor(self.y_val, dtype=torch.int8)
        
    def __normalize_data__(self, signal):
        return (signal - np.min(signal))/(np.max(signal)-np.min(signal))
    
    def __len__(self):
        return len(self.X_train)

    def __gettest__(self):
        return torch.tensor(self.X_train), self.y_train