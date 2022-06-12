import torch
import torchvision #what does this do?
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import math

# Dataset and Dataloader
class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt("9.Wine.csv",delimiter=',',dtype=np.float32,skiprows=1) #skipping the header
        self.x = torch.from_numpy(xy[:, 1:].astype(np.float32)) #first column is thr wine class category
        self.y = torch.from_numpy(xy[:, [0]].astype(np.float32))  #makes it n_samples x 1, column vector

        self.n_samples, self.n_features = xy.shape 

    def __getitem__(self,index):
        #dataset[0] - helps in indexing data samples
        return self.x[index], self.y[index]
    def __len__(self):
        #len(dataset)
        return self.n_samples




if __name__ == "__main__":

    dataset = WineDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=4,shuffle=True,num_workers=1) #what is num workers? multiple subprocesses

    dataiter = iter(dataloader)
    data = dataiter.next()

    data = dataiter.next()
    features,labels = data
    print("features :")
    print(features)
    print("labels :")
    print(labels)