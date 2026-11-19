import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        #Defining layers of the neural networks
        #There are 3 layers with a dim of 64, wich is arbitrary
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim) #The output is a brut value

    def forward(self, obs):
        #The input has to be a tensor, if it's not then it's convert into one
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs)) #ReLu is arbitrary
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
