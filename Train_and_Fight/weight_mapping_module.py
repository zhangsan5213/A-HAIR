'''
Map the model weights and game parameters.
'''
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_default_tensor_type(torch.cuda.FloatTensor)

class mapping(nn.Module):
    def __init__(self, n_bgn, n_end):
        super(mapping, self).__init__()

        ## layers
        self.lin00 = nn.Linear(n_bgn, 64)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.BN = nn.BatchNorm1d(32)
        self.lin01 = nn.Linear(32*60, 64)
        self.lin02 = nn.Linear(64, 32)
        self.lin03 = nn.Linear(32, n_end)

        self.relu = nn.LeakyReLU()
        self.initParameters()

    def forward(self, x):
        x = self.relu(self.lin00(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.BN(self.relu(self.conv4(x))).flatten()
        x = self.relu(self.lin01(x))
        x = self.relu(self.lin02(x))
        x = self.relu(self.lin03(x))

        return x

    def initParameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)