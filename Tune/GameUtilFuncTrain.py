import os
import time
import pickle
import timeit
import random
import platform
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from tqdm import tqdm
from scipy import spatial
from pathlib import Path
from collections import defaultdict

from MultiVarGaussian import *

torch.set_default_tensor_type(torch.cuda.FloatTensor)

class UtilFunc(nn.Module):
    def __init__(self, n_bgn, n_end):
        super(UtilFunc, self).__init__()
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

if __name__ == "__main__":
    all_data, all_segs, all_reward = np.load("../Train_and_Fight/data_and_models/ExtractedGameData.npy", allow_pickle=True)
    joined_reward = list(itertools.chain(*all_reward))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    m = UtilFunc(2, 1).to(device)
    opt = optim.Adam(m.parameters())
    num_epoch = 5
    losses = []

    for _ in tqdm(range(num_epoch), ncols = 50):
        for reward in all_reward:
            if len(reward) < 5: continue
            gpr = GPR()
            gpr.fit(np.array(reward[:-1])[:,:-1], np.array(reward[:-1])[:,-1])
            choices = [random.choice(reward[:-1]) for _ in range(9)]
            mu, cov = gpr.predict(np.vstack([np.array(choices), np.array(reward[-1]).reshape(1,-1)])[:,:-1])
            temp_max = abs(mu).max()
            mu, cov = mu/temp_max, np.diag(cov)/temp_max
            scores = [m.forward(torch.Tensor([mu[i], cov[i]]).to(device).float().view(1,1,-1)) for i in range(len(mu))]
            loss = 0
            for score in scores[:-1]:
                loss += scores[-1] - score
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

    torch.save(m, "./data_and_models/GameUtilFunc.pt")