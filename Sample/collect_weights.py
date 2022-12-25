'''
Collect the perturbed weights and the corresponding loss so as to find the scenarios to solve.
'''
import os
import time
import pickle
import timeit
import platform
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import spatial
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from train import *
from model_E_rotate_molecule_decay import GOODLE_E

def f_perturb(_state_dict):
    state_dict = _state_dict.copy()
    for key in state_dict.keys():
        if ("weight" in key) or ("bias" in key):
            state_dict[key] = torch.normal(mean=state_dict[key], std=state_dict[key].abs()*0.05)
    return state_dict

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bgn_state_dict = torch.load("./temp_weights/bgn.pt", map_location=device).state_dict()
    trial_model = GOODLE_E(device=device)
    all_names = os.listdir("./all_data_qm9/")

    point_num = 50
    sample_num = 50

    for i in range(point_num):
        np.random.shuffle(all_names)
        trial_model.load_state_dict(f_perturb(bgn_state_dict))

        # preds, trues = [], []
        losss = []

        for j in tqdm(range(sample_num), ncols=80):
            this_name = "./pre_processed_atomic_number_qm9/" + all_names[j] + ".npy"
            try:
                positions, atomic_numbers, distance_matrices, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks, true = np.load(this_name, allow_pickle=True)

                pred = trial_model.forward(positions, atomic_numbers, distance_matrices, ori_dist, dist_xs, dist_ys, dist_zs, kRs, accessible_R, ks)
                true = torch.tensor(true, device=device)
                loss = F.mse_loss(pred, true)

                # trues.append(true.item())
                # preds.append(pred.item())
                losss.append(loss.item())
                
            except:
                continue

        torch.save(trial_model.state_dict(), "./temp_state_dicts/{}.pt".format("%e"%(sum(losss)/len(losss))))
        # exit()