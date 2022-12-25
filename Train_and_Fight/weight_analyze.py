'''
Analyze the weight collected from the model and alter the strategy of the AI opponent accordingly.
'''
import os
import pickle
import numpy as np
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn

import sys
sys.path.insert(1, '../Sample/')

from weight_mapping_module import *
from model_E_rotate_molecule_decay import GOODLE_E

# ### The read part. ###

# size = 32
# record = []

# state_dicts = ["../Sample/temp_state_dicts/" + i for i in os.listdir("../Sample/temp_state_dicts/")]

# for state_dict in tqdm(state_dicts, ncols=50):
#     loss = float(state_dict.split("/")[-1][:-3])
#     state_dict = torch.load(state_dict, map_location="cuda")
#     record.append([loss, list()])

#     for key in state_dict.keys():
#         this_key = state_dict[key].flatten()
#         this_len = this_key.numel()
#         if this_len >= size:
#             for j in range(this_len // size):
#                 record[-1][-1].append(this_key[j*size:(j+1)*size].detach().cpu().numpy())

# np.save("./temp_save/record.npy", record, allow_pickle=True)

# ### The load part. ###

# record = np.load("./temp_save/record.npy", allow_pickle=True)
# n_points, n_segments = len(record), len(record[0][1])
# seg_info = [list() for _ in range(n_segments)]
# seg_loss = [rec[0] for rec in record]

# for i in range(n_points):
#     for j in range(n_segments):
#         seg_info[j].append(record[i][1][j])

# np.save("./temp_save/seg_info.npy", seg_info, allow_pickle=True)
# np.save("./temp_save/seg_loss.npy", seg_loss, allow_pickle=True)

# ### Identify the loss surface geometry, map with the game for strategy modification. ###

seg_info = np.load("./temp_save/seg_info.npy", allow_pickle=True)
seg_reward = -np.load("./temp_save/seg_loss.npy", allow_pickle=True)
game_info = np.load("./data_and_models/ExtractedGameData.npy", allow_pickle=True)
game_reward = list(itertools.chain(*game_info[-1]))

seg_reward = seg_reward/abs(seg_reward).max() * np.max(abs(np.array(game_reward)), axis=0)[-1]

model_size = seg_info[0].shape[1]
game_size = len(game_reward[0]) - 1
device = "cuda" if torch.cuda.is_available() else "cpu"
loss_fn = nn.MSELoss()

# n_surrogate_epoch, surrogate_epoch_sample_num = 1000, 256
# game_surrogate = mapping(game_size, 1).to(device)
# opt = optim.Adam(game_surrogate.parameters())
# for epoch in tqdm(range(n_surrogate_epoch), ncols=100):
#     for i, game_info in enumerate(game_reward[:surrogate_epoch_sample_num]):
#         pred = game_surrogate.forward(torch.Tensor(game_info[:-1]).to(device).float().view(1,1,-1))
#         loss = loss_fn(pred, torch.Tensor([game_info[-1]]).to(device).float())
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
# torch.save(game_surrogate, "./data_and_models/game_surrogate_heavy.pt")

def shuffle_along_axis(array, axis):
    idx = np.random.rand(*array.shape).argsort(axis=axis)
    return np.take_along_axis(array,idx,axis=axis)
seg_info = shuffle_along_axis(seg_info, 0)

game_surrogate = torch.load("./data_and_models/game_surrogate_heavy.pt")
game_surrogate.eval()
n_mappping_epoch, mapping_sample_num = 100, 1000
game_candidates = [0] * 12 # number of actions in trainer
for i, seg in enumerate(tqdm(seg_info[:mapping_sample_num], ncols=50)):
    m = mapping(model_size, game_size).to(device)
    opt = optim.Adam(m.parameters())
    for epoch in range(n_mappping_epoch):
        for j, s in enumerate(seg):
            pred = m.forward(torch.Tensor(s).to(device).float().view(1,1,-1)).view(1,1,-1)
            loss = loss_fn(game_surrogate(pred), torch.Tensor([seg_reward[j]]).to(device).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    seg_candidate = seg[np.argmax(seg_reward)]
    game_candidate = m.forward(torch.Tensor(seg_candidate).to(device).float().view(1,1,-1)).view(1,1,-1)
    game_candidates.append(game_candidate.detach().cpu().numpy().flatten().tolist()[6])

np.save("./data_and_models/game_candidates.npy", game_candidates, allow_pickle=True)