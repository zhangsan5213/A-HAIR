from trainer_ukyo import *

import os
import cv2
import copy
import pickle
import datetime
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.distributions import Categorical

matplotlib.use("TKAgg")

def readData(model_trainer, path):
    ## read the detailed timeline of all moves.
    ## lots of screenshots are required.
    model_trainer.hp1 = 100
    model_trainer.hp2 = 100
    data_flag = 0
    data = None ## in case an empty folder is present
    for root, _, files in os.walk(path): break
    files.sort()
    for file in files:
        model_trainer.game_img = cv2.imread(root + "/" + file, cv2.COLOR_BGR2RGB)
        if "game_imgs_train_old" in root:
            model_trainer.findHPs_smallWindow()
            model_trainer.findStates_detailed_smallWindow()
        else:
            model_trainer.findHPs()
            model_trainer.findStates_detailed()

        ## IMPROVED VERSION WITH DETAILED RECOGNITION OF THE CHARACTER STATES.
        ## THIS HAS NOT BEEN USED IN THE TRAINING SO FAR.
        if data_flag == 0:
            data = np.array(model_trainer.p1 + [np.ceil(model_trainer.hp1)] + model_trainer.p2 + [np.ceil(model_trainer.hp2)] + [model_trainer.zoomed])
            data_flag = 1
        else:
            data = np.vstack([data, np.array(model_trainer.p1 + [model_trainer.hp1] + model_trainer.p2 + [model_trainer.hp2] + [model_trainer.zoomed])])

    for j in range(1, len(data)):
        if data[j][3] > data[j-1][3]: data[j][3] = data[j-1][3]
        if data[j][7] > data[j-1][7]: data[j][7] = data[j-1][7]

    return data

def dataSegmentation(data, path, player=1, idle_states_indices=[9,10,11,12,13,14,15,16,46,50, 31,32,33,34,35,36,37,38,49,51]):
    ## segmented by the idling states and the change of HPs.
    if player == 1:
        p1_states = data[:,2].tolist()
        p2_states = data[:,6].tolist()
        p1_hps = data[:,3].tolist()
        p2_hps = data[:,7].tolist()
    elif player == 2:
        p2_states = data[:,2].tolist()
        p1_states = data[:,6].tolist()
        p2_hps = data[:,3].tolist()
        p1_hps = data[:,7].tolist()

    images = os.listdir(path)
    images.sort()
    if "game_imgs_train_old" in path:
        timestamps = [datetime.datetime.fromtimestamp(os.stat(os.path.join(path, image)).st_ctime) for image in images]
        images_time = [ts.hour*3600 + ts.minute*60 + ts.second + ts.microsecond/1000000 for ts in timestamps]
        images_time = [t - images_time[0] for t in images_time]
    else:
        images_time = [float(image[7:-4]) for image in images]

    segs, temp, hurting_flag = [], [], 0
    for i in range(1, len(data)):
        casting = (p1_states[i] not in idle_states_indices) or (p2_states[i] not in idle_states_indices)
        hurting = (p1_hps[i] < p1_hps[i-1]) or (p2_hps[i] < p2_hps[i-1])

        if (not casting) and (not hurting) and (temp != []):
            segs.append(temp)
            temp, hurting_flag = [], 0
        elif (hurting_flag == 0) and hurting:
            # temp.append([int(p1_states[i]), int(p2_states[i]), p1_hps[i] < p1_hps[i-1], p2_hps[i] < p2_hps[i-1], images_time[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]])
            temp.append([data[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]])
            hurting_flag = 1
        elif (hurting_flag == 1) and (temp != []) and (not hurting):
            segs.append(temp)
            if casting:
                # temp = [[int(p1_states[i]), int(p2_states[i]), p1_hps[i] < p1_hps[i-1], p2_hps[i] < p2_hps[i-1], images_time[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]]]
                temp = [[data[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]]]
            else:
                temp = []
            hurting_flag = 0
        else:
            # temp.append([int(p1_states[i]), int(p2_states[i]), p1_hps[i] < p1_hps[i-1], p2_hps[i] < p2_hps[i-1], images_time[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]])
            temp.append([data[i], p1_hps[i]-p1_hps[i-1], p2_hps[i]-p2_hps[i-1]])

    return segs

action_indices_h = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 17, 18, 19, 20, 21, 44, 45, 46]
action_indices_u = [22, 23, 24, 25, 26, 27, 28, 29, 30, 39, 40, 41, 42, 43, 47, 48]

movement_list = ["down_back+s_back_y", "face", "back+down_down_face+down_face_back+up_y", "back"] ## 0~3
fight_list = ["r", "t", "y", "back+down_down_face+down_face_face+up_y", "g", "h"] ## 4~9
skill_list = ["down_back+down_back_y", "back+down_down_face+down_face_up_y", "down_down+face_face_h", "back+down_down_face+down_face_up_y"] ## 10~13
combo_list = ["down_r", "down_t", "down_y", "down_f", "down_g", "down_h", "face_up", "back_up"] ## 14~21
action_list= movement_list + fight_list + skill_list + combo_list
model_trainer = trainer(None, "./winMark.png", action_list)

total_dirs = ["G:\\Samurai\\game_imgs_train_old\\game_imgs_train_32\\" + i for i in os.listdir("G:\\Samurai\\game_imgs_train_old\\game_imgs_train_32\\")] + \
             ["G:\\Samurai\\game_imgs_train_old\\game_imgs_train_33\\" + i for i in os.listdir("G:\\Samurai\\game_imgs_train_old\\game_imgs_train_33\\")]

# test = readData(model_trainer, total_dirs[0])
# segs = dataSegmentation(test, total_dirs[0])
# news = [s[-1] for s in segs]
# for i in range(len(news)):
#     reward = news[i][-2] - news[i][-1]
#     news[i] = np.hstack([news[i][0].reshape(1,-1), np.array([reward]).reshape(1,-1)]).flatten().tolist()
# news = [i for i in news if i[-1]!=0]

all_data, all_seg_data, all_news = [], [], []
for single_dir in tqdm(total_dirs, ncols=80):
    try:
        if "game_imgs_train_old" in single_dir: ## 1p is Haohmaru
            data = readData(model_trainer, single_dir)
            segs = dataSegmentation(data, single_dir, 1)
            news = [s[-1] for s in segs]
            for i in range(len(news)):
                reward = news[i][-2] - news[i][-1]
                news[i] = np.hstack([news[i][0].reshape(1,-1), np.array([reward]).reshape(1,-1)]).flatten().tolist()
            news = [i for i in news if i[-1]!=0]

        else: ## 1P is Ukyo
            data = readData(model_trainer, single_dir)
            segs = dataSegmentation(data, single_dir, 2)
            news = [s[-1] for s in segs]
            for i in range(len(news)):
                reward = news[i][-2] - news[i][-1]
                news[i] = np.hstack([news[i][0].reshape(1,-1), np.array([reward]).reshape(1,-1)]).flatten().tolist()
            news = [i for i in news if i[-1]!=0]

        all_data.append(data)
        all_seg_data.append(segs)
        all_news.append(news)

    except:
        continue

np.save("./data_and_models/ExtractedGameData.npy", [all_data, all_seg_data, all_news], allow_pickle=True)