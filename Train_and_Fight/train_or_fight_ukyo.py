'''
Main functions for training and fighting.
'''
import os
import sys
import time
import pathlib
import argparse
import torch
from torch.autograd import Variable
from torch.distributions import Categorical

from trainer_ukyo import *
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train samshoII with MPO.')
    parser.add_argument('--train', type=int, default=1, help='Default 1 for training. Fight the Ukyo Agents with 0.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for training.')
    parser.add_argument('--statedict', type=str, default="", help='Path where the state dicts are kept.')
    parser.add_argument('--agentsdir', type=str, default="./data_and_models/samuraiAgents_Ukyo/", help='Path where the agent models are kept.')
    args = parser.parse_args()

    torch.manual_seed(42)
    num_action = len(action_list)
    num_state = 10 ## output dimension of model_trainer.grabGame()
    model_trainer = trainer(handle, "./winMark.png", action_list)

    actor  = CategoricalActor(num_state, num_action)
    critic = Critic(num_state, num_action)
    actor.cuda()
    critic.cuda()

    if args.statedict != "":
        mpo = MPO(None, actor, critic, num_state, num_action, device="cuda", save_path=args.statedict, episodes=args.episodes)
        mpo.load_model()
    elif bool(args.train):
        mpo = MPO(None, actor, critic, num_state, num_action, device="cuda", episodes=args.episodes)
    else:
        mpo = MPO(None, actor, critic, num_state, num_action, device="cuda", episodes=args.episodes)
        mpo.init_for_fight(args.agentsdir)

    if bool(args.train):
        mean_rewards, mean_q_losses, mean_policies = [], [], []
        mpo.train_samurai(model_trainer, mean_rewards, mean_q_losses, mean_policies)
    else:
        mpo.fight_samurai(model_trainer, coach=np.load("./data_and_models/game_candidates.npy"))
