import gym
import torch
import numpy as np

from mpo import MPO
from mpo_nets import CategoricalActor, Critic

from tensorboardX import SummaryWriter


np.random.seed(0)
torch.manual_seed(0)


if __name__ == '__main__':
    writer = SummaryWriter("./exp/")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_envs = 5
    env_name = 'LunarLander-v2'

    vec_env = gym.vector.make(env_name, num_envs=5)

    obs_shape = vec_env.observation_space.shape[-1]
    action_shape = vec_env.action_space[0].n

    actor = CategoricalActor(obs_shape, action_shape)
    critic = Critic(obs_shape, action_shape)

    if device == 'cuda':
        actor.cuda()
        critic.cuda()
        
    def train():
        mpo = MPO(vec_env, actor, critic, obs_shape, action_shape, device=device)
        # mpo.load_model()
        mpo.train()

    def evaluate():
        env = gym.make(env_name)
        mpo = MPO(vec_env, actor, critic, obs_shape, action_shape, device=device)
        mpo.load_model()

        obs = env.reset()

        while True:
            act, _ = mpo.actor.action(torch.Tensor(np.array([obs])).to(device))
            act = act.cpu().detach().numpy()[0]

            obs, r, d, _ = env.step(act)
            env.render()

            if d:
                obs = env.reset()

    #evaluate()
    train()
