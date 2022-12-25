import os
import time
import random
import pathlib
import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical
from scipy.optimize import minimize
from pydirectinput import press, keyUp, keyDown

from mpo_traj_buffer import TrajBuffer

class MPO(object):
    """
        Maximum A Posteriori Policy Optimization (MPO) ; Discrete action-space ; Retrace

        Params:
            env: gym environment
            actor: actor network
            critic: critic network
            obs_shape: shape of observation (from env)
            action_shape: shape of action
            dual_constraint: learning rate of η in g(η)
            kl_constraint: Hard constraint on KL
            learning_rate: Bellman equation's decay for Q-retrace
            clip: 
            alpha: scaling factor of the lagrangian multiplier in the M-step
            episodes: number of iterations to sample episodes + do updates
            sample_episodes: number of episodes to sample
            episode_length: length of each episode
            lagrange_it: number of Lagrangian optimization steps
            runs: amount of training updates before updating target parameters
            device: pytorch device
            save_path: path to save model to
    """
    def __init__(self, env, actor, critic, obs_shape, action_shape,
                 dual_constraint=0.1, kl_constraint=0.01,
                 learning_rate=0.99, alpha=1.0,
                 episodes=1000, sample_episodes=1, episode_length=600,
                 lagrange_it=10, runs=50, device='cpu',
                 save_path="./data_and_models/mpo_" + str(time.time()).split(".")[0] + ".pt"):
        # initialize env
        self.env = env

        # initialize some hyperparameters
        self.α = alpha  
        self.ε = dual_constraint 
        self.ε_kl = kl_constraint
        self.γ = learning_rate ## used to be 0.99
        self.episodes = episodes
        self.sample_episodes = sample_episodes
        self.episode_length = episode_length
        self.lagrange_it = lagrange_it
        # self.mb_size = (episode_length) * env.num_envs
        self.mb_size = (episode_length) * 1
        self.runs = runs
        self.device = device

        # initialize networks and optimizer
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.critic = critic
        self.target_critic = deepcopy(critic)
        for target_param, param in zip(self.target_critic.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-5)

        self.actor = actor
        self.target_actor = deepcopy(actor)
        for target_param, param in zip(self.target_actor.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
            target_param.requires_grad = False
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-5)

        # initialize Lagrange Multiplier
        self.η = np.random.rand()
        self.η_kl = 0.0

        # buffer and others
        self.buffer = TrajBuffer(env, episode_length, 100000, obs_dim = obs_shape)
        self.save_path = save_path

    def _sample_trajectory(self):
        mean_reward = 0

        for _ in range(self.sample_episodes):
            obs = self.env.reset()
            done = False

            obs_b = np.zeros([self.episode_length, self.env.num_envs, self.obs_shape])
            action_b = np.zeros([self.episode_length, self.env.num_envs])
            reward_b = np.zeros([self.episode_length, self.env.num_envs])
            prob_b = np.zeros([self.episode_length, self.env.num_envs, self.action_shape])
            done_b = np.zeros([self.episode_length, self.env.num_envs])

            for steps in range(self.episode_length):
                action, prob = self.target_actor.action(torch.from_numpy(np.expand_dims(obs, axis=0)).to(self.device).float())
                action = np.reshape(action.cpu().numpy(), -1)
                prob = prob.cpu().numpy()

                obs_b[steps] = obs
                done_b[steps] = done

                obs, reward, done, _ = self.env.step(action)
                mean_reward += reward

                action_b[steps] = action
                reward_b[steps] = reward
                prob_b[steps] = prob
            
            self.buffer.put(obs_b, action_b, reward_b, prob_b, done_b)

        return mean_reward / self.episode_length / self.sample_episodes

    def _sample_trajectory_samurai(self, model_trainer, episode, load_save=1, lag=1, coach=None):
        print("Sampling now for episode ", str(episode).rjust(4, "0"), ".\n")

        mean_reward = 0
        if episode == 0:
            model_trainer.activateGame()
        else:
            model_trainer.resumeGame()
        print("Game activated.\n")

        for sample_episode in range(self.sample_episodes):
            # episode_game_img_save_path = "./game_imgs_train/" + str(epoch + int(sys.argv[1])) + "/"
            # pathlib.Path(episode_game_img_save_path).mkdir(parents=True, exist_ok=True)
            # this_save_path = "./game_imgs_train/" + str(episode) + "/" + str(sample_episode) + "/"
            this_save_path = "./game_imgs_train/" + str(episode) + "/"
            pathlib.Path(this_save_path).mkdir(parents=True, exist_ok=True)

            img_index = 0
            temp_hp_p1 = 100
            temp_hp_p2 = 100
            model_trainer.hp1 = 100
            model_trainer.hp2 = 100
            relative_direction = -1
            back = "a" if load_save==1 else "d"
            face = "d" if load_save==1 else "a"
            # back = "a"

            model_trainer.reloadFromSave(load_save)

            info_vec = model_trainer.grabGame()#this_save_path + str(img_index).rjust(3, "0") + ".png")
            last_info_vec = info_vec.copy()

            done = False

            obs_b = np.zeros([self.episode_length, 1, self.obs_shape])
            action_b = np.zeros([self.episode_length, 1])
            reward_b = np.zeros([self.episode_length, 1])
            prob_b = np.zeros([self.episode_length, 1, self.action_shape])
            done_b = np.zeros([self.episode_length, 1])

            for steps in range(self.episode_length):
                info_vec = model_trainer.grabGame()#this_save_path + str(img_index).rjust(3, "0") + ".png")
                if info_vec[1] == -1: info_vec[0:4] = last_info_vec[0:4]
                if info_vec[6] == -1: info_vec[5:8] = last_info_vec[5:8] ## if not detected, replace everything except the HPs

                torched = torch.from_numpy(np.expand_dims(info_vec, axis=0)).to(self.device).float()
                action, prob = self.target_actor.action(torched)
                action, prob = np.reshape(action.cpu().numpy(), -1), prob.cpu().numpy()

                if model_trainer.position_1_projectile == -1:
                    if coach!=None:
                        prob_list = prob.flatten().tolist()
                        ## This is only called during fighting for luring the players.
                        prob_index = random.choice(coach)
                        if 0<=prob_index<len(prob_list):
                            prob_list[prob_index] = prob_list[prob_index] + abs(np.random.normal())
                            new_action = np.argmax(prob_list)
                            model_trainer.inputAction(new_action, face, back)
                        else:
                            model_trainer.inputAction(action[0], face, back)
                    else:
                        model_trainer.inputAction(action[0], face, back)
                    time.sleep(lag*0.4 + 0.2)

                    next_info_vec = model_trainer.grabGame()#this_save_path + str(img_index).rjust(3, "0") + ".png")
                    if (next_info_vec[1]!=-1) and (next_info_vec[6]!=-1):
                        relative_direction = np.sign(next_info_vec[1]-next_info_vec[6]) ## x_{Ukyo} - x_{Haohmaru}
                    keyUp(back)
                    if relative_direction < 0:
                        print("Ukyo on the left")
                        face, back = "d", "a"
                    else:
                        print("Ukyo on the right")
                        face, back = "a", "d"
                    keyDown(back)

                    read_hp_p1, read_hp_p2 = next_info_vec[4], next_info_vec[-1]
                    reward = ((read_hp_p1-temp_hp_p1) + (temp_hp_p2-read_hp_p2)) * 1
                    mean_reward += reward

                    obs_b[steps] = info_vec
                    done_b[steps] = bool(done)
                    action_b[steps] = action
                    reward_b[steps] = reward
                    prob_b[steps] = prob

                    done = bool(model_trainer.endGame())
                    if done:
                        done_b[steps+1] = done
                        break

                    temp_hp_p1, temp_hp_p2 = read_hp_p1, read_hp_p2
                    info_vec = next_info_vec
                    img_index += 1

                else:
                    time.sleep(lag*0.55 + 0.2)
                    info_vec = model_trainer.grabGame(this_save_path + str(img_index).rjust(3, "0") + ".png")
                    temp_hp_p1, temp_hp_p2 = info_vec[4], info_vec[-1]
                    continue
            
            keyUp(back)
            self.buffer.put(obs_b, action_b, reward_b, prob_b, done_b)

        print(done)
        model_trainer.pauseGame()
        print("Game minimized.\n")
        return mean_reward / self.episode_length / self.sample_episodes, temp_hp_p1, temp_hp_p2

    def _update_critic_retrace(self, state_batch, action_batch, policies_batch, reward_batch, done_batch):
        try:
            cutoff_index = np.where(done_batch == True)[0][0] - 8
            ## The last eight frames are removed since they refer to the end-round animation.
        except:
            cutoff_index = -1

        state_batch_last = state_batch[cutoff_index]

        state_batch = state_batch[0:cutoff_index]
        action_batch = action_batch[0:cutoff_index]
        policies_batch = policies_batch[0:cutoff_index]
        reward_batch = reward_batch[0:cutoff_index]

        action_size = policies_batch.shape[-1]
        nsteps = state_batch.shape[0]
        n_envs = state_batch.shape[1]

        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            policies, a_log_prob, entropy = self.actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))
            target_policies, _, _ = self.target_actor.evaluate_action(state_batch.view(-1, self.obs_shape), action_batch.view(-1, 1))

        qval = self.critic(state_batch.view(-1, self.obs_shape))
        val = (qval * policies).sum(1, keepdim=True)

        old_policies = policies_batch.view(-1, action_size)
        policies = policies.view(-1, action_size)
        target_policies = target_policies.view(-1, action_size)

        val = val.view(-1, 1)
        qval = qval.view(-1, action_size)
        a_log_prob = a_log_prob.view(-1, 1)
        actions = action_batch.view(-1, 1)

        q_i = qval.gather(1, actions.long())
        rho = policies / (old_policies + 1e-10)
        rho_i = rho.gather(1, actions.long())

        with torch.no_grad():
            next_qval = self.critic(state_batch_last).detach()
            next_policies = self.actor.get_action_prob(state_batch_last).detach()
            next_val = (next_qval * next_policies).sum(1, keepdim=True)
        
        q_retraces = reward_batch.new(nsteps + 1, n_envs, 1).zero_()
        q_retraces[-1] = next_val

        for step in reversed(range(nsteps)):
            q_ret = reward_batch[step] + self.γ * q_retraces[step + 1] * (1 - done_batch[step + 1])
            q_retraces[step] = q_ret
            q_ret = (rho_i[step] * (q_retraces[step] - q_i[step])) + val[step]
        
        q_retraces = q_retraces[:-1]
        q_retraces = q_retraces.view(-1, 1)

        q_loss = (q_i - q_retraces.detach()).pow(2).mean() * 0.5
        q_loss.backward()
        clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        return q_loss.detach()

    def _categorical_kl(self, p1, p2):
        p1 = torch.clamp_min(p1, 0.0001)
        p2 = torch.clamp_min(p2, 0.0001)
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

    def _update_param(self):
        # Update policy parameters
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        # Update critic parameters
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def train(self):
        # start training
        start_time = time.time()

        for episode in range(self.episodes):

            # Update replay buffer
            mean_reward = self._sample_trajectory()
            mean_q_loss = 0
            mean_policy = 0

            # Find better policy by gradient descent
            for _ in range(self.runs):
                state_batch, action_batch, reward_batch, policies_batch, done_batch = self.buffer.get()

                state_batch = torch.from_numpy(state_batch).to(self.device).float()
                action_batch = torch.from_numpy(action_batch).to(self.device).float()
                reward_batch = torch.from_numpy(reward_batch).to(self.device).float()
                policies_batch = torch.from_numpy(policies_batch).to(self.device).float()
                done_batch = torch.from_numpy(done_batch).to(self.device).float()

                reward_batch = torch.unsqueeze(reward_batch, dim=-1)
                done_batch = torch.unsqueeze(done_batch, dim=-1)

                # Update Q-function
                q_loss = self._update_critic_retrace(state_batch, action_batch, policies_batch, reward_batch, done_batch)
                mean_q_loss += q_loss

                # Sample values
                state_batch = state_batch.view(self.mb_size, *tuple(state_batch.shape[2:]))
                action_batch = action_batch.view(self.mb_size, *tuple(action_batch.shape[2:]))

                with torch.no_grad():
                    actions = torch.arange(self.action_shape)[..., None].expand(self.action_shape, self.mb_size).to(self.device)
                    b_p = self.target_actor.forward(state_batch)
                    b = Categorical(probs=b_p)
                    b_prob = b.expand((self.action_shape, self.mb_size)).log_prob(actions).exp()
                    target_q = self.target_critic.forward(state_batch)
                    target_q = target_q.transpose(0, 1)
                    b_prob_np = b_prob.cpu().numpy() 
                    target_q_np = target_q.cpu().numpy()
                
                # E-step
                # Update Dual-function
                def dual(η):
                    """
                    dual function of the non-parametric variational
                    g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                    """
                    max_q = np.max(target_q_np, 0)
                    return η * self.ε + np.mean(max_q) \
                        + η * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q) / η), axis=0)))

                bounds = [(1e-6, None)]
                res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                self.η = res.x[0]

                # calculate the new q values
                qij = torch.softmax(target_q / self.η, dim=0)

                # M-step
                # update policy based on lagrangian
                for _ in range(self.lagrange_it):
                    π_p = self.actor.forward(state_batch)
                    π = Categorical(probs=π_p)
                    loss_p = torch.mean(
                        qij * π.expand((self.action_shape, self.mb_size)).log_prob(actions)
                    )
                
                    kl = self._categorical_kl(p1=π_p, p2=b_p)

                    # Update lagrange multipliers by gradient descent
                    self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

                    if self.η_kl < 0.0:
                        self.η_kl = 0.0

                    self.actor_optimizer.zero_grad()
                    loss_policy = -(loss_p + self.η_kl * (self.ε_kl - kl))
                    loss_policy.backward()
                    clip_grad_norm_(self.actor.parameters(), 5.0)
                    self.actor_optimizer.step()
                    mean_policy += loss_policy.item()

            # Update target parameters
            self._update_param()

            print(f"Episode = {episode} ; "
                  f"Mean reward = {np.mean(mean_reward) / self.episode_length / self.sample_episodes} ; "
                  f"Mean Q loss = {mean_q_loss / self.runs} ; "
                  f"Policy loss = {mean_policy / self.runs} ; "
                  f"η = {self.η} ; η_kl = {self.η_kl} ; "
                  f"time = {(time.time() - start_time):.2f}")

            # Save model
            self.save_model()

    def winrate_test(self, model_trainer, agents_dir, num_round=10):
        test_agents_path = os.listdir(agents_dir)
        win_rates = []
        activate = 0
        for agent_path in test_agents_path:
            if agent_path.endswith(".pt"):
                self.save_path = os.path.join(agents_dir, agent_path)
                self.load_model() ## load the saved agents to the mpo model
                this_win_rate = []

                for i in range(num_round):
                    _, hp1, hp2 = self._sample_trajectory_samurai(model_trainer, episode=activate, load_save=1, lag=0)
                    this_win_rate.append(int(hp1>hp2))
                    activate += 1

                win_rates.append([self.save_path, np.mean(this_win_rate)])

        with open(os.path.join(agents_dir, "win_rate_rec.txt"), "w+") as f:
            for i in range(len(win_rates)):
                f.write(str(win_rates[i][0]) + "\t" + str(win_rates[i][1]) + "\n")

    def init_for_fight(self, agents_dir):
        self.agents_dir = agents_dir
        self.agents = os.listdir(agents_dir)
        self.agents.sort(reverse=True)
        self.save_path = os.path.join(agents_dir, self.agents[-1])
        self.load_model() ## load the saved agents to the mpo model
        self.num_agents = len(self.agents)
        print("Agents initiated for fight.")

    def fight_samurai(self, model_trainer, coach=None):
        for episode in range(self.episodes):
            _, hp1, hp2 = self._sample_trajectory_samurai(model_trainer, episode, load_save=0, lag=0, coach=coach)
            self.save_path = os.path.join(self.agents_dir, self.agents[episode % self.num_agents]) ## 0~10
            self.load_model()
            time.sleep(2)

    def train_samurai(self, model_trainer, mean_rewards, mean_q_losses, mean_policies):
        # start training
        start_time = time.time()
        for episode in range(self.episodes):

            # Update replay buffer
            mean_reward = self._sample_trajectory_samurai(model_trainer, episode)
            mean_rewards.append(mean_reward)
            mean_q_loss = 0
            mean_policy = 0

            # Find better policy by gradient descent
            for _ in range(self.runs):
                state_batch, action_batch, reward_batch, policies_batch, done_batch = self.buffer.get()

                state_batch = torch.from_numpy(state_batch).to(self.device).float()
                action_batch = torch.from_numpy(action_batch).to(self.device).float()
                reward_batch = torch.from_numpy(reward_batch).to(self.device).float()
                policies_batch = torch.from_numpy(policies_batch).to(self.device).float()
                done_batch = torch.from_numpy(done_batch).to(self.device).float()

                reward_batch = torch.unsqueeze(reward_batch, dim=-1)
                done_batch = torch.unsqueeze(done_batch, dim=-1)

                # Update Q-function
                q_loss = self._update_critic_retrace(state_batch, action_batch, policies_batch, reward_batch, done_batch)
                mean_q_loss += q_loss

                # Sample values
                state_batch = state_batch.view(self.mb_size, *tuple(state_batch.shape[2:]))
                action_batch = action_batch.view(self.mb_size, *tuple(action_batch.shape[2:]))

                with torch.no_grad():
                    actions = torch.arange(self.action_shape)[..., None].expand(self.action_shape, self.mb_size).to(self.device)
                    b_p = self.target_actor.forward(state_batch)
                    b = Categorical(probs=b_p)
                    b_prob = b.expand((self.action_shape, self.mb_size)).log_prob(actions).exp()
                    target_q = self.target_critic.forward(state_batch)
                    target_q = target_q.transpose(0, 1)
                    b_prob_np = b_prob.cpu().numpy() 
                    target_q_np = target_q.cpu().numpy()
                
                # E-step
                # Update Dual-function
                def dual(η):
                    """
                    dual function of the non-parametric variational
                    g(η) = η*ε + η \sum \log (\sum \exp(Q(a, s)/η))
                    """
                    max_q = np.max(target_q_np, 0)
                    return η * self.ε + np.mean(max_q) \
                        + η * np.mean(np.log(np.sum(b_prob_np * np.exp((target_q_np - max_q) / η), axis=0)))

                bounds = [(1e-6, None)]
                res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
                self.η = res.x[0]

                # calculate the new q values
                qij = torch.softmax(target_q / self.η, dim=0)

                # M-step
                # update policy based on lagrangian
                for _ in range(self.lagrange_it):
                    π_p = self.actor.forward(state_batch)
                    π = Categorical(probs=π_p)
                    loss_p = torch.mean(
                        qij * π.expand((self.action_shape, self.mb_size)).log_prob(actions)
                    )
                
                    kl = self._categorical_kl(p1=π_p, p2=b_p)

                    # Update lagrange multipliers by gradient descent
                    self.η_kl -= self.α * (self.ε_kl - kl).detach().item()

                    if self.η_kl < 0.0:
                        self.η_kl = 0.0

                    self.actor_optimizer.zero_grad()
                    loss_policy = -(loss_p + self.η_kl * (self.ε_kl - kl))
                    loss_policy.backward()
                    clip_grad_norm_(self.actor.parameters(), 5.0)
                    self.actor_optimizer.step()
                    mean_policy += loss_policy.item()

            # Update target parameters
            self._update_param()

            print(f"Episode = {episode} ; "
                  f"Mean reward = {np.mean(mean_reward) / self.episode_length / self.sample_episodes} ; "
                  f"Mean Q loss = {mean_q_loss / self.runs} ; "
                  f"Policy loss = {mean_policy / self.runs} ; "
                  f"η = {self.η} ; η_kl = {self.η_kl} ; "
                  f"time = {(time.time() - start_time):.2f}")
            mean_q_losses.append(mean_q_loss.view(1,).detach().cpu().numpy()[0])
            mean_policies.append(mean_policy)

            # Save model
            self.save_model()
            
    def load_model(self):
        if pathlib.Path(self.save_path).exists():
            checkpoint = torch.load(self.save_path)
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
            self.η = checkpoint['lagrange_η']
            self.η_kl = checkpoint['lagrange_η_kl']
            self.critic.train()
            self.target_critic.train()
            self.actor.train()
            self.target_actor.train()

    def save_model(self):
        data = {
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'critic_optim_state_dict': self.critic_optimizer.state_dict(),
            'actor_optim_state_dict': self.actor_optimizer.state_dict(),
            'lagrange_η': self.η,
            'lagrange_η_kl': self.η_kl
        }
        torch.save(data, self.save_path)
