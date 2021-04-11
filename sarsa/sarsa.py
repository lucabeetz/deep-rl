"""
This is an implementation of the SARSA algorithm using a simple neural network as
function approximator
"""

import math
import random
import gym
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class Memory():
    """
    Memory class for storing on-policy experiences for training
    """

    def __init__(self):
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.size = 0
        self.reset()

    def reset(self):
        """
        Delete all experiences stored in memory
        """
        for k in self.data_keys:
            setattr(self, k, [])
        self.size = 0

    def add_experience(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory
        """
        experience = (state, action, reward, next_state, done)
        for i, k in enumerate(self.data_keys):
            getattr(self, k).append(experience[i])
        self.size += 1

    def sample(self):
        """
        Return all experiences currently stored in memory and create an additional list containing
        all next_actions needed for calculating the Sarsa target
        """
        batch = {k: getattr(self, k) for k in self.data_keys}
        batch['next_actions'] = np.zeros_like(batch['actions'])
        batch['next_actions'][:-1] = batch['actions'][1:]

        # Convert batch to tensors
        for k in batch:
            batch[k] = np.array(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))

        self.reset()
        return batch


class QNet(nn.Module):
    """
    NN used as function approximator for the Q-function
    """

    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Sarsa():
    """
    Class for a Sarsa agent
    """

    def __init__(self, qnet, n_actions, gamma, eps_start, eps_end, eps_decay):
        self.qnet = qnet
        self.n_actions = n_actions

        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def act(self, state, timestep=1e5):
        """
        Pick an action according to an epsilon-greedy policy with a linearly decaying epsilon
        """
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * timestep / self.eps_decay)

        eps_threshold = max(self.eps_end, self.eps_start -
                            timestep * (self.eps_start - self.eps_end) / self.eps_decay)

        if sample > eps_threshold:
            state = torch.tensor(state, dtype=torch.float32)
            return self.qnet(state).argmax()
        else:
            return torch.tensor(random.randrange(self.n_actions), dtype=torch.long)

    def calc_q_loss(self, batch):
        """
        Calculate the SARSA Q-loss
        loss = (Q(a, s) - r + gamma * Q(a_next, s_next))**2
        """
        states = batch['states']
        next_states = batch['next_states']

        q_preds = self.qnet(states)
        with torch.no_grad():
            next_q_preds = self.qnet(next_states)

        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        act_next_q_preds = next_q_preds.gather(-1,
                                               batch['next_actions'].long().unsqueeze(-1)).squeeze(-1)

        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * act_next_q_preds
        q_loss = nn.functional.mse_loss(act_q_preds, act_q_targets)
        return q_loss

    def train(self, batch, optim):
        """
        Run one optimization step
        """
        loss = self.calc_q_loss(batch)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss


def evaluate(agent):
    """
    Evaluate the current agent on a single episode
    """
    env = gym.make('CartPole-v0')
    state = env.reset()
    done = False
    episode_return = 0
    while not done:
        action = agent.act(state).item()
        state, reward, done, _ = env.step(action)
        episode_return += reward
    return episode_return


def run_trial():
    """
    Run a single trial of training
    """
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    qnet = QNet(obs_dim, n_actions)
    agent = Sarsa(qnet, n_actions, 0.99, 1.0, 0.05, 1e4)
    optim = torch.optim.RMSprop(qnet.parameters(), lr=0.01)
    memory = Memory()

    return_hist = []
    timestep = 1

    while timestep < 1e5:
        state = env.reset()
        done = False
        while not done:
            # Pick action and run a single environment step
            action = agent.act(state, timestep).item()
            next_state, reward, done, _ = env.step(action)
            # Add experience to memory for training
            memory.add_experience(state, action, reward, next_state, done)

            state = next_state

            # Run a single training step every 32 timesteps
            if timestep % 32 == 0:
                batch = memory.sample()
                agent.train(batch, optim)

            # Evaluate the current agent every 1000 agents
            if timestep % 1000 == 0:
                eval_return = evaluate(agent)
                return_hist.append(eval_return)

            timestep += 1

    return np.array(return_hist)


def main():
    """
    Runs multiple trials of the implemented SARSA agent
    """
    all_returns = []

    for i in range(10):
        trial_return = run_trial()
        all_returns.append(trial_return)
        print(f'Trial {i+1}, average trial return: {np.mean(trial_return)}')

    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    x = range(mean_returns.shape[0])
    plt.plot(x, mean_returns)
    plt.title('Mean return over 10 trials')
    plt.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    plt.ylabel('Mean return')
    plt.xlabel('1000 frames')
    plt.savefig('avg_return.png')
    plt.show()


if __name__ == '__main__':
    main()
