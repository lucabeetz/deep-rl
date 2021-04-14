import math
import random
import gym
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.data_keys = ['states', 'actions', 'rewards', 'next_states', 'dones']
        self.reset()

    def reset(self):
        self.size = 0
        self.head = 0
        for k in self.data_keys:
            setattr(self, k, [])

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        for i, k in enumerate(self.data_keys):
            if self.size < self.capacity:
                getattr(self, k).append(None)
            getattr(self, k)[self.head] = experience[i]

        self.head = (self.head + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size):
        batch_idxs = np.random.randint(0, self.size, batch_size)
        batch = {k: [] for k in self.data_keys}
        for idx in batch_idxs:
            for k in self.data_keys:
                batch[k].append(getattr(self, k)[idx])

        for k in batch:
            batch[k] = np.array(batch[k])
            batch[k] = torch.from_numpy(batch[k].astype(np.float32))
        return batch

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.SELU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class DQN():
    def __init__(self, qnet, n_actions, gamma, tau_start, tau_end, tau_decay):
        self.qnet = qnet
        self.n_actions = n_actions

        self.gamma = gamma
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_decay = tau_decay

    def act(self, state, timestep=1e4):
        """
        Pick an action using a Boltzmann policy
        """
        tau = max(self.tau_end, self.tau_start -
                            timestep * (self.tau_start - self.tau_end) / self.tau_decay)

        state = torch.tensor(state, dtype=torch.float32)
        action_values = self.qnet(state)
        z = action_values - max(action_values)
        action_probs = torch.exp(z / tau) / torch.sum(torch.exp(z / tau))
        return Categorical(probs=action_probs).sample()

    def calc_q_loss(self, batch):
        """
        Calculate the Q-Learning loss
        loss = (Q(a, s) - r + gamma * Q_max(s_next))**2
        """
        states = batch['states']
        next_states = batch['next_states']

        q_preds = self.qnet(states)
        with torch.no_grad():
            next_q_preds = self.qnet(next_states)

        act_q_preds = q_preds.gather(-1, batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        act_next_max_q, _ = next_q_preds.max(dim=-1, keepdim=False)

        act_q_targets = batch['rewards'] + self.gamma * (1 - batch['dones']) * act_next_max_q
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
    Run a single trial
    """
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    qnet = QNet(obs_dim, n_actions)
    agent = DQN(qnet, n_actions, 0.99, 5.0, 0.5, 4000)
    optim = torch.optim.Adam(qnet.parameters())
    memory = ReplayMemory(10000)

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

            # Run multiple training steps
            if timestep > 32 and timestep % 4 == 0:
                batch = memory.sample(32)
                for _ in range(4):
                    agent.train(batch, optim)

            # Evaluate the current agent every 1000 timesteps
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