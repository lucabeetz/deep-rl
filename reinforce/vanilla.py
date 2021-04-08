import gym
import torch
import numpy as np
from torch import nn
from torch.distributions.categorical import Categorical

gamma = 0.99
alpha = 1e-2


class Policy(nn.Module):
    def __init__(self, obs_dims, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dims, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )
        self.reset()

    def forward(self, x):
        return self.model(x)

    def reset(self):
        self.rewards = []
        self.log_probs = []

    def act(self, obs):
        x = torch.tensor(obs, dtype=torch.float32)
        pdparams = self.forward(x)
        pd = Categorical(logits=pdparams)
        action = pd.sample()
        self.log_probs.append(pd.log_prob(action))
        return action.item()


def train(policy, optimizer):
    T = len(policy.rewards)
    returns = torch.empty(T)
    total_return = 0
    for i in reversed(range(T)):
        total_return = policy.rewards[i] + gamma * total_return
        returns[i] = total_return

    log_probs = torch.stack(policy.log_probs)
    loss = -(log_probs * returns)
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = Policy(obs_dim, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)

    for epsi in range(300):
        observation = env.reset()
        for t in range(200):
            env.render()
            action = policy.act(observation)
            observation, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done:
                break

        train(policy, optimizer)
        print(f"Episode: {epsi}, return: {np.sum(policy.rewards)}")
        policy.reset()


if __name__ == '__main__':
    main()
