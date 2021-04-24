import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions.categorical import Categorical

gamma = 0.99
alpha = 1e-2


class Policy(nn.Module):
    """Policy network for taking actions, storing rewards and action probabilites"""

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
        """Delete stored rewards and action log probs"""
        self.rewards = []
        self.log_probs = []

    def act(self, obs, eval=False):
        """Pick action using a softmax distribution"""
        x = torch.tensor(obs, dtype=torch.float32)
        pdparams = self.forward(x)
        pd = Categorical(logits=pdparams)
        action = pd.sample()
        if not eval:
            self.log_probs.append(pd.log_prob(action))
        return action


def train(policy, optimizer):
    """Run a single training step on the policy network"""
    T = len(policy.rewards)
    returns = torch.empty(T)
    total_return = 0
    # Calculate "reward to go" for each timestep
    for i in reversed(range(T)):
        total_return = policy.rewards[i] + gamma * total_return
        returns[i] = total_return

    log_probs = torch.stack(policy.log_probs)
    loss = -(log_probs * returns)
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def evaluate_policy(policy):
    """Evaluate policy by running a single episode"""
    env = gym.make('CartPole-v0')
    state = env.reset()
    done = False
    episode_return = 0
    while not done:
        action = policy.act(state, eval=True).item()
        state, reward, done, _ = env.step(action)
        episode_return += reward
    return episode_return


def run_trial():
    """Run single trial for 10000 training steps"""
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy = Policy(obs_dim, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)

    return_hist = []
    timestep = 1

    while timestep < 1e5:
        observation = env.reset()
        done = False
        while not done:
            action = policy.act(observation).item()
            observation, reward, done, _ = env.step(action)
            policy.rewards.append(reward)

            if timestep % 1000 == 0:
                eval_return = evaluate(policy)
                return_hist.append(eval_return)

            timestep += 1

        # Train policy after each episode
        train(policy, optimizer)
        policy.reset()

    return np.array(return_hist)


def main():
    """Run a REINFORCE agent for ten trials on the CartPole-v0 env and plot results"""
    all_returns = []

    for i in range(10):
        trial_return = run_trial()
        all_returns.append(trial_return)
        print(f'Trial {i+1}, average trial return: {np.mean(trial_return)}')

    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    x = range(mean_returns.shape[0])
    plt.plot(x, mean_returns)
    plt.title('Mean return over 10 trials - REINFORCE')
    plt.fill_between(x, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    plt.ylabel('Mean return')
    plt.xlabel('1000 frames')
    plt.savefig('avg_return.png')
    plt.show()


if __name__ == '__main__':
    main()
