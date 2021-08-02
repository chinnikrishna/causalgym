import torch
from causalgym.envs.task1 import Task1Env
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from torchvision import utils

class DQNAgent:
    def __init__(self, device, batch_size=64, lr=5e-4, update_step=10, gamma=0.99, min_eps=1.0):
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.update_step = update_step
        self.gamma = gamma
        self.warmup_batches = 5

        # Target Network
        self.target_net = QNet().to(device)
        # Policy Network
        self.policy_net = QNet().to(device)
        # Buffer to store experiences
        self.replay_buffer = ReplayBuffer(device)
        # Num steps
        self.time_step = 0
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        # Epsilon for explore/exploit
        self.min_eps = min_eps
        self.epsilons = 0.01 / np.logspace(-2, 0, 20000, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (1.0 - self.min_eps) + self.min_eps

    def get_action(self, state):
        eps = self.min_eps if self.time_step > 20000 else self.epsilons[self.time_step]
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            actions = self.policy_net(state).cpu().data.numpy()
        self.time_step += 1
        if random.random() > eps:
            return np.argmax(actions)
        else:
            return random.choice(np.arange(5))


    def process_observation(self, state, action, reward, next_state, done):
        next_state = np.moveaxis(next_state, 2, 0)
        next_state = torch.Tensor(next_state[np.newaxis, :])
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) > (self.batch_size * self.warmup_batches):
            experiences = self.replay_buffer.sample_batch()
            states, actions, rewards, next_states, dones = experiences
            q_sp = self.target_net(next_states).detach()
            q_sp_max = q_sp.max(1)[0].unsqueeze(1)
            q_target = rewards + (self.gamma * q_sp_max * (1 - dones))
            q_policy = self.policy_net(states).gather(1, actions)
            td_error = q_policy - q_target
            loss = td_error.pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.time_step % 100 == 0:
                print(self.time_step, loss.cpu().data.numpy())
        if self.time_step % self.update_step == 0:
            for target, policy in zip(self.target_net.parameters(),
                                      self.policy_net.parameters()):
                target.data.copy_(policy.data)


class QNet(nn.Module):
    """
    Takes observation image of 256x256 as input and outputs an action
    """
    def __init__(self):
        super(QNet, self).__init__()
        self.qnet = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(28800, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        return self.qnet(x)

class ReplayBuffer:
    def __init__(self, device, buffer_size=int(1e4), batch_size=32):
        self.device = device
        self.batch_size = batch_size

        self.buffer = deque(maxlen=buffer_size)

        self.experience = namedtuple('experience', field_names=['state', 'action', 'reward',
                                    'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample_batch(self):
        experiences = random.sample(self.buffer, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = Task1Env()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = DQNAgent(device)
    for i in range(10):
        state, reward, done = env.reset()
        steps = 0
        while not done:
            env.render()
            state = np.moveaxis(state, 2, 0)
            state = torch.Tensor(state[np.newaxis, :])
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.process_observation(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if steps > 10000:
                break
            steps += 1
    # nrow = 8
    # padding = 1
    # tensor = agent.target_net.qnet[0].weight.cpu()
    # print(tensor.shape)
    # n,c,w,h = tensor.shape
    # tensor = tensor[:,0,:,:].unsqueeze(dim=1)
    # # tensor = tensor.view(n*c, -1, w, h)
    # rows = np.min((tensor.shape[0] // nrow + 1, 64))
    # grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    # plt.figure( figsize=(nrow,rows) )
    # plt.imshow(grid.numpy().transpose((1, 2, 0)))
    # plt.savefig('fil.png')


