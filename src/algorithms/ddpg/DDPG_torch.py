import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import math
from collections import deque
import random
from memory import Memory

# Hyper Parameters
LAYER1_SIZE = 512
LAYER2_SIZE = 512
LEARNING_RATE = 0.0001
TAU = 0.001

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99

actor_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'actor')
critic_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'critic')

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,learning_rate):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layer1 = nn.Linear(state_dim, LAYER1_SIZE)
        # self.batch_norm1 = nn.BatchNorm1d(LAYER1_SIZE)
        self.layer2 = nn.Linear(LAYER1_SIZE, LAYER2_SIZE)
        # self.batch_norm2 = nn.BatchNorm1d(LAYER2_SIZE)
        self.layer3 = nn.Linear(LAYER2_SIZE, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        action_linear = torch.sigmoid(x[:, 0].unsqueeze(1))
        action_angular = torch.tanh(x[:, 1].unsqueeze(1))
        return torch.cat([action_linear, action_angular], dim=-1)

    def train_network(self, q_gradient_input, state_input):
        self.optimizer.zero_grad()
        action_output = self(state_input)
        parameters_gradients = torch.autograd.grad(action_output, self.parameters(), -q_gradient_input, retain_graph=True)
        for param, grad in zip(self.parameters(), parameters_gradients):
            param.grad = grad
        
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,learning_rate):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layer1 = nn.Linear(state_dim, LAYER1_SIZE)
        self.layer2 = nn.Linear(LAYER1_SIZE + action_dim, LAYER2_SIZE)
        self.layer3 = nn.Linear(LAYER2_SIZE, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state_input, action_input):
        x = torch.relu(self.layer1(state_input))
        x = torch.relu(self.layer2(torch.cat([x, action_input], dim=1)))
        return self.layer3(x)

    def train_network(self, y_input, state_input, action_input):
        self.optimizer.zero_grad()
        q_value_output = self(state_input, action_input)
        loss = nn.MSELoss()(q_value_output, y_input)
        loss.backward()
        
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        return loss

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class DDPG:
    def __init__(self, env, state_dim, action_dim,learning_rate=0.0001, batch_size=128, gamma=0.99, tau=0.001, theta=0.15, sigma=0.2):
        self.name = 'DDPG'
        self.environment = env
        self.time_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_network = ActorNetwork(self.state_dim, self.action_dim,learning_rate)
        self.critic_network = CriticNetwork(self.state_dim, self.action_dim,learning_rate)

        self.target_actor_network = ActorNetwork(self.state_dim, self.action_dim,learning_rate)
        self.target_critic_network = CriticNetwork(self.state_dim, self.action_dim,learning_rate)

        self.update_target(self.target_actor_network, self.actor_network, 1.0)
        self.update_target(self.target_critic_network, self.critic_network, 1.0)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        # self.replay_buffer = Memory(REPLAY_BUFFER_SIZE, alpha=0.7)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        # self.noise = OrnsteinUhlenbeckProcess(action_dim, theta=theta, sigma=sigma)


    def update_target(self, target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def train(self):
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = torch.tensor([data[0] for data in minibatch], dtype=torch.float32)
        action_batch = torch.tensor([data[1] for data in minibatch], dtype=torch.float32)
        reward_batch = torch.tensor([data[2] for data in minibatch], dtype=torch.float32).unsqueeze(1)
        next_state_batch = torch.tensor([data[3] for data in minibatch], dtype=torch.float32)
        done_batch = torch.tensor([data[4] for data in minibatch], dtype=torch.float32).unsqueeze(1)

        next_action_batch = self.target_actor_network(next_state_batch)
        q_value_batch = self.target_critic_network(next_state_batch, next_action_batch).detach()
        y_batch = reward_batch + (1 - done_batch) * self.gamma * q_value_batch

        loss=self.critic_network.train_network(y_batch, state_batch, action_batch)

        action_batch_for_gradients = self.actor_network(state_batch)
        q_gradient_batch = torch.autograd.grad(self.critic_network(state_batch, action_batch_for_gradients), action_batch_for_gradients, grad_outputs=torch.ones_like(y_batch), retain_graph=True)[0]

        self.actor_network.train_network(q_gradient_batch, state_batch)

        self.update_target(self.target_actor_network, self.actor_network, self.tau)
        self.update_target(self.target_critic_network, self.critic_network, self.tau)
        return loss

    # def train(self):
    #     state_batch, action_batch, reward_batch, next_state_batch, done_batch, indices, weights = self.replay_buffer.getMiniBatch(self.batch_size)

    #     state_batch = torch.tensor(state_batch, dtype=torch.float32)
    #     action_batch = torch.tensor(action_batch, dtype=torch.float32)
    #     reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1)
    #     next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
    #     done_batch = torch.tensor(done_batch, dtype=torch.float32).unsqueeze(1)
    #     weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)  # Convert weights to tensor

    #     next_action_batch = self.target_actor_network(next_state_batch)
    #     q_value_batch = self.target_critic_network(next_state_batch, next_action_batch).detach()
    #     y_batch = reward_batch + (1 - done_batch) * self.gamma * q_value_batch

    #     # print(y_batch,"BATCH Y")
    #     loss = self.critic_network.train_network(y_batch, state_batch, action_batch)

    #     # print(loss,"LOSS")
    #     action_batch_for_gradients = self.actor_network(state_batch)
    #     q_gradient_batch = torch.autograd.grad(self.critic_network(state_batch, action_batch_for_gradients), action_batch_for_gradients, grad_outputs=torch.ones_like(y_batch), retain_graph=True)[0]

    #     self.actor_network.train_network(q_gradient_batch, state_batch)

    #     self.update_target(self.target_actor_network, self.actor_network, self.tau)
    #     self.update_target(self.target_critic_network, self.critic_network, self.tau)
        
    #     # Update priorities in memory based on TD errors or some other measure
    #     td_errors = y_batch - self.critic_network(state_batch, action_batch).detach()
    #     self.replay_buffer.updatePriorities(indices, td_errors.abs().cpu().numpy() + self.replay_buffer.epsilon)
        
    #     return loss

    

    def action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor_network(state).detach().numpy()[0]
        # noise = self.noise.sample()
        # return action + noise
        return action

    def perceive(self):
        loss = 0
        # print(self.replay_buffer.getCurrentSize(),"SIZE")
        if self.replay_buffer.count() == REPLAY_START_SIZE:
            print('\n---------------Start training---------------')
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            loss = self.train()

        if self.time_step % 10000 == 0 and self.time_step > 0:
            self.save_network(self.time_step)
            
        return loss,self.time_step

    def addMemory(self, state, action, reward, next_state, done):
        # td_error = 0  # Optionally compute TD error here, if you want to prioritize based on TD error
        # self.replay_buffer.addMemory(state, action, reward, next_state, done, td_error)
        self.replay_buffer.add(state,action,reward,next_state,done)

    def save_network(self, time_step):
        torch.save(self.actor_network.state_dict(), actor_model_dir + f'actor-network-{time_step}.pth')
        torch.save(self.critic_network.state_dict(), critic_model_dir + f'critic-network-{time_step}.pth')

    def load_network(self):
        actor_checkpoint = torch.load(actor_model_dir + 'actor-network.pth')
        self.actor_network.load_state_dict(actor_checkpoint)
        self.target_actor_network.load_state_dict(actor_checkpoint)
        critic_checkpoint = torch.load(critic_model_dir + 'critic-network.pth')
        self.critic_network.load_state_dict(critic_checkpoint)
        self.target_critic_network.load_state_dict(critic_checkpoint)
        print("Successfully loaded networks")
