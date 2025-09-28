import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import memory as memory


class DeepQ(nn.Module):
    def __init__(self, inputs, outputs, memory_size, discount_factor, learning_rate, learn_start, alpha=0.6, beta_start=0.4, beta_frames=1000000):
        super(DeepQ, self).__init__()
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memory_size)
        self.discount_factor = discount_factor
        self.learn_start = learn_start
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.targetModel = None
        self.optimizer = None
        self.iter_count = 0
        self.criterion = nn.MSELoss()
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_idx = 0

    def initNetworks(self, hiddenLayers, training=True, model_path=None):
        if training:
            self.model = self.createModel(hiddenLayers, activationType="relu").to(self.device)
            self.targetModel = self.createModel(hiddenLayers, activationType="relu").to(self.device)
            self.updateTargetNetwork()  # Ensure target network is in sync initially
        else:
            self.model = torch.load(model_path).to(self.device)

    def createModel(self, hiddenLayers, activationType="relu"):
        layers = []
        input_dim = self.input_size

        for h in hiddenLayers:
            layers.append(nn.Linear(input_dim, h))
            if activationType == "LeakyReLU":
                layers.append(nn.LeakyReLU(0.01))
            else:
                layers.append(nn.ReLU())
            input_dim = h

        layers.append(nn.Linear(input_dim, self.output_size))
        layers.append(nn.Identity())  # Equivalent to Keras's "linear" activation

        model = nn.Sequential(*layers)
        self.optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate, alpha=0.9, eps=1e-06)
        print(model)
        return model

    def updateTargetNetwork(self):
        self.targetModel.load_state_dict(self.model.state_dict())

    def getQValues(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model(state).cpu().numpy()[0]

    def getTargetQValues(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.targetModel(state).cpu().numpy()[0]

    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    def calculateTarget(self, qValuesNewState, reward, isFinal):
        if isFinal:
            return reward
        else:
            return reward + self.discount_factor * self.getMaxQ(qValuesNewState)

    def selectAction(self, qValues, explorationRate):
        if random.random() < explorationRate:
            return np.random.randint(0, self.output_size)
        else:
            return self.getMaxIndex(qValues)

    def addMemory(self, state, action, reward, newState, isFinal):
        td_error = self.calculateTDError(state, action, reward, newState, isFinal)
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def calculateTDError(self, state, action, reward, newState, isFinal):
        current_q_values = self.getQValues(state)
        target_q_values = self.getTargetQValues(newState)
        target_value = self.calculateTarget(target_q_values, reward, isFinal)
        td_error = abs(current_q_values[action] - target_value)
        return float(td_error)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        if self.memory.getCurrentSize() > self.learn_start:
            miniBatch = self.memory.getMiniBatch(miniBatchSize)

            states, actions, rewards, next_states, is_finals = [], [], [], [], []

            for sample in miniBatch:
                states.append(sample['state'])
                actions.append(sample['action'])
                rewards.append(sample['reward'])
                next_states.append(sample['newState'])
                is_finals.append(sample['isFinal'])

            states = torch.FloatTensor(np.array(states, dtype=np.float32)).to(self.device)
            actions = torch.LongTensor(np.array(actions, dtype=np.int64)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32)).to(self.device)
            is_finals = torch.FloatTensor(np.array(is_finals, dtype=np.float32)).to(self.device)

            current_q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            if useTargetNetwork:
                next_q_values = self.targetModel(next_states).max(1)[0]
            else:
                next_q_values = self.model(next_states).max(1)[0]

            target_q_values = rewards + (self.discount_factor * next_q_values * (1 - is_finals))
            td_errors = target_q_values - current_q_values

            loss = torch.mean(td_errors ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_q_value = current_q_values.sum().item()
            q_value_count = current_q_values.numel()
            max_q_value = current_q_values.max().item()
            total_loss = loss.item()
            loss_count = 1

            return total_q_value, q_value_count, total_loss, loss_count, max_q_value


    def saveModel(self, path):
        torch.save(self.model, path)

    def loadWeights(self, path):
        self.model.load_state_dict(torch.load(path).state_dict())

    def printNetwork(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
