import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import memory  # Assuming memory.py has the same functionality
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

class DeepQ(nn.Module):
    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        super(DeepQ, self).__init__()
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.targetModel = None
        self.optimizer = None
        self.iter_count = 0
        self.criterion = nn.MSELoss()
        

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
        self.optimizer = optim.RMSprop(model.parameters(), lr=self.learningRate, alpha=0.9, eps=1e-06)
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
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    def selectAction(self, qValues, explorationRate):
        if random.random() < explorationRate:
            return np.random.randint(0, self.output_size)
        else:
            return self.getMaxIndex(qValues)

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        if self.memory.getCurrentSize() > self.learnStart:
            # start_time = time.time()

            # Step 1: Retrieve mini-batch from memory
            # retrieve_time = time.time()
            miniBatch = self.memory.getMiniBatch(miniBatchSize)
            # print(f"Time taken to retrieve mini-batch: {time.time() - retrieve_time:.4f} seconds")

            # Step 2: Initialize lists to store states and targets
            states, targets = [], []

            # Step 3: Process each sample in the mini-batch
            # sample_processing_time = 0.0
            total_q_value = 0.0
            q_value_count = 0
            max_q_value = float('-inf')
            total_loss = 0.0
            loss_count = 0

            for sample in miniBatch:
                # sample_start_time = time.time()

                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']
                isFinal = sample['isFinal']

                # Compute Q-values for the current state using getQValues
                # qValues_start_time = time.time()
                qValues = torch.FloatTensor(self.getQValues(state)).to(self.device)
                # print(f"Time taken to compute Q-values: {time.time() - qValues_start_time:.4f} seconds")

                # Compute Q-values for the next state using getTargetQValues or getQValues
                # qValuesNewState_start_time = time.time()
                if useTargetNetwork:
                    qValuesNewState = torch.FloatTensor(self.getTargetQValues(newState)).to(self.device)
                else:
                    qValuesNewState = torch.FloatTensor(self.getQValues(newState)).to(self.device)
                # print(f"Time taken to compute Q-values for new state: {time.time() - qValuesNewState_start_time:.4f} seconds")

                total_q_value += qValuesNewState.sum().item()
                q_value_count += qValuesNewState.numel()
                max_q_value = max(max_q_value, qValuesNewState.max().item())
                
                # Calculate the target Q-value
                # calculate_target_start_time = time.time()
                targetValue = self.calculateTarget(qValuesNewState.cpu().numpy(), reward, isFinal)
                # print(f"Time taken to calculate target Q-value: {time.time() - calculate_target_start_time:.4f} seconds")

                # Update the Q-value for the chosen action
                qValues[action] = targetValue

                states.append(state)
                targets.append(qValues.detach().cpu().numpy())

                # av_Q += torch.mean(qValuesNewState)
                # max_Q = max(max_Q, torch.max(qValuesNewState))
                # sample_processing_time += time.time() - sample_start_time

            # print(f"Time taken for sample processing: {sample_processing_time:.4f} seconds")

            # Step 4: Convert lists to tensors
            # conversion_time = time.time()
            states = torch.FloatTensor(np.array(states)).to(self.device)
            targets = torch.FloatTensor(np.array(targets)).to(self.device)
            # print(f"Time taken for conversion to tensors: {time.time() - conversion_time:.4f} seconds")
            
            # Step 5: Create dataset and data loader
            # dataset_creation_time = time.time()
            dataset = TensorDataset(states, targets)
            dataloader = DataLoader(dataset, batch_size=miniBatchSize, shuffle=True)
            # print(f"Time taken to create dataset and dataloader: {time.time() - dataset_creation_time:.4f} seconds")

            # Step 6: Training loop
            # training_time = 0.0
            self.model.train()
            for X_batch, Y_batch in dataloader:
                # training_start_time = time.time()
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, Y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                loss_count += 1
                # training_time += time.time() - training_start_time
            self.iter_count += 1
            # print(f"Time taken for training loop: {training_time:.4f} seconds")

            # if q_value_count > 0:
            #     avg_q_value = total_q_value / q_value_count
            #     self.writer.add_scalar("Av. Q", avg_q_value / miniBatchSize, self.iter_count)
            
            # self.writer.add_scalar("Max. Q", max_q_value, self.iter_count)
            
            # if loss_count > 0:
            #     avg_loss = total_loss / loss_count
            #     self.writer.add_scalar("loss", avg_loss / miniBatchSize, self.iter_count)
            
            # total_time = time.time() - start_time
            # print(f"Total learnOnMiniBatch time: {total_time:.4f} seconds")

            return total_q_value,q_value_count,total_loss,loss_count,max_q_value
        
    def saveModel(self, path):
        torch.save(self.model, path)

    def loadWeights(self, path):
        self.model.load_state_dict(torch.load(path).state_dict())

    def printNetwork(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
