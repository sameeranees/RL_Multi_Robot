import random

import numpy as np
from keras import models
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
import time
import memory


class DeepQ:
    """
    DQN abstraction.
    As a quick reminder:
        traditional Q-learning:
            Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
        DQN:
            target = reward(s,a) + gamma * max(Q(s')
    """

    def __init__(self, inputs, outputs, memorySize, discountFactor, learningRate, learnStart):
        """
        Parameters:
            - inputs: input size
            - outputs: output size
            - memorySize: size of the memory that will store each state
            - discountFactor: the discount factor (gamma)
            - learningRate: learning rate
            - learnStart: steps to happen before for learning. Set to 128
        """
        self.input_size = inputs
        self.output_size = outputs
        self.memory = memory.Memory(memorySize)
        self.discountFactor = discountFactor
        self.learnStart = learnStart
        self.learningRate = learningRate

    def initNetworks(self, hiddenLayers, training=True, model_path=None):
        # Normal: For simulation training
        if training:
            model = self.createModel(hiddenLayers, "relu", self.learningRate)
            self.model = model

            targetModel = self.createModel(hiddenLayers, "relu", self.learningRate)
            self.targetModel = targetModel

        # For simulation testing
        else:
            self.model = models.load_model(model_path)

    def createRegularizedModel(self, hiddenLayers, activationType, learningRate):
        bias = True
        dropout = 0
        regularizationFactor = 0.01
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(
                Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        else:
            if regularizationFactor > 0:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform',
                                W_regularizer=l2(regularizationFactor), bias=bias))
            else:
                model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform',
                                bias=bias))

            if activationType == "LeakyReLU":
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                if regularizationFactor > 0:
                    model.add(
                        Dense(layerSize, kernel_initializer='lecun_uniform', W_regularizer=l2(regularizationFactor),
                              bias=bias))
                else:
                    model.add(Dense(layerSize, kernel_initializer='lecun_uniform', bias=bias))
                if activationType == "LeakyReLU":
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation(activationType))
                if dropout > 0:
                    model.add(Dropout(dropout))
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform', bias=bias))
            model.add(Activation("linear"))
        optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()
        return model

    def createModel(self, hiddenLayers, activationType, learningRate):
        model = Sequential()
        if len(hiddenLayers) == 0:
            model.add(Dense(self.output_size, input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))
        else:
            model.add(Dense(hiddenLayers[0], input_shape=(self.input_size,), kernel_initializer='lecun_uniform'))
            if activationType == "LeakyReLU":
                model.add(LeakyReLU(alpha=0.01))
            else:
                model.add(Activation(activationType))

            for index in range(1, len(hiddenLayers)):
                layerSize = hiddenLayers[index]
                model.add(Dense(layerSize, kernel_initializer='lecun_uniform'))
                if activationType == "LeakyReLU":
                    model.add(LeakyReLU(alpha=0.01))
                else:
                    model.add(Activation(activationType))
            model.add(Dense(self.output_size, kernel_initializer='lecun_uniform'))
            model.add(Activation("linear"))

        optimizer = optimizers.RMSprop(learning_rate=learningRate, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=optimizer)
        model.summary()

        return model

    def printNetwork(self):
        i = 0
        for layer in self.model.layers:
            weights = layer.get_weights()
            print("layer ", i, ": ", weights)
            i += 1

    def backupNetwork(self, model, backup):
        weightMatrix = []
        for layer in model.layers:
            weights = layer.get_weights()
            weightMatrix.append(weights)
        i = 0
        for layer in backup.layers:
            weights = weightMatrix[i]
            layer.set_weights(weights)
            i += 1

    def updateTargetNetwork(self):
        self.backupNetwork(self.model, self.targetModel)

    # predict Q values for all the actions
    def getQValues(self, states):
        # Ensure states is a 2D array where each row is a state
        if len(states.shape) == 1:
            # If states is a single state, reshape it to (1, num_features)
            states = states.reshape(1, -1)
        predicted = self.model.predict(states)
        return predicted

    def getTargetQValues(self, states):
        # Ensure states is a 2D array where each row is a state
        if len(states.shape) == 1:
            # If states is a single state, reshape it to (1, num_features)
            states = states.reshape(1, -1)
        predicted = self.targetModel.predict(states)
        return predicted


    def getMaxQ(self, qValues):
        return np.max(qValues)

    def getMaxIndex(self, qValues):
        return np.argmax(qValues)

    # calculate the target function
    def calculateTarget(self, qValuesNewState, reward, isFinal):
        """
            target = reward(s,a) + gamma * max(Q(s')
        """
        if isFinal:
            return reward
        else:
            return reward + self.discountFactor * self.getMaxQ(qValuesNewState)

    # select the action with the highest Q value
    def selectAction(self, qValues, explorationRate):
        rand = random.random()
        if rand < explorationRate:
            action = np.random.randint(0, self.output_size)
        else:
            action = self.getMaxIndex(qValues)
        return action

    def selectActionByProbability(self, qValues, bias):
        qValueSum = 0
        shiftBy = 0
        for value in qValues:
            if value + shiftBy < 0:
                shiftBy = - (value + shiftBy)
        shiftBy += 1e-06

        for value in qValues:
            qValueSum += (value + shiftBy) ** bias

        probabilitySum = 0
        qValueProbabilities = []

        for value in qValues:
            probability = ((value + shiftBy) ** bias) / float(qValueSum)
            qValueProbabilities.append(probability + probabilitySum)
            probabilitySum += probability
        qValueProbabilities[len(qValueProbabilities) - 1] = 1

        rand = random.random()
        i = 0
        for value in qValueProbabilities:
            if rand <= value:
                return i
            i += 1

    def addMemory(self, state, action, reward, newState, isFinal):
        self.memory.addMemory(state, action, reward, newState, isFinal)

    def learnOnLastState(self):
        if self.memory.getCurrentSize() >= 1:
            return self.memory.getMemory(self.memory.getCurrentSize() - 1)

    def learnOnMiniBatch(self, miniBatchSize, useTargetNetwork=True):
        if self.memory.getCurrentSize() > self.learnStart:
            miniBatch = self.memory.getMiniBatch(miniBatchSize)

            states = np.array([sample['state'] for sample in miniBatch])
            actions = np.array([sample['action'] for sample in miniBatch])
            rewards = np.array([sample['reward'] for sample in miniBatch])
            newStates = np.array([sample['newState'] for sample in miniBatch])
            isFinals = np.array([sample['isFinal'] for sample in miniBatch])

            # Ensure states and newStates are 2D arrays
            states = np.array(states)
            newStates = np.array(newStates)

            # Get Q-values
            qValues = self.getQValues(states)
            qValuesNewStates = self.getTargetQValues(newStates) if useTargetNetwork else self.getQValues(newStates)

            # Calculate targets
            targets = np.array([
                self.calculateTarget(qValuesNewState, reward, isFinal)
                for qValuesNewState, reward, isFinal in zip(qValuesNewStates, rewards, isFinals)
            ])

            # Create Y_batch
            Y_batch = np.copy(qValues)
            for i in range(len(actions)):
                Y_batch[i, actions[i]] = targets[i]
                if isFinals[i]:
                    Y_batch[i] = np.full(self.output_size, rewards[i])

            if np.any(isFinals):
                final_rewards = np.array([rewards[i] * np.ones(self.output_size) for i in range(len(isFinals)) if isFinals[i]])
            else:
                final_rewards = np.empty((0, self.output_size))

            X_batch = np.vstack((states, newStates[isFinals] if np.any(isFinals) else np.empty((0, self.input_size))))
            Y_batch = np.vstack((Y_batch, final_rewards))

            # Evaluate and fit the model
            total_loss = self.model.evaluate(X_batch, Y_batch, verbose=0)
            self.model.fit(X_batch, Y_batch, batch_size=len(miniBatch), epochs=1, verbose=0)

            total_q_value = np.sum(qValues)
            max_q_value = np.max(qValues)
            q_value_count = qValues.size

            return total_q_value, q_value_count, total_loss, len(miniBatch), max_q_value






    def saveModel(self, path):
        self.model.save(path)

    def loadWeights(self, path):
        self.model.set_weights(load_model(path).get_weights())
