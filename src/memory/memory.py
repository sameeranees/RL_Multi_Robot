import random
import numpy as np
import collections

class Memory:
    """
    This class provides an abstraction to store the [s, a, r, a'] elements of each iteration.
    The information is stored in a deque that simulates a circular buffer.
    """

    def __init__(self, size):
        self.size = size
        self.memory = collections.deque(maxlen=size)  # Circular buffer
        self.position = 0

    def getMiniBatch(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)

    def getCurrentSize(self):
        print(len(self.memory))
        return len(self.memory)

    def getMemory(self, index):
        return self.memory[index]

    def addMemory(self, state, action, reward, newState, isFinal):
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = {
            'state': state,
            'action': action,
            'reward': reward,
            'newState': newState,
            'isFinal': isFinal
        }
        self.position = (self.position + 1) % self.size
