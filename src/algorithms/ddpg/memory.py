import random
import numpy as np
import collections

class Memory:
    def __init__(self, size, alpha=0.6):
        self.size = size
        self.memory = collections.deque(maxlen=size)  # Circular buffer
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha  # How much prioritization is used (0 - no prioritization, 1 - full prioritization)
        self.epsilon = 1e-6  # Small value to ensure non-zero priority

    def getMiniBatch(self, batch_size, beta=0.4):
        if len(self.memory) == self.size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        priorities = priorities ** self.alpha
        sampling_probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probabilities)
        
        # Pre-allocate arrays based on dynamic state and action dimensions
        sample_size = batch_size
        state_dim = self.memory[0]['state'].shape[0]  # Automatically determine state dimension
        action_dim = self.memory[0]['action'].shape[0]  # Automatically determine action dimension
        
        states = np.zeros((sample_size, state_dim), dtype=np.float32)
        actions = np.zeros((sample_size, action_dim), dtype=np.float32)
        rewards = np.zeros(sample_size, dtype=np.float32)
        next_states = np.zeros((sample_size, state_dim), dtype=np.float32)
        is_finals = np.zeros(sample_size, dtype=np.float32)
        weights = np.zeros(sample_size, dtype=np.float32)

        # Extract values from samples
        for i, idx in enumerate(indices):
            sample = self.memory[idx]
            states[i] = sample['state']
            actions[i] = sample['action']
            rewards[i] = sample['reward']
            next_states[i] = sample['newState']
            is_finals[i] = sample['isFinal']

        # Calculate weights
        total = len(self.memory)
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return states, actions, rewards, next_states, is_finals, indices, weights

    def getCurrentSize(self):
        return len(self.memory)

    def getMemory(self, index):
        return self.memory[index]

    def addMemory(self, state, action, reward, newState, isFinal, td_error=None):
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.size:
            self.memory.append(None)

        self.memory[self.position] = {
            'state': np.array(state, dtype=np.float32),  # Ensure state is a numerical numpy array
            'action': np.array(action, dtype=np.float32),  # Ensure action is a numerical numpy array
            'reward': reward,
            'newState': np.array(newState, dtype=np.float32),  # Ensure newState is a numerical numpy array
            'isFinal': isFinal
        }

        priority = max_priority if td_error is None else (abs(td_error) + self.epsilon)
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.size

    def updatePriorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
