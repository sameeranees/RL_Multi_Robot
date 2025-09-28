#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import os
import numpy as np
import math
import tensorflow as tf
import numpy as np
from collections import deque
import random


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

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
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
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class ActorNetwork:
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		# create actor network
		self.state_input,self.action_output,self.net,self.is_training = self.create_network(state_dim, action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_is_training = self.create_target_network(state_dim,action_dim,self.net)

		# define training rules
		self.create_training_method()
		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		self.load_network()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,-self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim, action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = tf.Variable(tf.random_uniform([layer2_size,action_dim],-0.003, 0.003))
		b3 = tf.Variable(tf.random_uniform([action_dim],-0.003,0.003))

		layer1 = tf.matmul(state_input,W1) + b1
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,W2) + b2
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='batch_norm_2',activation=tf.nn.relu)
		action = tf.matmul(layer2_bn, W3) + b3
		action_linear = self.batch_norm_layer(action[:, None, 0],training_phase=is_training,scope_bn='action_linear',activation=tf.sigmoid)
		action_angular = self.batch_norm_layer(action[:, None, 1],training_phase=is_training,scope_bn='action_angular',activation=tf.tanh)
		# action_linear = tf.sigmoid(action[:, None, 0])
		# action_angular = tf.tanh(action[:, None, 1])
		action = tf.concat([action_linear, action_angular], axis=-1)

		return state_input, action, [W1,b1,W2,b2,W3,b3], is_training

	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		is_training = tf.placeholder(tf.bool)
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.matmul(state_input,target_net[0]) + target_net[1]
		layer1_bn = self.batch_norm_layer(layer1,training_phase=is_training,scope_bn='target_batch_norm_1',activation=tf.nn.relu)
		layer2 = tf.matmul(layer1_bn,target_net[2]) + target_net[3]
		layer2_bn = self.batch_norm_layer(layer2,training_phase=is_training,scope_bn='target_batch_norm_2',activation=tf.nn.relu)
		action = tf.matmul(layer2_bn, target_net[4]) + target_net[5]
		action_linear = self.batch_norm_layer(action[:, None, 0], training_phase=is_training, scope_bn='target_action_linear', activation=tf.sigmoid)
		action_angular = self.batch_norm_layer(action[:, None, 1], training_phase=is_training, scope_bn='target_action_angular', activation=tf.tanh)
		# action_linear = tf.sigmoid(action[:, None, 0])
		# action_angular = tf.tanh(action[:, None, 1])
		action = tf.concat([action_linear, action_angular], axis=-1)

		return state_input, action, target_update, is_training

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.q_gradient_input:q_gradient_batch, self.state_input:state_batch, self.is_training: True})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={self.state_input:state_batch, self.is_training: True})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={self.state_input:[state], self.is_training: False})[0]

	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={self.target_state_input: state_batch, self.target_is_training: True})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def batch_norm_layer(self,x,training_phase,scope_bn,activation=None):
		return tf.cond(training_phase, 
		lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True, updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
		lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True, updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(actor_model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded actoe network")
		else:
			print("Could not find old network weights")

	def save_network(self,time_step):
		print('save actor-network...',time_step)
		self.saver.save(self.sess, actor_model_dir + 'actor-network', global_step=time_step)


critic_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'model', 'critic')


class CriticNetwork:
	def __init__(self,sess,state_dim,action_dim):
		self.time_step = 0
		self.sess = sess
		# create q network
		self.state_input, self.action_input, self.q_value_output,\
		self.net = self.create_q_network(state_dim,action_dim)

		# create target q network (the same structure with q network)
		self.target_state_input, self.target_action_input, self.target_q_value_output,\
		self.target_update = self.create_target_q_network(state_dim,action_dim,self.net)

		self.create_training_method()
		self.sess.run(tf.initialize_all_variables())
			
		self.update_target()
		self.load_network()

	def create_training_method(self):
		self.y_input = tf.placeholder("float",[None,1])
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output))
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output,self.action_input)

	def create_q_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE

		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size+action_dim)
		W2_action = self.variable([action_dim,layer2_size],layer1_size+action_dim)
		b2 = self.variable([layer2_size],layer1_size+action_dim)
		W3 = tf.Variable(tf.random_uniform([layer2_size,1],-0.003,0.003))
		b3 = tf.Variable(tf.random_uniform([1],-0.003,0.003))

		layer1 = tf.nn.relu(tf.matmul(state_input,W1) + b1)
		layer2 = tf.nn.relu(tf.matmul(layer1,W2) + tf.matmul(action_input,W2_action) + b2)
		q_value_output = tf.matmul(layer2,W3) + b3

		return state_input,action_input,q_value_output,[W1,b1,W2,W2_action,b2,W3,b3]

	def create_target_q_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		action_input = tf.placeholder("float",[None,action_dim])

		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + tf.matmul(action_input,target_net[3]) + target_net[4])
		q_value_output = tf.matmul(layer2,target_net[5]) + target_net[6]

		return state_input,action_input,q_value_output,target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,y_batch,state_batch,action_batch):
		self.time_step += 1
		self.sess.run(self.optimizer,feed_dict={self.y_input:y_batch, self.state_input:state_batch, self.action_input:action_batch})

	def gradients(self,state_batch,action_batch):
		return self.sess.run(self.action_gradients,feed_dict={self.state_input:state_batch, self.action_input:action_batch})[0]

	def target_q(self,state_batch,action_batch):
		return self.sess.run(self.target_q_value_output,feed_dict={self.target_state_input:state_batch, self.target_action_input:action_batch})

	def q_value(self,state_batch,action_batch):
		return self.sess.run(self.q_value_output,feed_dict={self.state_input:state_batch, self.action_input:action_batch})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))

	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state(critic_model_dir)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded critic network")
		else:
			print("Could not find old network weights")

	def save_network(self,time_step):
		print('save critic-network...',time_step)
		self.saver.save(self.sess, critic_model_dir + 'critic-network', global_step=time_step)


class DDPG:
    def __init__(self, env, state_dim, action_dim):
        self.name = 'DDPG'
        self.environment = env
        self.time_step = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    def train(self):
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        # for action_dim = 1
        action_batch = np.resize(action_batch, [BATCH_SIZE, self.action_dim])

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch, next_action_batch)
        y_batch = []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch, [BATCH_SIZE, 1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch, action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def action(self, state):
        action = self.actor_network.action(state)

        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)
        if self.replay_buffer.count() == REPLAY_START_SIZE:
            print('\n---------------Start training---------------')
        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.time_step += 1
            self.train()

        if self.time_step % 10000 == 0 and self.time_step > 0:
            self.actor_network.save_network(self.time_step)
            self.critic_network.save_network(self.time_step)

        return self.time_step