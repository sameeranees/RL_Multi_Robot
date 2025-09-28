import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

# from omni.isaac.python_app import OmniKitHelper

import omni
# from omni.isaac.dynamic_control import _dynamic_control
# from omni.isaac.utils.scripts.nucleus_utils import find_nucleus_server
import carb
# from omni.isaac.core.prims import RigidPrimView
from replay_buffer import ReplayBuffer
# import omni.physx
# import omni.isaac.core.utils.bounds as bounds_utils
# from omni.physx import get_physx_scene_query_interface

# from pxr import UsdGeom, Gf, Vt

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
import threading

import math
import random
# import omni.kit.commands
from omni.isaac.kit import SimulationApp
import point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from collections import deque
from deepq import DeepQ
from memory_3 import Memory

# from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading

GOAL_REACHED_DIST = 66
COLLISION_DIST = 0.28
TIME_DELTA = 0.3

simulation_app = SimulationApp({"headless": False})
simulation_app.update()

import asyncio
from omni import usd
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.prims import RigidPrimView
# from replay_buffer import ReplayBuffer
import omni.physx
import omni.isaac.core.utils.bounds as bounds_utils
from omni.physx import get_physx_scene_query_interface
import omni.kit.commands
from omni.isaac.core import SimulationContext, World
from omni.isaac.core.utils import extensions
from omni.isaac.core.utils.nucleus import get_assets_root_path, get_assets_root_path_async,find_nucleus_server
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading, update_stage, open_stage
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.render_product import create_hydra_texture
import omni.replicator.core as rep
import omni.graph.core as og
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.prims import XFormPrimView
# Locate Isaac Sim assets folder to load sample
# from omni.isaac.core.utils.nucleus import get_assets_root_path, is_file
from pxr import UsdGeom, Gf, Vt

extensions.enable_extension("omni.isaac.ros2_bridge")
extensions.enable_extension("omni.isaac.sensor")

# extensions.enable_extension("omni.sensors.nv.lidar")
# asset_path=find_nucleus_server("/home/isaac-sim/.local/share/ov/pkg/isaac_sim-2023.1.1")
# print(asset_path)
# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
print(device)
last_odom = None
last_odom2 = None
environment_dim = 20
environment_dim_2 = 360
point_cloud_data = np.ones(environment_dim) * 10
scanData = np.ones(environment_dim_2) * 10 
initial_point_cloud_data = point_cloud_data
initial_scan_data = scanData
flag_pc_data=False
flag_ls_data=False

old_distance = np.inf
old_theta = np.inf

targets_changed=0

def change_laser_distance(scans):
    new_scans=[]
    i=1
    temp_scan=[]
    scan=scans[0]
    while i <= len(scan):
        temp_scan.append(scan[i-1])
        if(i%7==0) or (i==360):
            new_scans.append(min(temp_scan))
            temp_scan=[]
        i+=1
    return new_scans

def evaluate(network, epoch, eval_episodes=10,velodyne_subscriber=None):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        env.get_logger().info(f"evaluating episode {_}")
        count = 0
        checkpoint_path = f'checkpoint_epoch_{_+1}'
        state = env.reset(velodyne_subscriber)
        done = False
        network.load(checkpoint_path,directory="./checkpoints")
        while not done and count < 1001:
            action = network.get_action(np.array(state))
            # env.get_logger().info(f"action : {action}")
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _target,_distance,_theta = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    env.get_logger().info("..............................................")
    env.get_logger().info(
        "Average Reward over %i Evaluation Episodes, Epoch %i: avg_reward %f, avg_col %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    env.get_logger().info("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# td3 network
class td3(object):
    def __init__(self, state_dim, action_dim, max_action,learningRate):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=learningRate)

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learningRate)

        self.max_action = max_action
        self.writer = SummaryWriter()
        # os.path.dirname(os.path.realpath(__file__)) + "/runs"
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        cumulated_reward,
        distance,
        theta,
        successes,
        failures,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            # (
            #     batch_states,
            #     batch_actions,
            #     batch_rewards,
            #     batch_dones,
            #     batch_next_states,
            # ) = replay_buffer.get_batch(batch_size)
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights = replay_buffer.getMiniBatch(batch_size)

            batch_states = torch.tensor(batch_states, dtype=torch.float32).to(device)
            batch_actions = torch.tensor(batch_actions, dtype=torch.float32).to(device)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1).to(device)
            batch_next_states = torch.tensor(batch_next_states, dtype=torch.float32).to(device)
            batch_dones = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1).to(device)
            weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)  # Convert weights to tensor

            # batch_states = torch.tensor([data[0] for data in minibatch], dtype=torch.float32)
            # batch_actions = torch.tensor([data[1] for data in minibatch], dtype=torch.float32)
            # batch_rewards = torch.tensor([data[2] for data in minibatch], dtype=torch.float32).unsqueeze(1)
            # batch_next_states = torch.tensor([data[3] for data in minibatch], dtype=torch.float32)
            # batch_dones = torch.tensor([data[4] for data in minibatch], dtype=torch.float32).unsqueeze(1)
            
            state = batch_states
            next_state = batch_next_states
            action = batch_actions
            reward = batch_rewards
            done = batch_dones

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            # Calculate the TD error and loss between the current Q value and the target Q value
            td_error1 = target_Q - current_Q1
            td_error2 = target_Q - current_Q2
            loss = (F.mse_loss(current_Q1, target_Q, reduction='none') + 
                    F.mse_loss(current_Q2, target_Q, reduction='none'))
            loss = (loss * weights).mean()

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_optimizer.step()

            td_errors = (td_error1 + td_error2) / 2
            td_errors = td_errors.detach().cpu().numpy() + replay_buffer.epsilon  # Add small epsilon to avoid zero priority
            replay_buffer.updatePriorities(indices, td_errors)

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()

                # Use deepQ = deepq.DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        # deepQ.initNetworks(network_structure)soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        env.get_logger().info(f"writing new results for a tensorboard")
        env.get_logger().info(f"loss, Av.Q, Max.Q, iterations : {av_loss / iterations}, {av_Q / iterations}, {max_Q}, {self.iter_count}")
        self.writer.add_scalar("loss", av_loss.item() / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q.item() / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q.item(), self.iter_count)
        if(successes > 0) or (failures > 0):
            self.writer.add_scalar("Cumulative Reward", cumulated_reward, self.iter_count)
            self.writer.add_scalar("Distance",distance,self.iter_count)
            self.writer.add_scalar("Theta",theta,self.iter_count)
            self.writer.add_scalar("Sucesses",successes,self.iter_count)
            self.writer.add_scalar("Failures",failures,self.iter_count)
        self.writer.flush()

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )

# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class Env(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0
        self.skid_x = 1.14435
        self.skid_y = 312.0143
        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        self.odometry_history = deque(maxlen=50)

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # Set up the ROS publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        

        # self.unpause = self.create_client(Empty, "/unpause_physics")
        # self.pause = self.create_client(Empty, "/pause_physics")
        # self.reset_proxy = self.create_client(Empty, "/reset_world")
        # self.req = Empty.Request
        # assets_root_path = get_assets_root_path()
        # assets_root_path = get_assets_root_path()
        # print(assets_root_path)
        self.asset_path = 'omniverse://localhost/Projects/isaac-env/ISAAC MATERIAL/Collected_Thesis4/Thesis1.usd'
        # self.asset_path = 'omniverse://localhost/Projects/isaac-env/ISAAC MATERIAL/Collected Thesis 2/Collected_Poster_scene/Poster_scene.usd'
        # omni.usd.get_context().open_stage(self.asset_path)
        # simulation_app.update()
        # while is_stage_loading():
        #     simulation_app.update()
        self.simulation_context = SimulationContext(stage_units_in_meters=1.0)
        self.simulation_context.set_simulation_dt(0.001,1000.0)
        omni.usd.get_context().open_stage(self.asset_path)
        # self.simulation_context.initialize_physics()
        # self.simulation_context=omni.usd.get_context()
        # self.simulation_context = World(stage_units_in_meters=1.0)
        # self.simulation_context.scene.add_default_ground_plane()
        self.simulation_context.get_physics_context()._create_new_physics_scene("/physicsScene")
        self.simulation_context.initialize_physics()
        # add_reference_to_stage(self.asset_path, "/World/Thesis")
        # add_reference_to_stage(self.asset_path, prim_path="/World/Thesis/arena_2")
        # add_reference_to_stage(self.asset_path, prim_path="/World/Thesis/SensorGraph")

        simulation_app.update()
        while is_stage_loading():
            simulation_app.update()
        # _, sensor = omni.kit.commands.execute(
        #     "IsaacSensorCreateRtxLidar",
        #     path="/sensor",
        #     parent=None,
        #     config="Example_Rotary",
        #     translation=(0, 0, 1.0),
        #     orientation=Gf.Quatd(0.5, 0.5, -0.5, -0.5),
        # )

        # render_product_path = rep.create.render_product(sensor.GetPath(),[1,1])
        # writer = rep.writers.get("RtxLidar"+"ROS2PublishLaserScan")
        # writer.initialize(topicName="scan_2",frameId="sim_lidar")
        # writer.attach(render_product_path)
        

        # Trigger the Action Graph
        # og.Controller.set(og.Controller.attribute("/ActionGraph/OnImpulseEvent.state:enableImpulse"), True)
        # og.Controller.set(og.Controller.attribute("/SensorGraph/OnImpulseEvent.state:enableImpulse"), True)
        # writer2 = rep.writers.get("RtxLidar"+"ROS2PublishPointCloud")
        # writer2.initialize(topicName="point_cloud",frameId="sim_lidar")
        # writer2.attach(render_product_path)

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

    # Perform an action and read a new state
    def step(self, action):
        global point_cloud_data
        global scanData
        target = False
        done=False
        max_linear = 0.5
        max_angular = 18
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0]) * max_linear
        vel_cmd.angular.z = float(action[1]) * max_angular
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        # while not self.unpause.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        

        # try:
        #     self.unpause.call_async(Empty.Request())
        # except:
        #     print("/unpause_physics service call failed")
        
        # need to initialize physics getting any articulation..etc
        
        # self.simulation_context.play()
        # propagate state for TIME_DELTA seconds
        # time.sleep(TIME_DELTA)
        # print(last_odom.pose.pose.position,"TRANSLATION")
        start_time = time.time()
        while not(time.time() - start_time >=0.2):# and (action[1] == 0):
            # print(time.time() - start_time, "TIME IT TOOK")
            self.simulation_context.step(render=True)
        # print(last_odom.pose.pose.position,"TRANSLATION")
        # start_time = time.time()
        # while not(time.time() - start_time >=0.3) and (action[1] !=0):
        #     self.simulation_context.step(render=True)

        # start_time = time.perf_counter()
        # while (time.perf_counter() - start_time) < TIME_DELTA:
        #     self.simulation_context.step(render=True)
        # for i in range(5):
        #     self.simulation_context.step(render=True)
        # while not self.pause.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        # self.simulation_context.pause()

        # simulation_app.close() PUT THIS SOMEHWERE LATER
        
        # try:

        #     pass
        #     self.pause.call_async(Empty.Request())
        # except (rclpy.ServiceException) as e:
        #     print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(point_cloud_data)
        v_state = []
        v_state[:] = point_cloud_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        # print(self.skid_x,self.skid_y)
        self.odom_x = last_odom.pose.pose.position.x * 100 + self.skid_x
        self.odom_y = last_odom.pose.pose.position.y * 100 + self.skid_y
        # print(self.odom_x,"ODOM X",self.odom_y,"ODOM Y")
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # print(self.odom_x,self.odom_y,'ODOMETRY')
        # print(distance,"THE DISTANCE TO TARGET")
        # print('ODOM',self.odom_x,self.odom_y,'; ', 'GOAL',self.goal_x,self.goal_y,self.goal_z)
        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target = True
            done = True

        self.odometry_history.append([self.odom_x,self.odom_y])
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        repetitive_behavior = self.detect_repetitive_behavior(self.odometry_history)
        reward = self.get_reward(target, collision, vel_cmd.linear.x,vel_cmd.angular.z, min_laser,repetitive_behavior,distance,theta)
        print("The reward for step is:", reward,"THE DISTANCE TO TARGET",distance,"THE ANGLE TO THE TARGET",theta)
        return state, reward, done, target, distance,theta

    def reset(self,change_target,change_robot=False):

        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        # while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset : service not available, waiting again...')

        # try:
        #     self.reset_proxy.call_async(Empty.Request())
        # except rclpy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed")
        # point_cloud_data = np.ones(environment_dim) * 10
        if self.simulation_context.is_stopped()==False:
            self.simulation_context.stop()
        
        self.skid_x = 1.14435
        self.skid_y = 312.0143

        if change_robot:
            self.change_robot_location()
        self.simulation_context.reset()

        global point_cloud_data
        global initial_point_cloud_data
        point_cloud_data = initial_point_cloud_data
        
        global scanData
        global initial_scan_data
        scanData = initial_scan_data

        global old_distance
        global old_theta
        global targets_changed

        old_distance = np.inf
        old_theta = np.inf
        
        # velodyne_subscriber.executor.spin_once()
        self.odometry_history.clear()

        self.simulation_context.initialize_physics()
        # print(point_cloud_data,'Reached here')
        simulation_app.update()

        while is_stage_loading():
            simulation_app.update()

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        # object_state = self.set_self_state

        # x = 0
        # y = 0
        # position_ok = False
        # while not position_ok:
        #     x = np.random.uniform(-4.5, 4.5)
        #     y = np.random.uniform(-4.5, 4.5)
        #     position_ok = check_pos(x, y)
        # object_state.pose.position.x = x
        # object_state.pose.position.y = y
        # # object_state.pose.position.z = 0.
        # object_state.pose.orientation.x = quaternion.x
        # object_state.pose.orientation.y = quaternion.y
        # object_state.pose.orientation.z = quaternion.z
        # object_state.pose.orientation.w = quaternion.w
        # self.set_state.publish(object_state)

        # self.odom_x = object_state.pose.position.x * 100 + 1.14435
        # self.odom_y = object_state.pose.position.y * 100 + 312.0143

        camera_view = ArticulationView(prim_paths_expr="/root/Camera/Camera")
        # camera_view.disable_rigid_body_physics()
        camera_view.set_local_poses(translations=[np.array([1005.75592,51.10774,1217.84581])],orientations=[np.array([0.93045,0.0307,0.33211,0.15171])])
        positions_target,orientations_target=camera_view.get_local_poses()
        # print(positions_target,orientations_target)
        
        # time.sleep(10)
        if(change_target):
            targets_changed += 1

        self.change_goal(targets_changed)
        
        
        self.odom_x = 0.0 + self.skid_x
        self.odom_y = 0.0 + self.skid_y
        # print(self.goal_x,self.goal_y,self.goal_z,"THE GOAL")
        # print(self.odom_x,self.odom_y,"THE ROBOT")
        # time.sleep(5)
        # randomly scatter boxes in the environment
        # self.random_box()
        self.publish_markers([0.0, 0.0])

        # while not self.unpause.wait_for_service(timeout_sec=1.0):
        #     self.node.get_logger().info('service not available, waiting again...')

        # try:
        #     self.unpause.call_async(Empty.Request())
        # except:
        #     print("/gazebo/unpause_physics service call failed")
        # self.simulation_context.initialize_physics()
        self.simulation_context.play()
        # print('SOMETHING WRONG AFTER')
        # time.sleep(TIME_DELTA)

        # while not self.pause.wait_for_service(timeout_sec=1.0):
        #     self.node.get_logger().info('service not available, waiting again...')
        self.simulation_context.step(render=True)
        # start_time = time.perf_counter()
        # while (time.perf_counter() - start_time) < TIME_DELTA:
        #     self.simulation_context.step(render=True)
        # for i in range(5):
        #     self.simulation_context.step(render=True)

        # self.simulation_context.pause()
        # try:
        #     self.pause.call_async(Empty.Request())
        # except:
        #     print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = point_cloud_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        # print(state)
        return state

    def close(self):
        if self.simulation_context.is_stopped()==False:
            self.simulation_context.stop()
        simulation_app.close()

    def check_pos(self,rand_x,rand_z,rigid_prim_view_target):
        # rigid_prim_view_cube = RigidPrimView(prim_paths_expr="/root/flw_Halle/Cube_[0-06]")
        # rigid_prim_view_cone = RigidPrimView(prim_paths_expr="/root/flw_Halle/Cone_[0-05]")
        # rigid_prim_view_sphere = RigidPrimView(prim_paths_expr="/root/flw_Halle/Sphere_[0-04]")
        rigid_prim_view_target.set_local_poses(translations=[np.array([rand_x,self.goal_y,rand_z])])
        positions_target,orientations_target=rigid_prim_view_target.get_local_poses()
        print(positions_target)
        # positions_cube,orientations_cube=rigid_prim_view_cube.get_world_poses()
        # positions_cone,orientations_cone=rigid_prim_view_cone.get_world_poses()
        # positions_sphere,orientations_sphere=rigid_prim_view_sphere.get_world_poses()
        cache = bounds_utils.create_bbox_cache()
        centroid, axes, half_extent = bounds_utils.compute_obb(cache, prim_path="/root/arena_2/Target")
        full_extent=half_extent*2
        print(orientations_target)
        extent = carb.Float3(full_extent[0],full_extent[1],full_extent[2])
        origin = carb.Float3(positions_target[0][2],positions_target[0][1],positions_target[0][0])
        rotation = carb.Float4(0.0, 0.0, 1.0, 0.0)
        # physX query to detect number of hits for a cubic region
        numHits = get_physx_scene_query_interface().overlap_box(extent, origin, rotation, self.report_hit, False)
        if numHits>0:
            print("The Target HITS")
            return False
        return True


    def report_hit(self, hit):
        return True

    def change_goal(self,targets_changed):
        rigid_prim_view = ArticulationView(prim_paths_expr="/root/arena_2/Target")
        goal_ok = False
        self.goal_y= 3.2

        if(targets_changed == 0):
            cords = [[502.0,0.0]]
        
        elif(targets_changed == 1):
            cords = [[-460,0.0]]

        elif(targets_changed >= 2):
            cords = [[502.0,0.0],[-460,0.0]]

        # cords = [[502.0,0.0]]
        # while not goal_ok:
        #     choice=
        #     rand_z= random.uniform(-383.0,405.0)
        #     rand_x= random.uniform(-942.0,892.0)
        #     goal_ok= self.check_pos(rand_x,rand_z,rigid_prim_view)
        choice = random.choice(cords)
        self.goal_x = choice[0]
        self.goal_z = choice[1]
        rigid_prim_view.set_local_poses(translations=[np.array([self.goal_x,self.goal_y,self.goal_z])])
        rigid_prim_view = ArticulationView(prim_paths_expr="/root/arena_2/Target")
        pos,rots=rigid_prim_view.get_world_poses()
        self.goal_x=float(pos[0][0])
        self.goal_y=float(pos[0][1])
        self.goal_z=float(pos[0][2])

        # time.sleep(5)

    def change_robot_location(self):
        rigid_prim_view = XFormPrimView("/root/robomaster")
        rigid_prim_view.set_default_state()
        rigid_prim_view.post_reset()
        
        cords=[[1.14435,312.0143],[1.14435,166.68101],[1.14435,-271.4367],[1.14435,-157.6057],[1.14435,-4.43865]]
        choice = random.choice(cords)

        rigid_prim_view = XFormPrim("/root/robomaster")
        rigid_prim_view.set_local_pose(translation=np.array([choice[0],choice[1],0.0]))

        # rigid_prim_view = XFormPrim("/root/robomaster/base_footprint")
        # rigid_prim_view.set_local_poses(translations=[np.array([choice[0],0.78,choice[1]])])
        # translation,orientation=rigid_prim_view.get_local_pose()
        
        print("CHOICES",choice[0],choice[1])
        
        self.skid_x = choice[0]
        self.skid_y = choice[1]
        # return skid_x, skid_y



    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = float(abs(action[0]))
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = float(abs(action[1]))
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        # print(min(laser_data),"COLLISIONS")
        if min_laser < COLLISION_DIST:
            # print(laser_data)
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, lin_vel, ang_vel, min_laser, repetitive_behavior, distance, theta):
        global old_distance
        global old_theta

        calc_reward = 0

        # Reward for reaching the target
        if target:
            env.get_logger().info("reward 200 for reaching target")
            return 200.0

        # Penalty for collision
        if collision:
            env.get_logger().info("penalty -200 for collision")
            return -200.0

        # Small penalty for non-zero actions to encourage efficient movement
        if lin_vel > 0:
            calc_reward += -5
        if ang_vel != 0:
            calc_reward += -1

        distance_reward = (old_distance - distance)
        if(abs(distance_reward) < 100):
            calc_reward += distance_reward
            if(distance<old_distance):
                calc_reward+= (1/distance)*100
        # if(distance < old_distance):
        #     calc_reward += 0.5 

        # Reward for reducing the angle to the target
        if(old_theta<0 and theta<0) or (old_theta>0 and theta>0):
            angle_reward = (abs(old_theta) - abs(theta)) * 4
            if(abs(angle_reward) < 100):
                calc_reward += angle_reward 
            if(abs(theta)<0.3):
                calc_reward+= (1/theta) *0.10
        # if(theta < old_theta):
        #     calc_reward += 0.1


        # Update old values
        old_distance = distance
        old_theta = theta

        return calc_reward/10 # Scale rewards
    
    @staticmethod
    def detect_repetitive_behavior(history, position_radius=0.25, orientation_threshold=0.1, threshold=0.8,min_history_length=25):
        positions = np.array([[pose[0], pose[1]] for pose in history])
        if len(history) < min_history_length:
            return False
        
        # Calculate distances between the last position and all other positions
        distances = np.linalg.norm(positions - positions[-1], axis=1)
        close_positions = distances < position_radius
        
        # Optionally, check orientations if necessary
        # orientations = np.array([[pose[2], pose[3], pose[4], pose[5]] for pose in history])
        # orientation_differences = np.linalg.norm(orientations - orientations[-1], axis=1)
        # close_orientations = orientation_differences < orientation_threshold

        # Check for repetitive behavior based on positions (and optionally orientations)
        return np.mean(close_positions) > threshold
    

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data
        # print(last_odom)

class Odom_subscriber2(Node):

    def __init__(self):
        super().__init__('odom_subscriber2')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom2',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom2
        last_odom2 = od_data
        # print('LASZ ODOM',last_odom2)



class Odom_publisher(Node):

    def __init__(self):
        super().__init__('odom_pub')
        self.odom_pub = Node('odo_pub').create_publisher(Odometry,"/odom2",10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # print(float(positions[0][0]))
        self.set_self_state = Odometry()
        # # self.set_self_state.model_name = "r1"
        positions,rotations=robomaster.get_world_poses()
        print(positions,'POSITION HERE')
        self.set_self_state.pose.pose.position.x = float(positions[0][0])
        self.set_self_state.pose.pose.position.y = float(positions[0][1])
        self.set_self_state.pose.pose.position.z = float(positions[0][2])
        self.set_self_state.pose.pose.orientation.x = float(rotations[0][1])
        self.set_self_state.pose.pose.orientation.y = float(rotations[0][2])
        self.set_self_state.pose.pose.orientation.z = float(rotations[0][3])
        self.set_self_state.pose.pose.orientation.w = float(rotations[0][0])
        # print(self.set_self_state)
        self.odom_pub.publish(self.set_self_state)

class Laser_subscriber(Node):

    def __init__(self):
        super().__init__('scan_subscriber')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData
        global initial_scan_data#
        global flag_ls_data
        scanData = od_data.ranges
        if flag_ls_data== False:
            initial_scan_data = scanData
            flag_ls_data = True
        # print(last_odom)

class Point_cloud_subscriber(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data
        global initial_point_cloud_data
        global flag_pc_data
        
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_2,'PCD')

            # Calculate dot product and magnitudes
            dots = data[:, 0] * 1 + data[:, 1] * 0
            mag1 = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
            mag2 = np.sqrt(1 ** 2 + 0 ** 2)

            # Handle division by zero case
            beta = np.arccos(np.divide(dots, mag1 * mag2, out=np.zeros_like(dots), where=(mag1 * mag2) != 0)) * np.sign(data[:, 1])

            dist = np.sqrt(np.sum(data ** 2, axis=1))

            # print(dist,'dist')
            # Find indices of gaps for each beta
            # Find indices of gaps for each beta
            gap_indices = np.zeros_like(beta, dtype=int)
            for j in range(len(gaps)):
                mask = (gaps[j][0] <= beta) & (beta < gaps[j][1])
                gap_indices[mask] = j

            # Update point_cloud_data
            for i in range(len(filtered_data)):
                point_cloud_data[gap_indices[i]] = min(point_cloud_data[gap_indices[i]], dist[i])
        # print(point_cloud_data,"DATA CHECK")
        if flag_pc_data== False:
            initial_point_cloud_data = point_cloud_data
            flag_pc_data = True

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

class PerformanceTracker:
    def __init__(self, max_episodes=50, success_threshold=0.5):
        self.episode_results = deque(maxlen=max_episodes)
        self.success_threshold = success_threshold

    def add_result(self, success):
        self.episode_results.append(success)

    def success_rate(self):
        if len(self.episode_results) == 0:
            return 0.0
        return sum(self.episode_results) / len(self.episode_results)

    def has_reached_threshold(self):
        return self.success_rate() >= self.success_threshold


if __name__ == '__main__':

    rclpy.init(args=None)

    # writer = SummaryWriter()
    performance_tracker = PerformanceTracker()

    seed = 1  # Random seed number
    eval_freq = 50e3  # After how many steps to perform the evaluation
    max_ep = 1000  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        50000  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.05  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 128  # Size of the mini-batch
    discount = 0.95  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.001  # Soft target update variable (should be close to 0)
    policy_noise = 0.1  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 4  # Frequency of Actor network updates
    buffer_size = 1000000  # Maximum size of the buffer
    file_name = "td3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model
    random_near_obstacle = False  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if save_model and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Create the training environment
    environment_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    network_inputs = state_dim
    network_outputs = action_dim
    memorySize = 1000000
    explorationRate = 1
    discountFactor = 0.99
    learningRate = 0.0001
    learnStart = 64
    updateTargetNetwork = 10000
    minibatch_size = 64
    network_structure = [300,300]

    # network = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
    # network.initNetworks(network_structure)

    # Create the network
    network = td3(state_dim, action_dim, max_action,learningRate)
    writer=network.writer
    # Create a replay buffer
    replay_buffer = Memory(buffer_size, alpha=0.7)
    if load_model:
        try:
            print("Will load existing model.")
            network.load(file_name, "./pytorch_models")
        except:
            print("Could not load the stored model parameters, initializing training with random parameters")

    # Create evaluation data store
    evaluations = []

    timestep = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    epoch = 1

    count_rand_actions = 0
    random_action = []

    env = Env()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Point_cloud_subscriber()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    check_start = max_ep
    check_end = max_ep * 2
    check_inc = 1

    rate = odom_subscriber.create_rate(2)
    # velodyne_subscriber.create_rate(2)
    successes = 0
    failures = 0
    cumulated_reward = 0
    distance = 0
    theta = 0
    try:
        while rclpy.ok():
            if timestep < max_timesteps:
                # On termination of episode
                if done:
                    env.get_logger().info(f"Done. timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"train")
                        network.train(
                            replay_buffer,
                            episode_timesteps,
                            cumulated_reward,
                            distance,
                            theta,
                            successes,
                            failures,
                            batch_size,
                            discount,
                            tau,
                            policy_noise,
                            noise_clip,
                            policy_freq
                        )

                    if(timesteps_since_eval >= check_start) and (timesteps_since_eval <= check_end):
                        checkpoint_path = f'checkpoint_epoch_{check_inc}'
                        network.save(checkpoint_path,directory="./checkpoints")
                        env.get_logger().info("Saving" + checkpoint_path)
                        check_start+=max_ep
                        check_end+=max_ep
                        check_inc+=1
                        
                    if timesteps_since_eval >= eval_freq:
                        check_end = max_ep * 2
                        check_inc = 1
                        env.get_logger().info("Validating")
                        timesteps_since_eval %= eval_freq
                        evaluations.append(
                            evaluate(network=network, epoch=epoch, eval_episodes=eval_ep,velodyne_subscriber=velodyne_subscriber)
                        )

                        network.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations)
                        epoch += 1
                    
                    if(performance_tracker.has_reached_threshold()):
                        state = env.reset(True,True)
                    else:
                        state = env.reset(False,True)
                    done = False

                    # _state = np.array(state)
                    # qValues = network.getQValues(_state)
                
                    # action = network.selectAction(qValues,explorationRate)
                    
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                    cumulated_reward = 0

                # print(state)
                # add some exploration noise
                if expl_noise > expl_min:
                    expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

                action = network.get_action(np.array(state))
                action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action = random_action
                        action[0] = -1

                # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                a_in = [(action[0] + 1) / 2, action[1]]
                next_state, reward, done, target,distance,theta = env.step(a_in)
                done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
                # done = 1 if episode_timesteps + 1 == max_ep else int(done)
                if(episode_timesteps+1 ==max_ep):
                    done =1
                    reward+=-100
                episode_reward += reward
                
                cumulated_reward += reward
                # Save the tuple in replay buffer
                replay_buffer.addMemory(state, action, reward, next_state,done)
                if done:
                    if target:
                        successes+=1
                        performance_tracker.add_result(1)
                    else:
                        failures+=1
                        performance_tracker.add_result(0)
                    
                # Update the counters
                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1


    except KeyboardInterrupt:
        pass

    rclpy.shutdown()