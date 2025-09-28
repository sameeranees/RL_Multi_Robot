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


# from omni.isaac.core import SimulationContext
# from omni.isaac.core.utils.nucleus import get_assets_root_path
# from omni.isaac.core.utils.prims import create_prim
# from omni.isaac.core.utils.stage import add_reference_to_stage, is_stage_loading

GOAL_REACHED_DIST = 58
COLLISION_DIST = 0.42
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
point_cloud_data = np.ones(environment_dim) * 10

# _, sensor = omni.kit.commands.execute(
#     "IsaacSensorCreateRtxLidar",
#     path="/sensor",
#     parent=None,
#     config="Example_Rotary",
#     translation=(0, 0, 1.0),
#     orientation=Gf.Quatd(0.5, 0.5, -0.5, -0.5),
# )
# render_product = rep.create.render_product(sensor.GetPath(), [1, 1])
# sensors.get_synthetic_data().activate_node_template("RtxLidar" + "ROS2PublishPointCloud", 0, [render_product.path])


def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        env.get_logger().info(f"evaluating episode {_}")
        count = 0
        state = env.reset()
        done = False
        while not done and count < 2001:
            action = network.get_action(np.array(state))
            # env.get_logger().info(f"action : {action}")
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
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


# TD3 network
class td3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


# # td3 network
# class td3(object):
#     def __init__(self, state_dim, action_dim, max_action):
#         # Initialize the Actor network
#         self.actor = Actor(state_dim, action_dim).to(device)
#         self.actor_target = Actor(state_dim, action_dim).to(device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

#         # Initialize the Critic networks
#         self.critic = Critic(state_dim, action_dim).to(device)
#         self.critic_target = Critic(state_dim, action_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

#         self.max_action = max_action
#         self.writer = SummaryWriter(log_dir="./DRL_robot_navigation_ros2/src/td3/scripts/runs")
#         # time.sleep(5)
#         # os.path.dirname(os.path.realpath(__file__)) + "/runs"
#         self.iter_count = 0

#     def get_action(self, state):
#         # Function to get the action from the actor
#         state = torch.Tensor(state.reshape(1, -1)).to(device)
#         return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

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
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
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
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

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

        self.upper = 5.0
        self.lower = -5.0

        self.set_self_state_1 = ModelState()
        self.set_self_state_1.model_name = "r1"
        self.set_self_state_1.pose.position.x = 0.0
        self.set_self_state_1.pose.position.y = 0.0
        self.set_self_state_1.pose.position.z = 0.0
        self.set_self_state_1.pose.orientation.x = 0.0
        self.set_self_state_1.pose.orientation.y = 0.0
        self.set_self_state_1.pose.orientation.z = 0.0
        self.set_self_state_1.pose.orientation.w = 1.0

        self.set_self_state_2 = ModelState()
        self.set_self_state_2.model_name = "r2"
        self.set_self_state_2.pose.position.x = 0.0
        self.set_self_state_2.pose.position.y = 0.0
        self.set_self_state_2.pose.position.z = 0.0
        self.set_self_state_2.pose.orientation.x = 0.0
        self.set_self_state_2.pose.orientation.y = 0.0
        self.set_self_state_2.pose.orientation.z = 0.0
        self.set_self_state_2.pose.orientation.w = 1.0

        self.set_self_state_3 = ModelState()
        self.set_self_state_3.model_name = "r3"
        self.set_self_state_3.pose.position.x = 0.0
        self.set_self_state_3.pose.position.y = 0.0
        self.set_self_state_3.pose.position.z = 0.0
        self.set_self_state_3.pose.orientation.x = 0.0
        self.set_self_state_3.pose.orientation.y = 0.0
        self.set_self_state_3.pose.orientation.z = 0.0
        self.set_self_state_3.pose.orientation.w = 1.0

        self.set_self_state_4 = ModelState()
        self.set_self_state_4.model_name = "r4"
        self.set_self_state_4.pose.position.x = 0.0
        self.set_self_state_4.pose.position.y = 0.0
        self.set_self_state_4.pose.position.z = 0.0
        self.set_self_state_4.pose.orientation.x = 0.0
        self.set_self_state_4.pose.orientation.y = 0.0
        self.set_self_state_4.pose.orientation.z = 0.0
        self.set_self_state_4.pose.orientation.w = 1.0

        # Set up the ROS publishers and subscribers
        self.vel_pub_1 = self.create_publisher(Twist, "/cmd_vel_1", 1)
        self.set_state_1 = self.create_publisher(ModelState, "gazebo/set_model_state_1", 10)
        
        self.vel_pub_2 = self.create_publisher(Twist, "/cmd_vel_2", 1)
        self.set_state_2 = self.create_publisher(ModelState, "gazebo/set_model_state_2", 10)

        self.vel_pub_3 = self.create_publisher(Twist, "/cmd_vel_3", 1)
        self.set_state_3 = self.create_publisher(ModelState, "gazebo/set_model_state_3", 10)

        self.vel_pub_4 = self.create_publisher(Twist, "/cmd_vel_4", 1)
        self.set_state_4 = self.create_publisher(ModelState, "gazebo/set_model_state_4", 10)

        # self.unpause = self.create_client(Empty, "/unpause_physics")
        # self.pause = self.create_client(Empty, "/pause_physics")
        # self.reset_proxy = self.create_client(Empty, "/reset_world")
        # self.req = Empty.Request
        # assets_root_path = get_assets_root_path()
        # assets_root_path = get_assets_root_path()
        # print(assets_root_path)
        self.asset_path = 'omniverse://localhost/Projects/isaac-env/ISAAC MATERIAL/Collected_Thesis_3/Collected_Thesis/Thesis.usd'
        # self.asset_path = 'omniverse://localhost/Projects/isaac-env/ISAAC MATERIAL/Collected Thesis 2/Collected_Poster_scene/Poster_scene.usd'
        # omni.usd.get_context().open_stage(self.asset_path)
        # simulation_app.update()
        # while is_stage_loading():
        #     simulation_app.update()
        self.simulation_context = SimulationContext(stage_units_in_meters=1.0)
        self.simulation_context.set_simulation_dt(1/200.0,1/60.0)
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
    def step(self, action,robot):
        global point_cloud_data_1
        global point_cloud_data_2
        global point_cloud_data_3
        global point_cloud_data_4
        
        target = False
        max_linear = 2
        max_angular = 18
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0]) * max_linear
        vel_cmd.angular.z = float(action[1]) * max_angular
        
        if robot==1:
            self.vel_pub_1.publish(vel_cmd)
        if robot==2:
            self.vel_pub_2.publish(vel_cmd)
        if robot==3:
            self.vel_pub_3.publish(vel_cmd)
        if robot==4:
            self.vel_pub_4.publish(vel_cmd)
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
        self.simulation_context.step(render=True)
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

        if(robot==1):
            point_cloud_data = point_cloud_data_1
            last_odom = last_odom_1
        if(robot==2):
            point_cloud_data = point_cloud_data_2
            last_odom = last_odom_2
        if(robot==3):
            point_cloud_data = point_cloud_data_3
            last_odom = last_odom_3
        if(robot==4):
            point_cloud_data = point_cloud_data_4
            last_odom = last_odom_4

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(point_cloud_data)
        v_state = []
        v_state[:] = point_cloud_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x * 100 + 1.14435
        self.odom_y = last_odom.pose.pose.position.y * 100 + 312.0143
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

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self,robot):

        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        # while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('reset : service not available, waiting again...')

        # try:
        #     self.reset_proxy.call_async(Empty.Request())
        # except rclpy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed")
        if self.simulation_context.is_stopped()==False:
            self.simulation_context.stop()
        
        self.simulation_context.reset()
        
        # self.simulation_context.initialize_physics()

        simulation_app.update()

        while is_stage_loading():
            simulation_app.update()

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        if(robot==1):
            object_state = self.set_self_state_1
        if(robot==2):
            object_state = self.set_self_state_2
        if(robot==3):
            object_state = self.set_self_state_3
        if(robot==4):
            object_state = self.set_self_state_4

        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        if(robot==1):
            self.set_state_1.publish(object_state)
        if(robot==2):
            self.set_state_2.publish(object_state)
        if(robot==3):
            self.set_state_3.publish(object_state)
        if(robot==4):
            self.set_state_4.publish(object_state)
        # self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x * 100 + 1.14435
        self.odom_y = object_state.pose.position.y * 100 + 312.0143

        camera_view = ArticulationView(prim_paths_expr="/root/Camera/Camera")
        # camera_view.disable_rigid_body_physics()
        camera_view.set_local_poses(translations=[np.array([1005.75592,51.10774,1217.84581])],orientations=[np.array([0.93045,0.0307,0.33211,0.15171])])
        positions_target,orientations_target=camera_view.get_local_poses()
        print(positions_target,orientations_target)
        
        # time.sleep(10)
        self.change_goal()
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

        if(robot==1):
            point_cloud_data = point_cloud_data_1
            last_odom = last_odom_1
        if(robot==2):
            point_cloud_data = point_cloud_data_2
            last_odom = last_odom_2
        if(robot==3):
            point_cloud_data = point_cloud_data_3
            last_odom = last_odom_3
        if(robot==4):
            point_cloud_data = point_cloud_data_4
            last_odom = last_odom_4

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

    def change_goal(self):
        rigid_prim_view = ArticulationView(prim_paths_expr="/root/arena_2/Target")
        goal_ok = False
        self.goal_y= 3.2

        # cords = [[502.0,0.0],[-460,0.0]] 
        cords = [[502.0,0.0]]
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


    # def change_goal(self):
    #     # Place a new goal and check if its location is not on one of the obstacles
    #     if self.upper < 10:
    #         self.upper += 0.004
    #     if self.lower > -10:
    #         self.lower -= 0.004

    #     goal_ok = False

    #     while not goal_ok:
    #         self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
    #         self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
    #         goal_ok = check_pos(self.goal_x, self.goal_y)

    # def random_box(self):
    #     # Randomly change the location of the boxes in the environment on each reset to randomize the training
    #     # environment
    #     for i in range(4):
    #         name = "cardboard_box_" + str(i)

    #         x = 0
    #         y = 0
    #         box_ok = False
    #         while not box_ok:
    #             x = np.random.uniform(-6, 6)
    #             y = np.random.uniform(-6, 6)
    #             box_ok = check_pos(x, y)
    #             distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
    #             distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
    #             if distance_to_robot < 1.5 or distance_to_goal < 1.5:
    #                 box_ok = False
    #         box_state = ModelState()
    #         box_state.model_name = name
    #         box_state.pose.position.x = x
    #         box_state.pose.position.y = y
    #         box_state.pose.position.z = 0.0
    #         box_state.pose.orientation.x = 0.0
    #         box_state.pose.orientation.y = 0.0
    #         box_state.pose.orientation.z = 0.0
    #         box_state.pose.orientation.w = 1.0
    #         self.set_state.publish(box_state)

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
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            env.get_logger().info("reward 100")
            return 100.0
        elif collision:
            env.get_logger().info("reward -100")
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber_1')
        super().__init__('odom_subscriber_2')
        super().__init__('odom_subscriber_3')
        super().__init__('odom_subscriber_4')
        self.subscription_1 = self.create_subscription(
            Odometry,
            '/odom_1',
            self.odom_callback_1,
            10)
        self.subscription_2 = self.create_subscription(
            Odometry,
            '/odom_2',
            self.odom_callback_2,
            10)
        self.subscription_3 = self.create_subscription(
            Odometry,
            '/odom_3',
            self.odom_callback_3,
            10)
        self.subscription_4 = self.create_subscription(
            Odometry,
            '/odom_4',
            self.odom_callback_4,
            10)
        self.subscription_1
        self.subscription_2
        self.subscription_3
        self.subscription_4

    def odom_callback_1(self, od_data):
        global last_odom_1
        last_odom_1 = od_data
        # print(last_odom_1)
    
    def odom_callback_2(self, od_data):
        global last_odom_2
        last_odom_2 = od_data
        # print(last_odom_2)

    def odom_callback_3(self, od_data):
        global last_odom_3
        last_odom_3 = od_data
        # print(last_odom_3)

    def odom_callback_4(self, od_data):
        global last_odom_4
        last_odom_4 = od_data
        # print(last_odom_4)

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



class Point_cloud_subscriber(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_1')
        super().__init__('point_cloud_subscriber_2')
        super().__init__('point_cloud_subscriber_3')
        super().__init__('point_cloud_subscriber_4')
        self.subscription_1 = self.create_subscription(
            PointCloud2,
            "/point_cloud_1",
            self.point_cloud_callback_1,
            10)
        self.subscription_2 = self.create_subscription(
            PointCloud2,
            "/point_cloud_2",
            self.point_cloud_callback_2,
            10)
        self.subscription_3 = self.create_subscription(
            PointCloud2,
            "/point_cloud_3",
            self.point_cloud_callback_3,
            10)
        self.subscription_4 = self.create_subscription(
            PointCloud2,
            "/point_cloud_4",
            self.point_cloud_callback_4,
            10)
        self.subscription_1
        self.subscription_2
        self.subscription_3
        self.subscription_4

        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback_1(self, v):
        global point_cloud_data_1
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data_1 = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_1_2,'PCD')

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

            # Update point_cloud_data_1
            for i in range(len(filtered_data)):
                point_cloud_data_1[gap_indices[i]] = min(point_cloud_data_1[gap_indices[i]], dist[i])
        # print(point_cloud_data,"DATA CHECK")
    
    def point_cloud_callback_2(self, v):
        global point_cloud_data_2
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data_2 = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_2_2,'PCD')

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

            # Update point_cloud_data_2
            for i in range(len(filtered_data)):
                point_cloud_data_2[gap_indices[i]] = min(point_cloud_data_2[gap_indices[i]], dist[i])
        # print(point_cloud_data,"DATA CHECK")
    
    def point_cloud_callback_3(self, v):
        global point_cloud_data_3
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data_3 = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_3_2,'PCD')

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

            # Update point_cloud_data_3
            for i in range(len(filtered_data)):
                point_cloud_data_3[gap_indices[i]] = min(point_cloud_data_3[gap_indices[i]], dist[i])
        # print(point_cloud_data,"DATA CHECK")
    
    def point_cloud_callback_4(self, v):
        global point_cloud_data_4
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data_4 = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_4_2,'PCD')

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

            # Update point_cloud_data_4
            for i in range(len(filtered_data)):
                point_cloud_data_4[gap_indices[i]] = min(point_cloud_data_4[gap_indices[i]], dist[i])
        # print(point_cloud_data_4,"DATA CHECK")



if __name__ == '__main__':

    rclpy.init(args=None)

    seed = 0  # Random seed number
    eval_freq = 5e3  # After how many steps to perform the evaluation
    max_ep = 2000  # maximum number of steps per episode
    eval_ep = 10  # number of episodes for evaluation
    max_timesteps = 5e6  # Maximum number of steps to perform
    expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
    expl_decay_steps = (
        500000  # Number of steps over which the initial exploration noise will decay over
    )
    expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
    batch_size = 40  # Size of the mini-batch
    discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
    tau = 0.004  # Soft target update variable (should be close to 0)
    policy_noise = 0.2  # Added noise for exploration
    noise_clip = 0.5  # Maximum clamping values of the noise
    policy_freq = 2  # Frequency of Actor network updates
    buffer_size = 1e4  # Maximum size of the buffer
    file_name = "td3_velodyne"  # name of the file to store the policy
    save_model = True  # Weather to save the model or not
    load_model = False  # Weather to load a stored model
    random_near_obstacle = True  # To take random actions near obstacles or not

    # Create the network storage folders
    if not os.path.exists("./robot_1"):
        os.makedirs("./robot_1")

    if not os.path.exists("./robot_2"):
        os.makedirs("./robot_2")

    if not os.path.exists("./robot_3"):
        os.makedirs("./robot_3")

    if not os.path.exists("./robot_4"):
        os.makedirs("./robot_4")    
    
    if not os.path.exists("./robot_1/results"):
        os.makedirs("./robot_1/results")
    if save_model and not os.path.exists("./robot_1/pytorch_models"):
        os.makedirs("./robot_1/pytorch_models")
    
    if not os.path.exists("./robot_2/results"):
        os.makedirs("./robot_2/results")
    if save_model and not os.path.exists("./robot_2/pytorch_models"):
        os.makedirs("./robot_2/pytorch_models")

    if not os.path.exists("./robot_3/results"):
        os.makedirs("./robot_3/results")
    if save_model and not os.path.exists("./robot_3/pytorch_models"):
        os.makedirs("./robot_3/pytorch_models")

    if not os.path.exists("./robot_4/results"):
        os.makedirs("./robot_4/results")
    if save_model and not os.path.exists("./robot_4/pytorch_models"):
        os.makedirs("./robot_4/pytorch_models")

    # Create the training environment
    environment_dim = 20
    robot_dim = 4

    torch.manual_seed(seed)
    np.random.seed(seed)
    state_dim = environment_dim + robot_dim
    action_dim = 2
    max_action = 1

    # Create the network
    network_1 = td3(state_dim, action_dim, max_action)
    network_2 = td3(state_dim, action_dim, max_action)
    network_3 = td3(state_dim, action_dim, max_action)
    network_4 = td3(state_dim, action_dim, max_action)
    # Create a replay buffer
    replay_buffer_1 = ReplayBuffer(buffer_size, seed)
    replay_buffer_2 = ReplayBuffer(buffer_size, seed)
    replay_buffer_3 = ReplayBuffer(buffer_size, seed)
    replay_buffer_4 = ReplayBuffer(buffer_size, seed)

    if load_model:
        try:
            print("Will load existing model.")
            network_1.load(file_name, "./robot_1/pytorch_models")
            network_2.load(file_name, "./robot_2/pytorch_models")
            network_3.load(file_name, "./robot_3/pytorch_models")
            network_4.load(file_name, "./robot_4/pytorch_models")
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
    point_cloud_subscriber = Point_cloud_subscriber()
    # odom_subscriber2 = Odom_subscriber2()
    global robomaster
    robomaster = ArticulationView(prim_paths_expr="/root/robomaster/base_footprint")
    # odom_publisher = Odom_publisher()
    
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    # executor.add_node(odom_subscriber2)
    executor.add_node(point_cloud_subscriber)
    # executor.add_node(odom_publisher)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    rate = odom_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            if timestep < max_timesteps:
                # On termination of episode
                if done:
                    env.get_logger().info(f"Done. timestep : {timestep}")
                    if timestep != 0:
                        env.get_logger().info(f"train")
                        network_1.train(
                            replay_buffer_1,
                            episode_timesteps,
                            batch_size,
                            discount,
                            tau,
                            policy_noise,
                            noise_clip,
                            policy_freq,
                        )
                        network_2.train(
                            replay_buffer_1,
                            episode_timesteps,
                            batch_size,
                            discount,
                            tau,
                            policy_noise,
                            noise_clip,
                            policy_freq,
                        )
                        network_3.train(
                            replay_buffer_1,
                            episode_timesteps,
                            batch_size,
                            discount,
                            tau,
                            policy_noise,
                            noise_clip,
                            policy_freq,
                        )
                        network_4.train(
                            replay_buffer_1,
                            episode_timesteps,
                            batch_size,
                            discount,
                            tau,
                            policy_noise,
                            noise_clip,
                            policy_freq,
                        )

                    if timesteps_since_eval >= eval_freq:
                        env.get_logger().info("Validating")
                        timesteps_since_eval %= eval_freq
                        evaluations.append(
                            [   
                                evaluate(network=network_1, epoch=epoch, eval_episodes=eval_ep),
                                evaluate(network=network_2, epoch=epoch, eval_episodes=eval_ep),
                                evaluate(network=network_3, epoch=epoch, eval_episodes=eval_ep),
                                evaluate(network=network_4, epoch=epoch, eval_episodes=eval_ep)
                            ]
                        )

                        network_1.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations[:][0])
                        # epoch += 1

                        network_2.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations[:][1])
                        # epoch += 1

                        network_3.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations[:][2])
                        # epoch += 1

                        network_4.save(file_name, directory="./pytorch_models")
                        np.save("./results/%s" % (file_name), evaluations[:][3])
                        
                        epoch += 1

                    state_1 = env.reset(robot=1)
                    state_2 = env.reset(robot=2)
                    state_3 = env.reset(robot=3)
                    state_4 = env.reset(robot=4)

                    done = False

                    episode_reward_1 = 0
                    episode_reward_2 = 0
                    episode_reward_3 = 0
                    episode_reward_4 = 0

                    episode_timesteps = 0
                    episode_num += 1

                # add some exploration noise
                if expl_noise > expl_min:
                    expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)


                #EDIT HERE AFTER BREAK!!!!
                action_1 = network_1.get_action(np.array(state_1))
                action_1 = (action_1 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                action_2 = network_2.get_action(np.array(state_2))
                action_2 = (action_2 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                action_3 = network_3.get_action(np.array(state_3))
                action_3 = (action_3 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                action_4 = network_4.get_action(np.array(state_4))
                action_4 = (action_4 + np.random.normal(0, expl_noise, size=action_dim)).clip(
                     -max_action, max_action
                )

                # If the robot is facing an obstacle, randomly force it to take a consistent random action.
                # This is done to increase exploration in situations near obstacles.
                # Training can also be performed without it
                
                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state_1[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action_1 = random_action
                        action_1[0] = -1
                
                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state_2[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action_2 = random_action
                        action_2[0] = -1

                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state_3[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action_3 = random_action
                        action_3[0] = -1

                if random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state_4[4:-8]) < 0.6
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-1, 1, 2)

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action_4 = random_action
                        action_4[0] = -1

                # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
                a_in = [(action_1[0] + 1) / 2, action_1[1]]
                next_state_1, reward_1, done_1, target_1 = env.step(a_in)
                done_bool_1 = 0 if episode_timesteps + 1 == max_ep else int(done_1)
                done_1 = 1 if episode_timesteps + 1 == max_ep else int(done_1)
                episode_reward_1 += reward_1

                # Save the tuple in replay buffer
                replay_buffer_1.add(state_1, action_1, reward_1, done_bool_1, next_state_1)

                a_in = [(action_2[0] + 1) / 2, action_2[1]]
                next_state_2, reward_2, done_2, target_2 = env.step(a_in)
                done_bool_2 = 0 if episode_timesteps + 1 == max_ep else int(done_2)
                done_2 = 1 if episode_timesteps + 1 == max_ep else int(done_2)
                episode_reward_2 += reward_2

                # Save the tuple in replay buffer
                replay_buffer_2.add(state_2, action_2, reward_2, done_bool_2, next_state_2)

                a_in = [(action_3[0] + 1) / 2, action_3[1]]
                next_state_3, reward_3, done_3, target_3 = env.step(a_in)
                done_bool_3 = 0 if episode_timesteps + 1 == max_ep else int(done_3)
                done_3 = 1 if episode_timesteps + 1 == max_ep else int(done_3)
                episode_reward_3 += reward_3

                # Save the tuple in replay buffer
                replay_buffer_3.add(state_3, action_3, reward_3, done_bool_3, next_state_3)

                a_in = [(action_4[0] + 1) / 2, action_4[1]]
                next_state_4, reward_4, done_4, target_4 = env.step(a_in)
                done_bool_4 = 0 if episode_timesteps + 1 == max_ep else int(done_4)
                done_4 = 1 if episode_timesteps + 1 == max_ep else int(done_4)
                episode_reward_4 += reward_4

                # Save the tuple in replay buffer
                replay_buffer_4.add(state_4, action_4, reward_4, done_bool_4, next_state_4)

                # Update the counters
                state_1 = next_state_1
                state_2 = next_state_2
                state_3 = next_state_3
                state_4 = next_state_4

                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1
        env.close()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()