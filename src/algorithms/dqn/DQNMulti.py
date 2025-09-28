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
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from collections import deque
# from deepq_torch import DeepQ
from deep_q_torch_2 import DeepQ
from distutils.dir_util import copy_tree
import json


GOAL_REACHED_DIST = 66
COLLISION_DIST = 0.28
ROBOT_COLLISION_DIST = 37.00
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
# last_odom = None

last_odom_rob_1 = None
last_odom_rob_2 = None
last_odom_rob_3 = None
last_odom_rob_4 = None
last_odom_rob_5 = None

environment_dim = 20
environment_dim_2 = 360

point_cloud_data_1 = np.ones(environment_dim) * 10
point_cloud_data_2 = np.ones(environment_dim) * 10
point_cloud_data_3 = np.ones(environment_dim) * 10
point_cloud_data_4 = np.ones(environment_dim) * 10
point_cloud_data_5 = np.ones(environment_dim) * 10

scanData_1 = np.ones(environment_dim_2) * 10
scanData_2 = np.ones(environment_dim_2) * 10 
scanData_3 = np.ones(environment_dim_2) * 10 
scanData_4 = np.ones(environment_dim_2) * 10 
scanData_5 = np.ones(environment_dim_2) * 10 

initial_point_cloud_data_1 = point_cloud_data_1
initial_point_cloud_data_2 = point_cloud_data_2
initial_point_cloud_data_3 = point_cloud_data_3
initial_point_cloud_data_4 = point_cloud_data_4
initial_point_cloud_data_5 = point_cloud_data_5

initial_scan_data_1 = scanData_1
initial_scan_data_2 = scanData_2
initial_scan_data_3 = scanData_3
initial_scan_data_4 = scanData_4
initial_scan_data_5 = scanData_5

flag_pc_data_1=False
flag_pc_data_2=False
flag_pc_data_3=False
flag_pc_data_4=False
flag_pc_data_5=False

flag_ls_data_1=False
flag_ls_data_2=False
flag_ls_data_3=False
flag_ls_data_4=False
flag_ls_data_5=False

old_distance_1 = np.inf
old_distance_2 = np.inf
old_distance_3 = np.inf
old_distance_4 = np.inf
old_distance_5 = np.inf

old_theta_1 = np.inf
old_theta_2 = np.inf
old_theta_3 = np.inf
old_theta_4 = np.inf
old_theta_5 = np.inf

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
        self.vel_pub_1 = self.create_publisher(Twist, "/cmd_vel_rob_1", 1)
        self.vel_pub_2 = self.create_publisher(Twist, "/cmd_vel_rob_2", 1)
        self.vel_pub_3 = self.create_publisher(Twist, "/cmd_vel_rob_3", 1)
        self.vel_pub_4 = self.create_publisher(Twist, "/cmd_vel_rob_4", 1)
        self.vel_pub_5 = self.create_publisher(Twist, "/cmd_vel_rob_5", 1)
        
        self.set_state = self.create_publisher(ModelState, "gazebo/set_model_state", 10)
        

        # self.unpause = self.create_client(Empty, "/unpause_physics")
        # self.pause = self.create_client(Empty, "/pause_physics")
        # self.reset_proxy = self.create_client(Empty, "/reset_world")
        # self.req = Empty.Request
        # assets_root_path = get_assets_root_path()
        # assets_root_path = get_assets_root_path()
        # print(assets_root_path)
        self.asset_path = 'omniverse://localhost/Projects/isaac-env/ISAAC MATERIAL/Collected_Thesis5/Thesis1.usd'
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

    def get_distance(self,last_odom,translate_pos_x,translate_pos_y):
        # Calculate robot heading from odometry data
        odom_x = last_odom.pose.pose.position.x * 100 + translate_pos_x
        odom_y = last_odom.pose.pose.position.y * 100 + translate_pos_y
        
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
            [odom_x - self.goal_x, odom_y - self.goal_y]
        )

        return distance, angle,odom_x,odom_y
    
    def get_skew(self, angle, odom_x, odom_y):
        skew_x = self.goal_x - odom_x
        skew_y = self.goal_y - odom_y
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
        
        return theta

    def rob_to_rob(self,odom_x,odom_y,rob_odoms):
        distances=[]
        collision = False
        for odom_x_1,odom_y_1 in rob_odoms:
            distance = np.linalg.norm([odom_x - odom_x_1, odom_y - odom_y_1])
            if distance < ROBOT_COLLISION_DIST :
                print("COLLISION PASSIERT HIER!")
                collision = True
            distances.append(distance)
        return distances,collision

    # Perform an action and read a new state
    def step(self, action_1,action_2,action_3,action_4,action_5):
        global point_cloud_data_1, point_cloud_data_2,point_cloud_data_3,point_cloud_data_4,point_cloud_data_5
        global scanData_1, scanData_2, scanData_3, scanData_4, scanData_5
        global old_distance_1, old_distance_2, old_distance_3,old_distance_4,old_distance_5
        global old_theta_1, old_theta_2,old_theta_3, old_theta_4, old_theta_5
        global last_odom_rob_1, last_odom_rob_2,last_odom_rob_3,last_odom_rob_4,last_odom_rob_5

        done_1=False
        done_2=False
        done_3=False
        done_4=False
        done_5=False

        max_linear = 0.5
        
        target_1 = False
        target_2 = False
        target_3 = False
        target_4 = False
        target_5 = False
        
        # max_linear = 2
        max_angular = 18
        
        # Publish the robot action
        vel_cmd_1 = Twist()
        vel_cmd_1.linear.x = float(action_1[0]) * max_linear
        vel_cmd_1.angular.z = float(action_1[1]) * max_angular

        vel_cmd_2 = Twist()
        vel_cmd_2.linear.x = float(action_2[0]) * max_linear
        vel_cmd_2.angular.z = float(action_2[1]) * max_angular

        vel_cmd_3 = Twist()
        vel_cmd_3.linear.x = float(action_3[0]) * max_linear
        vel_cmd_3.angular.z = float(action_3[1]) * max_angular

        vel_cmd_4 = Twist()
        vel_cmd_4.linear.x = float(action_4[0]) * max_linear
        vel_cmd_4.angular.z = float(action_4[1]) * max_angular

        vel_cmd_5 = Twist()
        vel_cmd_5.linear.x = float(action_5[0]) * max_linear
        vel_cmd_5.angular.z = float(action_5[1]) * max_angular

        self.vel_pub_1.publish(vel_cmd_1)
        self.vel_pub_2.publish(vel_cmd_2)
        self.vel_pub_3.publish(vel_cmd_3)
        self.vel_pub_4.publish(vel_cmd_4)
        self.vel_pub_5.publish(vel_cmd_5)

        self.publish_markers(action_1)

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
        start_time = time.time()
        while not(time.time() - start_time >=0.2):# and (action[1] == 0):
            # print(time.time() - start_time, "TIME IT TOOK")
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

        # read velodyne laser state
        done_1, collision_1, min_laser_1 = self.observe_collision(point_cloud_data_1)
        done_2, collision_2, min_laser_2 = self.observe_collision(point_cloud_data_2)
        done_3, collision_3, min_laser_3 = self.observe_collision(point_cloud_data_3)
        done_4, collision_4, min_laser_4 = self.observe_collision(point_cloud_data_4)
        done_5, collision_5, min_laser_5 = self.observe_collision(point_cloud_data_5)
        
        v_state_1 = []
        v_state_1[:] = scanData_1[:]
        laser_state_1 = [v_state_1]

        v_state_2 = []
        v_state_2[:] = scanData_2[:]
        laser_state_2 = [v_state_2]

        v_state_3 = []
        v_state_3[:] = scanData_3[:]
        laser_state_3 = [v_state_3]

        v_state_4 = []
        v_state_4[:] = scanData_4[:]
        laser_state_4 = [v_state_4]

        v_state_5 = []
        v_state_5[:] = scanData_5[:]
        laser_state_5 = [v_state_5]


        # # Calculate robot heading from odometry data
        # self.odom_x = last_odom.pose.pose.position.x * 100 + 1.14435
        # self.odom_y = last_odom.pose.pose.position.y * 100 + 312.0143
        # quaternion = Quaternion(
        #     last_odom.pose.pose.orientation.w,
        #     last_odom.pose.pose.orientation.x,
        #     last_odom.pose.pose.orientation.y,
        #     last_odom.pose.pose.orientation.z,
        # )
        # euler = quaternion.to_euler(degrees=False)
        # angle = round(euler[2], 4)

        # # Calculate distance to the goal from the robot
        # distance = np.linalg.norm(
        #     [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        # )

        distance_1, angle_1, odom_x_1, odom_y_1 = self.get_distance(last_odom_rob_1,1.14435,312.0143)
        distance_2, angle_2, odom_x_2, odom_y_2 = self.get_distance(last_odom_rob_2,1.14435,166.68101)
        distance_3, angle_3, odom_x_3, odom_y_3 = self.get_distance(last_odom_rob_3,1.14435,-271.4367)
        distance_4, angle_4, odom_x_4, odom_y_4 = self.get_distance(last_odom_rob_4,1.14435,-157.6057)
        distance_5, angle_5, odom_x_5, odom_y_5 = self.get_distance(last_odom_rob_5,1.14435,-4.43865)

        collision_odom_1 = False
        collision_odom_2 = False
        collision_odom_3 = False
        collision_odom_4 = False
        collision_odom_5 = False 

        rob_1_dist, collision_odom_1 = self.rob_to_rob(odom_x_1, odom_y_1,[(odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_2_dist, collision_odom_2 = self.rob_to_rob(odom_x_2, odom_y_2,[(odom_x_1, odom_y_1),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_3_dist, collision_odom_3 = self.rob_to_rob(odom_x_3, odom_y_3,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_4_dist, collision_odom_4 = self.rob_to_rob(odom_x_4, odom_y_4,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_5, odom_y_5)])
        
        rob_5_dist, collision_odom_5 = self.rob_to_rob(odom_x_5, odom_y_5,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4)])
        
        # print(rob_1_dist)
        # print(rob_2_dist)
        # print(rob_3_dist)
        # print(rob_4_dist)
        # print(rob_5_dist)
        if (collision_1==True) or (collision_odom_1==True):
            collision_1 = True
        if (collision_2==True) or (collision_odom_2==True):
            collision_2 = True
        if (collision_3==True) or (collision_odom_3==True):
            collision_3 = True
        if (collision_4==True) or (collision_odom_4==True):
            collision_4 = True
        if (collision_5==True) or (collision_odom_5==True):
            collision_5 = True


        done_1 = collision_1
        done_2 = collision_2
        done_3 = collision_3
        done_4 = collision_4
        done_5 = collision_5

        rob_1_vel = [vel_cmd_2.linear.x,
                     vel_cmd_3.linear.x,
                     vel_cmd_4.linear.x,
                     vel_cmd_5.linear.x]
        
        rob_2_vel = [vel_cmd_1.linear.x,
                     vel_cmd_3.linear.x,
                     vel_cmd_4.linear.x,
                     vel_cmd_5.linear.x]
        
        rob_3_vel = [vel_cmd_1.linear.x,
                     vel_cmd_2.linear.x,
                     vel_cmd_4.linear.x,
                     vel_cmd_5.linear.x]
        
        rob_4_vel = [vel_cmd_1.linear.x,
                     vel_cmd_2.linear.x,
                     vel_cmd_3.linear.x,
                     vel_cmd_5.linear.x]
        
        rob_5_vel = [vel_cmd_1.linear.x,
                     vel_cmd_2.linear.x,
                     vel_cmd_3.linear.x,
                     vel_cmd_4.linear.x]
        
        # print(self.odom_x,self.odom_y,'ODOMETRY')
        # print(distance,"THE DISTANCE TO TARGET")
        # print('ODOM',self.odom_x,self.odom_y,'; ', 'GOAL',self.goal_x,self.goal_y,self.goal_z)
        # Calculate the relative angle between the robots heading and heading toward the goal
        # skew_x = self.goal_x - self.odom_x
        # skew_y = self.goal_y - self.odom_y
        # dot = skew_x * 1 + skew_y * 0
        # mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        # mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        # beta = math.acos(dot / (mag1 * mag2))
        # if skew_y < 0:
        #     if skew_x < 0:
        #         beta = -beta
        #     else:
        #         beta = 0 - beta
        # theta = beta - angle
        # if theta > np.pi:
        #     theta = np.pi - theta
        #     theta = -np.pi - theta
        # if theta < -np.pi:
        #     theta = -np.pi - theta
        #     theta = np.pi - theta

        theta_1 = self.get_skew(angle_1, odom_x_1, odom_y_1)
        theta_2 = self.get_skew(angle_2, odom_x_2, odom_y_2)
        theta_3 = self.get_skew(angle_3, odom_x_3, odom_y_3)
        theta_4 = self.get_skew(angle_4, odom_x_4, odom_y_4)
        theta_5 = self.get_skew(angle_5, odom_x_5, odom_y_5)


        # Detect if the goal has been reached and give a large positive reward
        if distance_1 < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target_1 = True
            done_1 = True

        if distance_2 < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target_2 = True
            done_2 = True
            
        if distance_3 < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target_3 = True
            done_3 = True
        
        if distance_4 < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target_4 = True
            done_4 = True

        if distance_5 < GOAL_REACHED_DIST:
            env.get_logger().info("GOAL is reached!")
            target_5 = True
            done_5 = True

        # print(distance,"THE DISTANCE TO TARGET", theta, "THE ANGLE TO THE TARGET")
        # self.odometry_history.append([self.odom_x,self.odom_y])
        robot_state_1 = [distance_1, theta_1, action_1[0], action_1[1]] + rob_1_dist + rob_1_vel
        robot_state_2 = [distance_2, theta_2, action_2[0], action_2[1]] + rob_2_dist + rob_2_vel
        robot_state_3 = [distance_3, theta_3, action_3[0], action_3[1]] + rob_3_dist + rob_3_vel
        robot_state_4 = [distance_4, theta_4, action_4[0], action_4[1]] + rob_4_dist + rob_4_vel
        robot_state_5 = [distance_5, theta_5, action_5[0], action_5[1]] + rob_5_dist + rob_5_vel

        # robot_state = robot_state_1 + robot_state_2 + robot_state_3 + robot_state_4 + robot_state_5
        # laser_state = laser_state_1 + laser_state_2 + laser_state_3 + laser_state_4 + laser_state_5

        state_1 = np.append(change_laser_distance(laser_state_1), robot_state_1)
        state_2 = np.append(change_laser_distance(laser_state_2), robot_state_2)
        state_3 = np.append(change_laser_distance(laser_state_3), robot_state_3)
        state_4 = np.append(change_laser_distance(laser_state_4), robot_state_4)
        state_5 = np.append(change_laser_distance(laser_state_5), robot_state_5)
        
        states = [state_1,state_2,state_3,state_4,state_5]

        # repetitive_behavior = self.detect_repetitive_behavior(self.odometry_history)
        reward_1,old_distance_1,old_theta_1 = self.get_reward(target_1, collision_1, vel_cmd_1.linear.x,vel_cmd_1.angular.z, min_laser_1,distance_1,theta_1,old_distance_1,old_theta_1)
        reward_2,old_distance_2,old_theta_2 = self.get_reward(target_2, collision_2, vel_cmd_2.linear.x,vel_cmd_2.angular.z, min_laser_2,distance_2,theta_2,old_distance_2,old_theta_2)
        reward_3,old_distance_3,old_theta_3 = self.get_reward(target_3, collision_3, vel_cmd_3.linear.x,vel_cmd_3.angular.z, min_laser_3,distance_3,theta_3,old_distance_3,old_theta_3)
        reward_4,old_distance_4,old_theta_4 = self.get_reward(target_4, collision_4, vel_cmd_4.linear.x,vel_cmd_4.angular.z, min_laser_4,distance_4,theta_4,old_distance_4,old_theta_4)
        reward_5,old_distance_5,old_theta_5 = self.get_reward(target_5, collision_5, vel_cmd_5.linear.x,vel_cmd_5.angular.z, min_laser_5,distance_5,theta_5,old_distance_5,old_theta_5)
        
        rewards = [reward_1,reward_2,reward_3,reward_4,reward_5]
        dones = [done_1,done_2,done_3,done_4,done_5]
        targets = [target_1,target_2,target_3,target_4,target_5]

        distances = []
        thetas=[]
        if done_1 or done_2 or done_3 or done_4 or done_5:
            distances=[distance_1,distance_2,distance_3,distance_4,distance_5]
            thetas=[theta_1,theta_2,theta_3,theta_4,theta_5]

        print("The reward for step is:", reward_1,"THE DISTANCE TO TARGET",distance_1,"THE ANGLE TO THE TARGET",theta_1)
        print("The reward for step is:", reward_2,"THE DISTANCE TO TARGET",distance_2,"THE ANGLE TO THE TARGET",theta_2)
        print("The reward for step is:", reward_3,"THE DISTANCE TO TARGET",distance_3,"THE ANGLE TO THE TARGET",theta_3)
        print("The reward for step is:", reward_4,"THE DISTANCE TO TARGET",distance_4,"THE ANGLE TO THE TARGET",theta_4)
        print("The reward for step is:", reward_5,"THE DISTANCE TO TARGET",distance_5,"THE ANGLE TO THE TARGET",theta_5)
    
        return states, rewards, dones, targets,distances,thetas
    

    def reset(self,change_target,change_robot=False):
        global old_distance_1, old_distance_2, old_distance_3,old_distance_4,old_distance_5
        global old_theta_1, old_theta_2,old_theta_3, old_theta_4, old_theta_5

        old_distance_1 = np.inf
        old_distance_2 = np.inf
        old_distance_3 = np.inf
        old_distance_4 = np.inf
        old_distance_5 = np.inf

        old_theta_1 = np.inf
        old_theta_2 = np.inf
        old_theta_3 = np.inf
        old_theta_4 = np.inf
        old_theta_5 = np.inf


        # Resets the state of the environment and returns an initial observation.
        #rospy.wait_for_service("/gazebo/reset_world")
        # while not self.reset_proxy.wait_for_service,(timeout_sec=1.0):
        #     self.get_logger().info('reset : service not available, waiting again...')

        # try:
        #     self.reset_proxy.call_async(Empty.Request())
        # except rclpy.ServiceException as e:
        #     print("/gazebo/reset_simulation service call failed")
        # point_cloud_data = np.ones(environment_dim) * 10
        if self.simulation_context.is_stopped()==False:
            self.simulation_context.stop()
        
        if change_robot:
            self.change_robot_location()
        self.simulation_context.reset()

        global point_cloud_data_1, point_cloud_data_2,point_cloud_data_3,point_cloud_data_4,point_cloud_data_5
        global initial_point_cloud_data_1, initial_point_cloud_data_2,initial_point_cloud_data_3,initial_point_cloud_data_4,initial_point_cloud_data_5
        global targets_changed
        point_cloud_data_1 = initial_point_cloud_data_1
        point_cloud_data_2 = initial_point_cloud_data_2
        point_cloud_data_3 = initial_point_cloud_data_3
        point_cloud_data_4 = initial_point_cloud_data_4
        point_cloud_data_5 = initial_point_cloud_data_5
        
        global scanData_1, scanData_2, scanData_3, scanData_4, scanData_5
        global initial_scan_data_1,initial_scan_data_2,initial_scan_data_3,initial_scan_data_4,initial_scan_data_5
        scanData_1 = initial_scan_data_1
        scanData_2 = initial_scan_data_2
        scanData_3 = initial_scan_data_3
        scanData_4 = initial_scan_data_4
        scanData_5 = initial_scan_data_5

        # velodyne_subscriber.executor.spin_once()
        self.odometry_history.clear()

        self.simulation_context.initialize_physics()
        # print(point_cloud_data,'Reached here')
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()
        simulation_app.update()
        while is_stage_loading():
            simulation_app.update()

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

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
        # simulation_app.update()
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

        # v_state = []
        # v_state[:] = point_cloud_data[:]
        # laser_state = [v_state]

        v_state_1 = []
        v_state_1[:] = scanData_1[:]
        laser_state_1 = [v_state_1]

        v_state_2 = []
        v_state_2[:] = scanData_2[:]
        laser_state_2 = [v_state_2]

        v_state_3 = []
        v_state_3[:] = scanData_3[:]
        laser_state_3 = [v_state_3]

        v_state_4 = []
        v_state_4[:] = scanData_4[:]
        laser_state_4 = [v_state_4]

        v_state_5 = []
        v_state_5[:] = scanData_5[:]
        laser_state_5 = [v_state_5]

        distance_1, angle_1, odom_x_1, odom_y_1 = self.get_distance(last_odom_rob_1,1.14435,312.0143)
        distance_2, angle_2, odom_x_2, odom_y_2 = self.get_distance(last_odom_rob_2,1.14435,166.68101)
        distance_3, angle_3, odom_x_3, odom_y_3 = self.get_distance(last_odom_rob_3,1.14435,-157.6057)
        distance_4, angle_4, odom_x_4, odom_y_4 = self.get_distance(last_odom_rob_4,1.14435,-271.4367)
        distance_5, angle_5, odom_x_5, odom_y_5 = self.get_distance(last_odom_rob_5,166.89903,-4.43865)

        theta_1 = self.get_skew(angle_1, odom_x_1, odom_y_1)
        theta_2 = self.get_skew(angle_2, odom_x_2, odom_y_2)
        theta_3 = self.get_skew(angle_3, odom_x_3, odom_y_3)
        theta_4 = self.get_skew(angle_4, odom_x_4, odom_y_4)
        theta_5 = self.get_skew(angle_5, odom_x_5, odom_y_5)

        rob_1_dist,_ = self.rob_to_rob(odom_x_1, odom_y_1,[(odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_2_dist,_ = self.rob_to_rob(odom_x_2, odom_y_2,[(odom_x_1, odom_y_1),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_3_dist,_ = self.rob_to_rob(odom_x_3, odom_y_3,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_4, odom_y_4),
                                                         (odom_x_5, odom_y_5)])
        
        rob_4_dist,_ = self.rob_to_rob(odom_x_4, odom_y_4,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_5, odom_y_5)])
        
        rob_5_dist,_ = self.rob_to_rob(odom_x_5, odom_y_5,[(odom_x_1, odom_y_1),
                                                         (odom_x_2, odom_y_2),
                                                         (odom_x_3, odom_y_3),
                                                         (odom_x_4, odom_y_4)])
        
        rob_1_vel = [0.0,
                     0.0,
                     0.0,
                     0.0]
        
        rob_2_vel = [0.0,
                     0.0,
                     0.0,
                     0.0]
        
        rob_3_vel = [0.0,
                     0.0,
                     0.0,
                     0.0]
        
        rob_4_vel = [0.0,
                     0.0,
                     0.0,
                     0.0]
        
        rob_5_vel = [0.0,
                     0.0,
                     0.0,
                     0.0]

        # distance = np.linalg.norm(
        #     [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        # )

        # skew_x = self.goal_x - self.odom_x
        # skew_y = self.goal_y - self.odom_y

        # dot = skew_x * 1 + skew_y * 0
        # mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        # mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        # beta = math.acos(dot / (mag1 * mag2))

        # if skew_y < 0:
        #     if skew_x < 0:
        #         beta = -beta
        #     else:
        #         beta = 0 - beta
        # theta = beta - angle

        # if theta > np.pi:
        #     theta = np.pi - theta
        #     theta = -np.pi - theta
        # if theta < -np.pi:
        #     theta = -np.pi - theta
        #     theta = np.pi - theta

        robot_state_1 = [distance_1, theta_1, 0,0] + rob_1_dist + rob_1_vel
        robot_state_2 = [distance_2, theta_2, 0,0] + rob_2_dist + rob_2_vel
        robot_state_3 = [distance_3, theta_3, 0,0] + rob_3_dist + rob_3_vel
        robot_state_4 = [distance_4, theta_4, 0,0] + rob_4_dist + rob_4_vel
        robot_state_5 = [distance_5, theta_5, 0,0] + rob_5_dist + rob_5_vel

        # robot_state = robot_state_1 + robot_state_2 + robot_state_3 + robot_state_4 + robot_state_5
        # laser_state = laser_state_1 + laser_state_2 + laser_state_3 + laser_state_4 + laser_state_5

        state_1 = np.append(change_laser_distance(laser_state_1), robot_state_1)
        state_2 = np.append(change_laser_distance(laser_state_2), robot_state_2)
        state_3 = np.append(change_laser_distance(laser_state_3), robot_state_3)
        state_4 = np.append(change_laser_distance(laser_state_4), robot_state_4)
        state_5 = np.append(change_laser_distance(laser_state_5), robot_state_5)
        
        states = [state_1,state_2,state_3,state_4,state_5]

        # print(state)
        return states

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
            # print(laser_data)
            env.get_logger().info("Collision is detected!")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, lin_vel, ang_vel, min_laser,distance,theta,old_distance,old_theta):
        calc_reward = 0

        # Reward for reaching the target
        if target:
            env.get_logger().info("reward 500 for reaching target")
            calc_reward= 500.0

        # Penalty for collision
        elif collision:
            env.get_logger().info("penalty -100 for collision")
            calc_reward= -100.0

        # Small penalty for non-zero actions to encourage efficient movement
        else:
            if lin_vel > 0.05:
                calc_reward -= 5
            if ang_vel != 0:
                calc_reward -= 1

            # Reward for moving closer to the target
            if distance < old_distance:
                calc_reward += 5
            if theta < old_theta:
                calc_reward += 1

        old_distance = distance
        old_theta = theta

        return calc_reward/10.0, old_distance,old_theta
         
        # else:
        #     r3 = lambda x: 1 - x if x < 1 else 0.0
        #     return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
    
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

class Odom_subscriber_rob_1(Node):

    def __init__(self):
        super().__init__('odom_subscriber2_1')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_rob_1',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom_rob_1
        last_odom_rob_1 = od_data
        # print('LASZ ODOM',last_odom2)

class Odom_subscriber_rob_2(Node):

    def __init__(self):
        super().__init__('odom_subscriber2_2')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_rob_2',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom_rob_2
        last_odom_rob_2 = od_data
        # print('LASZ ODOM',last_odom2)
    
class Odom_subscriber_rob_3(Node):

    def __init__(self):
        super().__init__('odom_subscriber2_3')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_rob_3',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom_rob_3
        last_odom_rob_3 = od_data
        # print('LASZ ODOM',last_odom2)

class Odom_subscriber_rob_4(Node):

    def __init__(self):
        super().__init__('odom_subscriber2_4')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_rob_4',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom_rob_4
        last_odom_rob_4 = od_data
        # print('LASZ ODOM',last_odom2)

class Odom_subscriber_rob_5(Node):

    def __init__(self):
        super().__init__('odom_subscriber2_5')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom_rob_5',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom_rob_5
        last_odom_rob_5 = od_data
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

class Laser_subscriber_rob_1(Node):

    def __init__(self):
        super().__init__('scan_subscriber_1')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2_rob_1',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData_1
        global initial_scan_data_1
        global flag_ls_data_1
        scanData_1 = od_data.ranges
        if flag_ls_data_1== False:
            initial_scan_data_1 = scanData_1
            flag_ls_data_1 = True
        # print(last_odom)

class Laser_subscriber_rob_2(Node):

    def __init__(self):
        super().__init__('scan_subscriber_2')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2_rob_2',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData_2
        global initial_scan_data_2
        global flag_ls_data_2
        scanData_2 = od_data.ranges
        if flag_ls_data_2== False:
            initial_scan_data_2 = scanData_2
            flag_ls_data_2 = True
        # print(last_odom)

class Laser_subscriber_rob_3(Node):

    def __init__(self):
        super().__init__('scan_subscriber_3')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2_rob_3',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData_3
        global initial_scan_data_3
        global flag_ls_data_3
        scanData_3 = od_data.ranges
        if flag_ls_data_3== False:
            initial_scan_data_3 = scanData_3
            flag_ls_data_3 = True
        # print(last_odom)

class Laser_subscriber_rob_4(Node):

    def __init__(self):
        super().__init__('scan_subscriber_4')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2_rob_4',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData_4
        global initial_scan_data_4
        global flag_ls_data_4
        scanData_4 = od_data.ranges
        if flag_ls_data_4== False:
            initial_scan_data_4 = scanData_4
            flag_ls_data_4 = True
        # print(last_odom)

class Laser_subscriber_rob_5(Node):

    def __init__(self):
        super().__init__('scan_subscriber_5')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_2_rob_5',
            self.scan_callback,
            10)
        self.subscription

    def scan_callback(self, od_data):
        global scanData_5
        global initial_scan_data_5
        global flag_ls_data_5
        scanData_5 = od_data.ranges
        if flag_ls_data_5== False:
            initial_scan_data_5 = scanData_5
            flag_ls_data_5 = True
        # print(last_odom)

class Point_cloud_subscriber_rob_1(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_1')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2_rob_1",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data_1
        global initial_point_cloud_data_1
        global flag_pc_data_1
        
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
        # print(point_cloud_data_1,"DATA CHECK")
        if flag_pc_data_1== False:
            initial_point_cloud_data_1 = point_cloud_data_1
            flag_pc_data_1 = True
        
class Point_cloud_subscriber_rob_2(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_2')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2_rob_2",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data_2
        global initial_point_cloud_data_2
        global flag_pc_data_2
        
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
        # print(point_cloud_data_2,"DATA CHECK")
        if flag_pc_data_2== False:
            initial_point_cloud_data_2 = point_cloud_data_2
            flag_pc_data_2 = True
        
    
class Point_cloud_subscriber_rob_3(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_3')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2_rob_3",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data_3
        global initial_point_cloud_data_3
        global flag_pc_data_3
        
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
        # print(point_cloud_data_3,"DATA CHECK")
        if flag_pc_data_3== False:
            initial_point_cloud_data_3 = point_cloud_data_3
            flag_pc_data_3 = True
        

class Point_cloud_subscriber_rob_4(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_4')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2_rob_4",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data_4
        global initial_point_cloud_data_4
        global flag_pc_data_4
        
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
        if flag_pc_data_4== False:
            initial_point_cloud_data_4 = point_cloud_data_4
            flag_pc_data_4 = True
        

class Point_cloud_subscriber_rob_5(Node):

    def __init__(self):
        super().__init__('point_cloud_subscriber_5')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/point_cloud_2_rob_5",
            self.point_cloud_callback,
            10)
        self.subscription
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

    def point_cloud_callback(self, v):
        global point_cloud_data_5
        global initial_point_cloud_data_5
        global flag_pc_data_5
        
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        if(len(data)!=0):
            # print(data,"COLLISION WOULD HAPPEBN")
            point_cloud_data_5 = np.ones(environment_dim) * 10
            data=np.array(data)
            # print(v)
            # print(data.size,"SIZE OF DATA")
            filtered_data = data[data[:, 2] > -0.2]
            data=filtered_data
            gaps=np.array(self.gaps)
            # print(len(gaps),'GAPS',point_cloud_data_5_2,'PCD')

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

            # Update point_cloud_data_5
            for i in range(len(filtered_data)):
                point_cloud_data_5[gap_indices[i]] = min(point_cloud_data_5[gap_indices[i]], dist[i])
        # print(point_cloud_data_5,"DATA CHECK")
        if flag_pc_data_5== False:
            initial_point_cloud_data_5 = point_cloud_data_5
            flag_pc_data_5 = True

def detect_monitor_files(training_dir):
    return [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.startswith('openaigym')]


def clear_monitor_files(training_dir):
    files = detect_monitor_files(training_dir)
    if len(files) == 0:
        return
    for file in files:
        print(file)
        os.unlink(file)

def get_action(fwd_action):
    action=[0,0]
    if fwd_action == 0:
        action[0] = 1
        action[1] = 0
    elif fwd_action == 1:
        action[0] = 0.1
        action[1] = 1
    elif fwd_action == 2:
        action[0] = 0.1
        action[1] = -1
    return action

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
    path = './training_results'
    outdir = './dqlearn'

    continue_execution = False
    resume_epoch = '1000'
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = outdir
    params_json = resume_path + '.json'

    writer_1 = SummaryWriter("./logger1")
    writer_2 = SummaryWriter("./logger2")
    writer_3 = SummaryWriter("./logger3")
    writer_4 = SummaryWriter("./logger4")
    writer_5 = SummaryWriter("./logger5")
    performance_tracker = PerformanceTracker()

    # layout = {
    #     "Average Q":{
    #         "Average Q": ["Multiline",["Avg_Q_1","Avg_Q_2","Avg_Q_3","Avg_Q_4","Avg_Q_5","Cumulative_Avg_Q"]]        
    #     },
    #     "Max Q":{
    #         "Max Q": ["Multiline",["Max_Q_1","Max_Q_2","Max_Q_3","Max_Q_4","Max_Q_5","Cumulative_Max_Q"]]
    #     },
    #     "Distances":{
    #         "Distances": ["Multiline",["Distance_1","Distance_2","Distance_3","Distance_4","Distance_5","Average_Distance"]]
    #     },
    #     "Thetas":{
    #         "Thetas": ["Multiline",["Theta_1","Theta_2","Theta_3","Theta_4","Theta_5","Aevrage_Theta"]]
    #     }
    # }
    # writer.add_custom_scalars(layout)

    env = Env()
    odom_subscriber_1 = Odom_subscriber_rob_1()
    odom_subscriber_2 = Odom_subscriber_rob_2()
    odom_subscriber_3 = Odom_subscriber_rob_3()
    odom_subscriber_4 = Odom_subscriber_rob_4()
    odom_subscriber_5 = Odom_subscriber_rob_5()

    velodyne_subscriber_1 = Point_cloud_subscriber_rob_1()
    velodyne_subscriber_2 = Point_cloud_subscriber_rob_2()
    velodyne_subscriber_3= Point_cloud_subscriber_rob_3()
    velodyne_subscriber_4 = Point_cloud_subscriber_rob_4()
    velodyne_subscriber_5 = Point_cloud_subscriber_rob_5()

    laser_subscriber_1 = Laser_subscriber_rob_1()
    laser_subscriber_2 = Laser_subscriber_rob_2()
    laser_subscriber_3 = Laser_subscriber_rob_3()
    laser_subscriber_4 = Laser_subscriber_rob_4()
    laser_subscriber_5 = Laser_subscriber_rob_5()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber_1)
    executor.add_node(odom_subscriber_2)
    executor.add_node(odom_subscriber_3)
    executor.add_node(odom_subscriber_4)
    executor.add_node(odom_subscriber_5)

    executor.add_node(velodyne_subscriber_1)
    executor.add_node(velodyne_subscriber_2)
    executor.add_node(velodyne_subscriber_3)
    executor.add_node(velodyne_subscriber_4)
    executor.add_node(velodyne_subscriber_5)
    
    executor.add_node(laser_subscriber_1)
    executor.add_node(laser_subscriber_2)
    executor.add_node(laser_subscriber_3)
    executor.add_node(laser_subscriber_4)
    executor.add_node(laser_subscriber_5)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = odom_subscriber_1.create_rate(2)
    rate = odom_subscriber_2.create_rate(2)
    rate = odom_subscriber_3.create_rate(2)
    rate = odom_subscriber_4.create_rate(2)
    rate = odom_subscriber_5.create_rate(2)

    if not continue_execution:
        epochs = 50000
        steps = 2000
        updateTargetNetwork = 10000
        explorationRate = 1
        epsilon_min = 0.05
        minibatch_size = 128
        learnStart = 128
        learningRate = 0.0001
        discountFactor = 0.95
        memorySize = 1000000
        network_inputs = 64
        network_outputs = 3
        network_structure = [300,300]
        current_epoch = 0
        epsilon_discount = 0.995
        update_frequency = 4
        
        deepQ_1 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_1.initNetworks(network_structure)
        deepQ_2 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_2.initNetworks(network_structure)
        deepQ_3 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_3.initNetworks(network_structure)
        deepQ_4 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_4.initNetworks(network_structure)
        deepQ_5 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_5.initNetworks(network_structure)

    else:
        with open(params_json) as outfile:
            d = json.load(outfile)
            epochs = d.get('epochs') + 500
            steps = d.get('steps')
            updateTargetNetwork = d.get('updateTargetNetwork')
            epsilon_min = 0.01
            explorationRate = d.get('explorationRate')
            minibatch_size = d.get('minibatch_size')
            learnStart = d.get('learnStart')
            learningRate = d.get('learningRate')
            discountFactor = d.get('discountFactor')
            memorySize = d.get('memorySize')
            network_inputs = d.get('network_inputs')
            network_outputs = d.get('network_outputs')
            network_structure = d.get('network_structure')
            current_epoch = d.get('current_epoch')
            epsilon_discount = 0.995

        deepQ_1 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_1.initNetworks(network_structure, training=True)
        deepQ_1.loadWeights(weights_path)

        deepQ_2 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_2.initNetworks(network_structure, training=True)
        deepQ_2.loadWeights(weights_path)

        deepQ_3 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_3.initNetworks(network_structure, training=True)
        deepQ_3.loadWeights(weights_path)

        deepQ_4 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_4.initNetworks(network_structure, training=True)
        deepQ_4.loadWeights(weights_path)

        deepQ_5 = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ_5.initNetworks(network_structure, training=True)
        deepQ_5.loadWeights(weights_path)

        clear_monitor_files(outdir)
        copy_tree(monitor_path, outdir)

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False
    
    stepCounter = 0
    highest_reward = 0
    start_time = time.time()
    successes = 0
    failures = 0

    highest_reward = -np.inf
    highest_reward_1 = -np.inf
    highest_reward_2 = -np.inf
    highest_reward_3 = -np.inf
    highest_reward_4 = -np.inf
    highest_reward_5 = -np.inf
    try:
        for epoch in range(current_epoch + 1, epochs + 1):

            episode_step = 0
            failure_episode_1 = False
            failure_episode_2 = False
            failure_episode_3 = False
            failure_episode_4 = False
            failure_episode_5 = False

            success_episode_1 = False
            success_episode_2 = False
            success_episode_3 = False
            success_episode_4 = False
            success_episode_5 = False

            episode_step = 0
            # Initialize the environment and get first state of the robot
            if(performance_tracker.has_reached_threshold()):
                observations = env.reset(True,False)
            else:
                observations = env.reset(False,False)
            # time.sleep(0.1)  # Give time for RL to reset the agent's position
            start_time = time.time()
            # env.done = False
            state_1 = observations[0]
            state_2 = observations[1]
            state_3 = observations[2]
            state_4 = observations[3]
            state_5 = observations[4]
            

            add_total_q_value=0
            add_q_value_count=0
            add_total_loss=0
            add_loss_count=0
            add_max_q_value = -np.inf

            add_total_q_value_1=0
            add_q_value_count_1=0
            add_total_loss_1=0
            add_loss_count_1=0
            add_max_q_value_1 = -np.inf

            add_total_q_value_2=0
            add_q_value_count_2=0
            add_total_loss_2=0
            add_loss_count_2=0
            add_max_q_value_2 = -np.inf

            add_total_q_value_3=0
            add_q_value_count_3=0
            add_total_loss_3=0
            add_loss_count_3=0
            add_max_q_value_3 = -np.inf

            add_total_q_value_4=0
            add_q_value_count_4=0
            add_total_loss_4=0
            add_loss_count_4=0
            add_max_q_value_4 = -np.inf

            add_total_q_value_5=0
            add_q_value_count_5=0
            add_total_loss_5=0
            add_loss_count_5=0
            add_max_q_value_5 = -np.inf

            done_1 = False
            done_2 = False
            done_3 = False
            done_4 = False
            done_5 = False

            distances=[]
            thetas=[]
            cumulated_reward = 0
            cumulated_reward_1 = 0
            cumulated_reward_2 = 0
            cumulated_reward_3 = 0
            cumulated_reward_4 = 0
            cumulated_reward_5 = 0

            while not (done_1 or done_2 or done_3 or done_4 or done_5) and episode_step <= steps:
                state_1 = np.float32(state_1)
                state_2 = np.float32(state_2)
                state_3 = np.float32(state_3)
                state_4 = np.float32(state_4)
                state_5 = np.float32(state_5)

                qValues_1 = deepQ_1.getQValues(state_1)
                qValues_2 = deepQ_1.getQValues(state_2)
                qValues_3 = deepQ_1.getQValues(state_3)
                qValues_4 = deepQ_1.getQValues(state_4)
                qValues_5 = deepQ_1.getQValues(state_5)
                
                fwd_action_1 = deepQ_1.selectAction(qValues_1, explorationRate)
                fwd_action_2 = deepQ_2.selectAction(qValues_2, explorationRate)
                fwd_action_3 = deepQ_3.selectAction(qValues_3, explorationRate)
                fwd_action_4 = deepQ_4.selectAction(qValues_4, explorationRate)
                fwd_action_5 = deepQ_5.selectAction(qValues_5, explorationRate)


                action_1 = get_action(fwd_action_1)
                action_2 = get_action(fwd_action_2)
                action_3 = get_action(fwd_action_3)
                action_4 = get_action(fwd_action_4)
                action_5 = get_action(fwd_action_5)

                observations, rewards, dones, infos,distances,thetas = env.step(action_1,action_2,action_3,action_4,action_5)
                    
                new_observation_1 = observations[0]
                new_observation_2 = observations[1]
                new_observation_3 = observations[2]
                new_observation_4 = observations[3]
                new_observation_5 = observations[4]
                
                reward_1 = rewards[0]
                reward_2 = rewards[1]
                reward_3 = rewards[2]
                reward_4 = rewards[3]
                reward_5 = rewards[4]

                info_1 = infos[0]
                info_2 = infos[1]
                info_3 = infos[2]
                info_4 = infos[3]
                info_5 = infos[4]

                done_1 = dones[0]
                done_2 = dones[1]
                done_3 = dones[2]
                done_4 = dones[3]
                done_5 = dones[4]

                if not(done_1 or done_2 or done_3 or done_4 or done_5) and (episode_step+1 > steps):
                    done=True
                    reward_1+=-100
                    reward_2+=-100
                    reward_3+=-100
                    reward_4+=-100
                    reward_5+=-100

                cumulated_reward_1 += reward_1
                cumulated_reward_2 += reward_2
                cumulated_reward_3 += reward_3
                cumulated_reward_4 += reward_4
                cumulated_reward_5 += reward_5

                cumulated_reward = cumulated_reward_1 + cumulated_reward_2 + cumulated_reward_3 + cumulated_reward_4 + cumulated_reward_5
                
                highest_reward_1 = max(highest_reward_1, cumulated_reward_1)
                highest_reward_2 = max(highest_reward_2, cumulated_reward_2)
                highest_reward_3 = max(highest_reward_3, cumulated_reward_3)
                highest_reward_4 = max(highest_reward_4, cumulated_reward_4)
                highest_reward_5 = max(highest_reward_5, cumulated_reward_5)
                
                highest_reward = max(highest_reward_1,highest_reward_2,highest_reward_3,highest_reward_4,highest_reward_5)

                deepQ_1.addMemory(state_1, fwd_action_1, reward_1, new_observation_1, done_1)
                deepQ_2.addMemory(state_2, fwd_action_2, reward_2, new_observation_2, done_2)
                deepQ_3.addMemory(state_3, fwd_action_3, reward_3, new_observation_3, done_3)
                deepQ_4.addMemory(state_4, fwd_action_4, reward_4, new_observation_4, done_4)
                deepQ_5.addMemory(state_5, fwd_action_5, reward_5, new_observation_5, done_5)

                if stepCounter >= learnStart and stepCounter % update_frequency == 0:
                    total_q_value_1,q_value_count_1,total_loss_1,loss_count_1,max_q_value_1 = deepQ_1.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value_1 += total_q_value_1
                    add_q_value_count_1 += q_value_count_1
                    add_total_loss_1 += total_loss_1
                    add_loss_count_1 += loss_count_1
                    add_max_q_value_1 = max(max_q_value_1,add_max_q_value_1)

                    total_q_value_2,q_value_count_2,total_loss_2,loss_count_2,max_q_value_2 = deepQ_2.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value_2 += total_q_value_2
                    add_q_value_count_2 += q_value_count_2
                    add_total_loss_2 += total_loss_2
                    add_loss_count_2 += loss_count_2
                    add_max_q_value_2 = max(max_q_value_2,add_max_q_value_2)

                    total_q_value_3,q_value_count_3,total_loss_3,loss_count_3,max_q_value_3 = deepQ_3.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value_3 += total_q_value_3
                    add_q_value_count_3 += q_value_count_3
                    add_total_loss_3 += total_loss_3
                    add_loss_count_3 += loss_count_3
                    add_max_q_value_3 = max(max_q_value_3,add_max_q_value_3)

                    total_q_value_4,q_value_count_4,total_loss_4,loss_count_4,max_q_value_4 = deepQ_4.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value_4 += total_q_value_4
                    add_q_value_count_4 += q_value_count_4
                    add_total_loss_4 += total_loss_4
                    add_loss_count_4 += loss_count_4
                    add_max_q_value_4 = max(max_q_value_4,add_max_q_value_4)

                    total_q_value_5,q_value_count_5,total_loss_5,loss_count_5,max_q_value_5 = deepQ_5.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value_5 += total_q_value_5
                    add_q_value_count_5 += q_value_count_5
                    add_total_loss_5 += total_loss_5
                    add_loss_count_5 += loss_count_5
                    add_max_q_value_5 = max(max_q_value_5,add_max_q_value_5)
                    
                    # total_q_value_5,q_value_count_5,total_loss_5,loss_count_5,max_q_value_5 = deepQ_5.learnOnMiniBatch(minibatch_size, useTargetNetwork=(stepCounter > updateTargetNetwork))
                    add_total_q_value += total_q_value_1 + total_q_value_2 + total_q_value_3 + total_q_value_4 + total_q_value_5
                    add_q_value_count += q_value_count_1 + q_value_count_2 + q_value_count_3 + q_value_count_4 + q_value_count_5
                    add_total_loss += total_loss_1 + total_loss_2 + total_loss_3 + total_loss_4 + total_loss_5
                    add_loss_count += loss_count_1 + loss_count_2 + loss_count_3 + loss_count_4 + loss_count_5
                    add_max_q_value = max(max_q_value_1,add_max_q_value_2,
                                          max_q_value_2,add_max_q_value_2,
                                          max_q_value_3,add_max_q_value_3,
                                          max_q_value_4,add_max_q_value_4,
                                          max_q_value_5,add_max_q_value_5)

                state_1 = new_observation_1
                state_2 = new_observation_2
                state_3 = new_observation_3
                state_4 = new_observation_4
                state_5 = new_observation_5

                if done_1 or done_2 or done_3 or done_4 or done_5:
                    if (info_1 or info_2 or info_3 or info_4 or info_5):
                        successes+=1
                        performance_tracker.add_result(1)
                    else:
                        failures+=1
                        performance_tracker.add_result(0)

                    last100Scores[last100ScoresIndex] = episode_step
                    last100ScoresIndex = (last100ScoresIndex + 1) % 100
                    last100Filled = last100Filled or last100ScoresIndex == 0

                    if last100Filled:
                        avg_steps = sum(last100Scores) / len(last100Scores)
                        elapsed_time = time.time() - start_time
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward_1} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward_2} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward_3} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward_4} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward_5} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Last 100 Steps: {avg_steps} - Reward: {cumulated_reward} - Exploration: {round(explorationRate, 2)} - Time: {elapsed_time:.2f} seconds")
                    else:
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward_1} - Exploration: {round(explorationRate, 2)}")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward_2} - Exploration: {round(explorationRate, 2)}")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward_3} - Exploration: {round(explorationRate, 2)}")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward_4} - Exploration: {round(explorationRate, 2)}")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward_5} - Exploration: {round(explorationRate, 2)}")
                        print(f"EP {epoch} - {episode_step + 1}/{steps} steps - Reward: {cumulated_reward} - Exploration: {round(explorationRate, 2)}")

                    if epoch % 100 == 0:
                        deepQ_1.saveModel(path + str(epoch) + '.h5')
                        deepQ_2.saveModel(path + str(epoch) + '.h5')
                        deepQ_3.saveModel(path + str(epoch) + '.h5')
                        deepQ_4.saveModel(path + str(epoch) + '.h5')
                        deepQ_5.saveModel(path + str(epoch) + '.h5')
                        copy_tree(outdir, str(epoch))
                        params = {
                            'epochs': epochs, 'steps': steps, 'updateTargetNetwork': updateTargetNetwork,
                            'explorationRate': explorationRate, 'minibatch_size': minibatch_size, 'learnStart': learnStart,
                            'learningRate': learningRate, 'discountFactor': discountFactor, 'memorySize': memorySize,
                            'network_inputs': network_inputs, 'network_outputs': network_outputs,
                            'network_structure': network_structure, 'current_epoch': epoch
                        }
                        with open(path + str(epoch) + '.json', 'w') as outfile:
                            json.dump(params, outfile)
                    
                    distance_1 = distances[0]
                    distance_2 = distances[1]
                    distance_3 = distances[2]
                    distance_4 = distances[3]
                    distance_5 = distances[4]
                    cum_distances= distance_1 + distance_2 + distance_3 + distance_4 + distance_5

                    theta_1 = thetas[0]
                    theta_2 = thetas[1]
                    theta_3 = thetas[2]
                    theta_4 = thetas[3]
                    theta_5 = thetas[4]
                    cum_thetas= theta_1 + theta_2 + theta_3 + theta_4 + theta_5

                    writer_1.add_scalar(f'Cumulated_Reward', cumulated_reward_1, epoch)
                    writer_2.add_scalar(f'Cumulated_Reward', cumulated_reward_2, epoch)
                    writer_3.add_scalar(f'Cumulated_Reward', cumulated_reward_3, epoch)
                    writer_4.add_scalar(f'Cumulated_Reward', cumulated_reward_4, epoch)
                    writer_5.add_scalar(f'Cumulated_Reward', cumulated_reward_5, epoch)
                    writer_1.add_scalar("Cumulative_Avg_Reward", cumulated_reward/5, epoch)

                    writer_1.add_scalar(f'Distance',distance_1,epoch)
                    writer_2.add_scalar(f'Distance',distance_2,epoch)
                    writer_3.add_scalar(f'Distance',distance_3,epoch)
                    writer_4.add_scalar(f'Distance',distance_4,epoch)
                    writer_5.add_scalar(f'Distance',distance_5,epoch)
                    writer_1.add_scalar("Avg_Distance",cum_distances/5,epoch)
                    writer_1.add_scalar("Best_Distance", min(distances), epoch)

                    writer_1.add_scalar(f'Theta',theta_1,epoch)
                    writer_2.add_scalar(f'Theta',theta_2,epoch)
                    writer_3.add_scalar(f'Theta',theta_3,epoch)
                    writer_4.add_scalar(f'Theta',theta_4,epoch)
                    writer_5.add_scalar(f'Theta',theta_5,epoch)
                    writer_1.add_scalar("Avg_Theta",cum_thetas/5,epoch)
                    writer_1.add_scalar("Best_Theta", min(abs(thetas[0]),abs(thetas[1]),abs(thetas[2]),abs(thetas[3]),abs(thetas[4])), (epoch+1))

                if add_q_value_count > 0:
                    av_qvalue_1 = add_total_q_value_1/add_q_value_count_1
                    av_qvalue_2 = add_total_q_value_2/add_q_value_count_2
                    av_qvalue_3 = add_total_q_value_3/add_q_value_count_3
                    av_qvalue_4 = add_total_q_value_4/add_q_value_count_4
                    av_qvalue_5 = add_total_q_value_5/add_q_value_count_5
                    av_qvalue = add_total_q_value/add_q_value_count

                    # writer.add_scalar("Avg_Q_1", av_qvalue_1, epoch)
                    # writer.add_scalar("Avg_Q_2", av_qvalue_2, epoch)
                    # writer.add_scalar("Avg_Q_3", av_qvalue_3, epoch)
                    # writer.add_scalar("Avg_Q_4", av_qvalue_4, epoch)
                    # writer.add_scalar("Avg_Q_5", av_qvalue_5, epoch)
                    # writer.add_scalar("Cumulative_Avg_Q", av_qvalue, epoch)

                

                # writer.add_scalar("Max_Q_1", add_max_q_value_1, epoch)
                # writer.add_scalar("Max_Q_2", add_max_q_value_2, epoch)
                # writer.add_scalar("Max_Q_3", add_max_q_value_3, epoch)
                # writer.add_scalar("Max_Q_4", add_max_q_value_4, epoch)
                # writer.add_scalar("Max_Q_5", add_max_q_value_5, epoch)
                # writer.add_scalar("Cumulativ_Max_Q", add_max_q_value, epoch)
            
                
                if add_loss_count_1 > 0:
                    av_loss_1 = add_total_loss_1/add_loss_count_1
                    writer_1.add_scalar(f'loss',av_loss_1, epoch)
                if add_loss_count_2 > 0:
                    av_loss_2 = add_total_loss_2/add_loss_count_2
                    writer_2.add_scalar(f'loss',av_loss_2, epoch)
                if add_loss_count_3 > 0:
                    av_loss_3 = add_total_loss_3/add_loss_count_3
                    writer_3.add_scalar(f'loss',av_loss_3, epoch)
                if add_loss_count_4 > 0:
                    av_loss_4 = add_total_loss_4/add_loss_count_4
                    writer_4.add_scalar(f'loss',av_loss_4, epoch)
                if add_loss_count_5 > 0:
                    av_loss_5 = add_total_loss_5/add_loss_count_5
                    writer_5.add_scalar(f'loss',av_loss_5, epoch)

                if(add_loss_count_1>0) and (add_loss_count_2>0) and (add_loss_count_3>0) and (add_loss_count_4>0) and (add_loss_count_5>0): 
                    cumulated_loss= av_loss_1 + av_loss_2 + av_loss_3 + av_loss_4 + av_loss_5
                    writer_1.add_scalar("Avg_loss", cumulated_loss/5, epoch)

                writer_1.add_scalar("Success",successes, epoch)
                writer_1.add_scalar("Collision",failures,epoch)
                writer_1.flush()
                writer_2.flush()
                writer_3.flush()
                writer_4.flush()
                writer_5.flush()
                stepCounter += 1

                if stepCounter % updateTargetNetwork == 0:
                    deepQ_1.updateTargetNetwork()
                    deepQ_2.updateTargetNetwork()
                    deepQ_3.updateTargetNetwork()
                    deepQ_4.updateTargetNetwork()
                    deepQ_5.updateTargetNetwork()

                episode_step += 1

            explorationRate = max(epsilon_min, explorationRate * epsilon_discount)

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()