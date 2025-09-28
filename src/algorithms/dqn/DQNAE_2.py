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
from deep_q_torch_2 import DeepQ
from distutils.dir_util import copy_tree
import json

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

class Env(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0

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
    def step(self, action):
        global point_cloud_data
        global scanData
        target = False
        done=False
        max_linear = 1.0
        max_angular = 18.0
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
        start_time = time.time()
        while not(time.time() - start_time >=0.2):
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
        done, collision, min_laser = self.observe_collision(point_cloud_data)
        v_state = []
        v_state[:] = scanData[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x * 100 + 1.14435
        self.odom_y = last_odom.pose.pose.position.y * 100 + 312.0143
        # prev_odom_x = self.odom_x
        # prev_odom_y = self.odom_y
        quaternion = Quaternion(
            last_odom.pose.pose.orientation.w,
            last_odom.pose.pose.orientation.x,
            last_odom.pose.pose.orientation.y,
            last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        # prev_yaw = euler[2]
        angle = round(euler[2], 4)
        # prev_time = time.time()

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
        # theta = (theta % (2 * np.pi) + (2 * np.pi)) % (2 * np.pi)

        #this here checks if the speed has been achieved and then continue ahead otherwise dont
        
        # print("out")
        # current_lin_speed = 0
        # current_ang_speed = 0 
        # lin_speed = (0.5 if action[0]!=0 else 0)
        # ang_speed = (3.0 if action[1]!=0 else 0)
        # i=0
        # print(not(current_lin_speed < lin_speed + 0.05) and not(current_lin_speed > lin_speed - 0.05) and lin_speed > 0)
        # while not(current_lin_speed < lin_speed + 0.05) and not(current_lin_speed > lin_speed - 0.05) and lin_speed > 0:
        #     self.simulation_context.step(render=True)  
        #     temp_odom_x = (last_odom.pose.pose.position.x * 100 + 1.14435)
        #     temp_odom_y = (last_odom.pose.pose.position.y * 100 + 312.0143)
        #     temp_time = time.time()
        #     dx = temp_odom_x - prev_odom_x
        #     dy = temp_odom_y - prev_odom_y
        #     dt = temp_time - prev_time
        #     d_dist = np.sqrt(dx**2 + dy**2)
        #     current_lin_speed = d_dist/dt
        #     prev_odom_x = temp_odom_x
        #     prev_odom_y = temp_odom_y
        #     prev_time = temp_time
        #     print(current_lin_speed,i)
        #     i+=1
        
        # # print(not(current_ang_speed < ang_speed + 0.05),not(current_ang_speed > ang_speed - 0.05),ang_speed > 0)
        # i=0
        # while not((current_ang_speed < ang_speed + 0.05) and(current_ang_speed > ang_speed - 0.05)) and ang_speed > 0:
        #     self.simulation_context.step(render=True)
        #     quaternion = Quaternion(
        #         last_odom.pose.pose.orientation.w,
        #         last_odom.pose.pose.orientation.x,
        #         last_odom.pose.pose.orientation.y,
        #         last_odom.pose.pose.orientation.z,
        #     )
        #     euler = quaternion.to_euler(degrees=False)
        #     temp_yaw = euler[2]
        #     temp_time = time.time()
        #     d_yaw = abs(temp_yaw - prev_yaw)
        #     dt = temp_time - prev_time
        #     current_ang_speed =  d_yaw / dt
        #     prev_yaw = temp_yaw
        #     prev_time = temp_time
        #     print(current_ang_speed,i)
        #     i+=1

        # print(theta)
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

        # print(distance,"THE DISTANCE TO TARGET", theta, "THE ANGLE TO THE TARGET")
        self.odometry_history.append([self.odom_x,self.odom_y])
        robot_state = [distance, theta]
        state = np.append(change_laser_distance(laser_state),robot_state)
        repetitive_behavior = self.detect_repetitive_behavior(self.odometry_history)
        reward = self.get_reward(target, collision, vel_cmd.linear.x,vel_cmd.angular.z, min_laser,repetitive_behavior,distance,theta)
        print("The reward for step is:", reward,"THE DISTANCE TO TARGET",distance,"THE ANGLE TO THE TARGET",theta)
        return state, reward, done, target, distance,theta

    def reset(self,change_target):

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
        object_state = self.set_self_state

        # x = 0
        # y = 0
        # position_ok = False
        # while not position_ok:
        #     x = np.random.uniform(-4.5, 4.5)
        #     y = np.random.uniform(-4.5, 4.5)
            # position_ok = check_pos(x, y)
        # object_state.pose.position.x = x
        # object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        # object_state.pose.orientation.x = quaternion.x
        # object_state.pose.orientation.y = quaternion.y
        # object_state.pose.orientation.z = quaternion.z
        # object_state.pose.orientation.w = quaternion.w
        # self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x * 100 + 1.14435
        self.odom_y = object_state.pose.position.y * 100 + 312.0143

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

        v_state = []
        v_state[:] = scanData[:]
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
        # theta = (theta % (2 * np.pi) + (2 * np.pi)) % (2 * np.pi)

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        elif theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta
        
        # theta= (theta + np.pi) % 2*np.pi - np.pi
        robot_state = [distance, theta]
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
    def get_reward(target, collision,lin_vel,ang_vel , min_laser,repetitive_behavior,distance,theta):
        global old_distance
        global old_theta

        calc_reward = 0

        # Reward for reaching the target
        if target:
            env.get_logger().info("reward 200")
            return 500.0

        # Penalty for collision
        if collision:
            env.get_logger().info("reward -200")
            return -200.0

        # Penalty for non-zero actions (to encourage efficient movement)
        # Increased the penalties for non-zero actions
        if lin_vel > 0.05:
            calc_reward += -0.5
        if ang_vel != 0:
            calc_reward += -0.2

        if(distance < old_distance -0.5) and (old_distance-distance <100):
            calc_reward += 1.5
            print("got reward here 1")
        elif(distance > old_distance +0.5):
            calc_reward += -0.5
            print("got reward here 2")

        if ((old_theta < 0 and theta < 0) or (old_theta > 0 and theta > 0)) and (distance <= (old_distance + 0.2)) and (old_distance-distance <100):
            if(abs(theta) < abs(old_theta)):
                calc_reward += 0.6
                print("got reward here 3")
            if(abs(theta) > abs(old_theta)):
                calc_reward += -0.2
                print("got reward here 4")

        old_distance =distance
        old_theta = theta

        return calc_reward

        # if distance != old_distance:
        #     # calc_reward+=0.5
        #     if distance < old_distance:
        #         calc_reward+=0.5
        #     else:
        #         calc_reward-=1.0
        #     old_distance=distance
        # if theta != old_theta: 
        #     # calc_reward+=0.1
        #     current_angle = (theta + np.pi) % (2 * np.pi)
        #     previous_angle = (old_theta + np.pi) % (2 * np.pi)

        #     distance_to_zero_current = min(current_angle, 2 * np.pi - current_angle)
        #     distance_to_zero_previous = min(previous_angle, 2 * np.pi - previous_angle)
        #     if distance_to_zero_current < distance_to_zero_previous:
        #         calc_reward+=0.1
        #     else:
        #         calc_reward-=0.2
        #     old_theta=theta
        # return calc_reward
        # if target:
        #     env.get_logger().info("reward 200")
        #     calc_reward += 200.0
        
        # if collision:
        #     env.get_logger().info("penalty -200")
        #     calc_reward += -200.0
        
        # if distance != old_distance:
        #     calc_reward-= 0.5
        #     rew=lambda x: 1.0 if x <= 60 else (0.0 if x >= 500 else 1 - (x - 60) / 440)
        #     if distance < old_distance:
        #         calc_reward += rew(distance)
        #     else:
        #         calc_reward -= rew(distance)
        #     old_distance=distance
        
        # if theta != old_theta:
        #     calc_reward-= 0.1
            
        #     current_angle = (theta + np.pi) % (2 * np.pi)
        #     previous_angle = (old_theta + np.pi) % (2 * np.pi)

        #     distance_to_zero_current = min(current_angle, 2 * np.pi - current_angle)
        #     distance_to_zero_previous = min(previous_angle, 2 * np.pi - previous_angle)
            
        #     rew=lambda x: 1 if x <= 0.17 else (1 - (x - 0.5) / 2*np.pi)

        #     if distance_to_zero_current <= distance_to_zero_previous:
        #         calc_reward += rew(distance_to_zero_current)*0.2
        #     else:
        #         calc_reward -= rew(distance_to_zero_current)*0.2
        #     old_theta=theta
        
        # return calc_reward
         
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
        
        
        self.gaps = [[0.0, np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        # self.gaps[-1][-1] += 0.03

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

    path = './training_results'
    outdir = './dqlearn'

    continue_execution = False
    resume_epoch = '1000'
    resume_path = path + resume_epoch
    weights_path = resume_path + '.h5'
    monitor_path = outdir
    params_json = resume_path + '.json'

    writer = SummaryWriter()
    performance_tracker = PerformanceTracker()

    env = Env()
    odom_subscriber = Odom_subscriber()
    velodyne_subscriber = Point_cloud_subscriber()
    laser_subscriber = Laser_subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(odom_subscriber)
    executor.add_node(velodyne_subscriber)
    executor.add_node(laser_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = odom_subscriber.create_rate(1000)

    if not continue_execution:
        epochs = 1000
        steps = 2000
        updateTargetNetwork = 10000
        explorationRate = 1
        epsilon_min = 0.01
        minibatch_size = 256
        learnStart = 500
        learningRate = 0.00025
        discountFactor = 0.99
        memorySize = 1000000
        network_inputs = 362
        network_outputs = 3
        network_structure = [300,300]
        current_epoch = 0
        epsilon_discount = 0.995

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure)
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

        deepQ = DeepQ(network_inputs, network_outputs, memorySize, discountFactor, learningRate, learnStart)
        deepQ.initNetworks(network_structure, training=True)
        deepQ.loadWeights(weights_path)
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

    try:
        env.reset(False)
        for i in range(800):
            action=[0.05,1]
            newObservation, reward, done, info,distance,theta = env.step(action)
            print(i)
        for i in range(800):
            action=[0,-1]
            newObservation, reward, done, info,distance,theta = env.step(action)

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()