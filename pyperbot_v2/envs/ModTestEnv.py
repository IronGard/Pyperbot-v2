import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from ..snakebot_description.snakebot_class_test import Snakebot
from ..resources.goal import Goal
from ..resources.plane import Plane
from ..utils.structure_loader import Loader
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import logging

#generate logger to log results
logger = logging.getLogger(__name__)
#TODO: add integration with Isaac code for the Loader to see if it can work

#setting up the environment
class ModTestEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self):
        #initialise environment
        #action space should actually be continuous - each joint can move in between from 0 and 1 different values 
        #TODO: need to set action space for prismatic and revolute joints, separately
        super(ModTestEnv, self).__init__()
        '''
        Non-normalised action space definition provided below:
        self.action_space = gym.spaces.Box(low = np.array([0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236]), 
                                           high = np.array([0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236]), 
                                           dtype = np.float32) #20 joints in the snakebot (to be printed + appended to a CSV file)
        '''
        #Normalised action space:
        self.action_space = gym.spaces.Box(low = np.array([-1]*20), high = np.array([1]*20), dtype = np.float32)
        #Observation space - 8 dimensional continuous array - x,y,z position and orientation of base, remaining distance to goal, and velocity of the robot
        #we require a general number of the infomration.
        self.observation_space = gym.spaces.Box(
            low = np.array([0, 0, 0, -3.1415, -3.1415, -3.1415, 0, 0]), #first three are base position, next three are base orientation, next is velocity, last is distance to goal
            high = np.array([200, 200, 200, 3.1415, 3.1415, 3.1415, 100, 20]),
            dtype = np.float32
        )
        self.random, _ = gym.utils.seeding.np_random() #setting the seed for the RL environment

        #Additional Params for the environment.
        self._snake = None
        self._client = None
        self._goal = None
        self._done = None
        self._prev_dist_to_goal = None #parameterising the distance remaining to the goal/final reward
        self.rendered_img = None
        self.rot_matrix = None

    def step(self, action):
        '''
        We quantify reward here based on the remaining distance to the goal. The distance
        is calculated using get_dist_to_goal function. 
        #TODO: set condition for done
        '''
        self._snake.apply_action(action) # apply action to the robot
        self._client.stepSimulation() #step the pybullet simulation after a step is taken to update position after action is applied.
        snake_joint_obs = self._snake.get_joint_observation() #here we primarily want the joint positions, not velocities
        base_pos, base_ori = p.getBasePositionAndOrientation(self._snake.get_ids()[0])
        base_ori = p.getEulerFromQuaternion(base_ori) #convert quaternion to euler angles
        dist_to_goal = self.get_dist_to_goal() #obtain distance of the snake remaining from the goal
        #set the reward based on remaining distance to goal
        reward = -dist_to_goal
        #TODO: repair done condition
        #get the basevelocity of the snake
        base_velocity, _ = p.getBaseVelocity(self._snake.get_ids()[0])
        #calculate normalised linear velocity of snake
        linear_velocity = np.linalg.norm(base_velocity)
        observation = np.array(list(base_pos) + list(base_ori) + [linear_velocity, dist_to_goal], dtype = np.float32)
        return (observation, reward, self._done, False, {"obs": snake_joint_obs})
        #change returned observation to be a numpy array instead of a list.

    def seed(self, seed = None):
        #Generate seed for the environment
        self._random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return math.hypot(self._goal.get_goals()[0][0] - p.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0], self._goal.get_goals()[0][1] - p.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1])
    
    def reset(self, seed = None):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        self._client = bc.BulletClient(connection_mode = p.DIRECT) #connect to the pybullet client
        # self._client.configureDebugVisualizer(self._client.COV_ENABLE_GUI, 0) #disable the GUI
        self._client.resetSimulation() #reset the simulation
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath()) #set the search path for the pybullet data
        self._client.setRealTimeSimulation(0) #set the simulation to not run in real time
        self._client.setGravity(0, 0, -9.81)
        self._snake = Snakebot(self._client) #load the snakebot
        self._plane = Plane(self._client._client) #load the plane
        self._goal = Goal(self._client._client, 3) #insert goal in random position
        self._done = False
        self.prev_dist_to_goal = self.get_dist_to_goal() #set prev dist to goal as current dist
        self._client.stepSimulation()
        self.joint_ob = self._snake.get_joint_observation()
        base_pos, ori = self._snake.get_base_observation()[0], self._snake.get_base_observation()[1]
        dist_to_goal = self.get_dist_to_goal()
        base_velocity, _ = p.getBaseVelocity(self._snake.get_ids()[0])
        linear_velocity = np.linalg.norm(base_velocity)
        observation = np.array(list(base_pos) + list(ori) + [linear_velocity, dist_to_goal], dtype = np.float32)
        #TODO: fix to get the actual observation to ensure that the data returned is actually in the observation space
        return (observation, {"obs": self.joint_ob})
    
    def render(self, mode = 'human'):
        '''
        Function to render the environment
        '''
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
        
        base_pos, _ = self._snake.get_base_observation()
        if self._client and self._snake:
            view_matrix = self._client.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = base_pos, distance = 6, yaw = 0, pitch = -10, roll = 0, upAxisIndex = 2)
            proj_matrix = self._client.computeProjectionMatrixFOV(fov = 60, aspect = 1.0, nearVal = 0.1, farVal = 100.0)
            (_, _, px, _, _) = self._client.getCameraImage(width = 100, height = 100, viewMatrix = view_matrix, projectionMatrix = proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL)
            self.rendered_img.set_data(px)
            plt.pause(1e-10)
            plt.draw()
            return np.array(px)
        
        self._client.resetDebugVisualizerCamera(2, 0, -20, base_pos)

    def close(self):
        '''
        Closes simulation by disconnecting from Pybullet client
        '''
        p.disconnect()