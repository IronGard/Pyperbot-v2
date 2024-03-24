'''
Primary Changes to Prev Environment:
* Uses StandardSnakebot class to generate snakebot and goals
* Reward Clipping/Observation Clipping?
* Repaired termination conditions
* Import new terrains
* Repair seeding
'''
# import gym
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from ..snakebot_description.updated_snakebot_class import UpdatedSnakeBot
from ..resources.goal import Goal, SoloGoal
from ..resources.plane import Plane
from ..resources.maze import Maze
from ..resources.lab import Lab
from ..resources.maze_small import MazeSmall
from ..resources.plane_small import PlaneSmall
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import logging 

#framework inspired by code provided by stable baselines.
logger = logging.getLogger(__name__)

#setting up the environment
class StandardTestEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self):
        #initialise environment
        #normalised action space for each joint - need to adjust the joint limits for the other snake robot
        super(StandardTestEnv, self).__init__()
        self.action_space = gym.spaces.Box(low = -1, 
                                           high = 1,
                                           shape = (20,), 
                                           dtype = np.float32) #20 joints in the snakebot (to be printed + appended to a CSV file)
        #Observation space - 7 dimensional array - x,y,z position and orientation of base, remaining distance to goal, and velocity of the robot
        #TODO: add VecNormalize to normalise observations before feeding to PPO agent.
        self.observation_space = gym.spaces.Box(
            low = np.array([-200, -200, -200, -3.1415, -3.1415, -3.1415, -100]), #first three are base position, next three are base orientation, next is velocity, last is distance to goal
            high = np.array([200, 200, 200, 3.1415, 3.1415, 3.1415, 100]),
            dtype = np.float32
        )
        #Additional Params for the environment.
        self._snake = None
        self._client = None
        self._finish_con = 0.5
        self._goal = None
        self._done = None
        self._env = None
        self.prev_dist_to_goal = 0
        self.rendered_img = None
        self.rot_matrix = None
        self.current_step = 0
        self.max_episode_steps = 100000
        self.reward = 0
        self.seed()
        self.reset()

    def step(self, action):
        '''
        We quantify reward here based on the remaining distance to the goal. The distance
        is calculated using get_dist_to_goal function. 
        #TODO: set condition for done
        '''
        self._snake.apply_action(action) # apply action to the robot
        self._client.stepSimulation() #step the pybullet simulation after a step is taken to update position after action is applied.
        observation = self._snake.get_observation()
        snake_joint_observation = self._snake.get_joint_observation()
        dist_to_goal = self.get_dist_to_goal()
        #set the reward based on improvement in distance to goal
        reward = dist_to_goal - self.prev_dist_to_goal
        self._done = False
        self._truncated = False
        #First termination condition - if distance to goal is less than finish condition, set reward to 10000 and end episode
        if dist_to_goal < self._finish_con:
            reward += 1000
            self._done = True
        #second termination condition - if snake falls off plane, set reward to -1000 and end episode
        elif self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][2] < 0:
            reward -= 1000
            self._done = True
        #if snake falls off boundaries in x and y axes, set reward to -1000 and end episode
        elif (self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0] < -10 or
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0] > 10 or 
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1] < -10 or 
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1] > 10):
            reward -= 1000
            self._done = True
        #check truncation condition
        elif self.current_step >= self.max_episode_steps:
            self._done = True
            self._truncated = True
            reward -= 1000
        #set prev distance to goal to current distance to goal after update
        self.prev_dist_to_goal = dist_to_goal
        #increment current step
        self.current_step += 1
        #check information needed for vectorised environments + add KL divergence for dictionary to monitor training process.
        return (observation, reward, self._done, self._truncated, {"obs": snake_joint_observation, "distance_to_goal": dist_to_goal, "reward": reward, "done": self._done})

    def seed(self, seed = None):
        #Generate seed for the environment
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        #cast seed to be smaller
        self.seed_value = seed
        return [seed]
        
    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return math.hypot(self._goal.get_goal_pos()[0] - self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0], self._goal.get_goal_pos()[1] - self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1])
    
    def reset(self, seed = None, env = "maze"):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        self.current_step = 0
        self._client = bc.BulletClient(connection_mode = p.DIRECT) #connect to pybullet client
        # self._client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._client.setRealTimeSimulation(0) 
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.resetSimulation() #reset the simulation
        self._client.setGravity(0, 0, -9.81)
        self._snake = UpdatedSnakeBot(self._client, "pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro", seed = int(self.seed_value))
        self._goal = SoloGoal(self._client, [5, 0, 0])
        if env == "maze":
            self._env = Maze(self._client)
        elif env == "lab":
            self._env = Lab(self._client)
        elif env == "maze-small":
            self._env = MazeSmall(self._client)
        elif env == "plane-small":
            self._env = PlaneSmall(self._client)
        self.plane = Plane(self._client) #insert plane
        self.done = False
        self.prev_dist_to_goal = self.get_dist_to_goal()
        self.joint_ob = self._snake.get_joint_observation()
        base_pos, ori = self._snake.get_base_observation()[0], self._snake.get_base_observation()[1]
        base_velocity, _ = self._client.getBaseVelocity(self._snake.get_ids()[0])
        linear_velocity = np.linalg.norm(base_velocity)
        observation = np.array(list(base_pos) + list(ori) + [linear_velocity], dtype = np.float32)
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
        self._client.disconnect()