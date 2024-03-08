import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data

from ..snakebot_description.updated_snakebot_class import UpdatedSnakeBot
from ..resources.goal import Goal
from ..resources.plane import Plane
from ..resources.maze import Maze
from ..resources.lab import Lab
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import logging 

#framework inspired by code provided by stable baselines.
logger = logging.getLogger(__name__)

#setting up the environment
class LoaderTestEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self):
        #initialise environment
        #normalised action space for each joint - need to adjust the joint limits for the other snake robot
        super(LoaderTestEnv, self).__init__()
        self.action_space = gym.spaces.Box(low = -1, 
                                           high = 1,
                                           shape = (20,), 
                                           dtype = np.float32) #20 joints in the snakebot (to be printed + appended to a CSV file)
        #Observation space - 8 dimensional continuous array - x,y,z position and orientation of base, remaining distance to goal, and velocity of the robot
        #we require a general number of the infomration.
        self.observation_space = gym.spaces.Box(
            low = np.array([0, 0, 0, -3.1415, -3.1415, -3.1415, 0]), #first three are base position, next three are base orientation, next is velocity, last is distance to goal
            high = np.array([200, 200, 200, 3.1415, 3.1415, 3.1415, 100]),
            dtype = np.float32
        )
        self.random, _ = gym.utils.seeding.np_random() #setting the seed for the RL environment
        
        #Additional Params for the environment.
        self._snake = None
        self._client = None
        self._finish_con = 0.5
        self._goal = None
        self._done = None
        self._env = None
        self.rendered_img = None
        self.rot_matrix = None
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
        reward = -dist_to_goal
        self._done = False
        if dist_to_goal < self._finish_con:
            reward = 100
            self._done = True
        return (observation, reward, self.done, False, {"obs": snake_joint_observation})

    def seed(self, seed = None):
        #Generate seed for the environment
        self.random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return math.hypot(self._goal.get_goals()[0][0] - self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0], self._goal.get_goals()[0][1] - self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1])
    
    def reset(self, seed = None, env = "maze"):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        self._client = bc.BulletClient(connection_mode = p.DIRECT) #connect to pybullet client
        #self._client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._client.setRealTimeSimulation(0) #set real time simulation to 0
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.resetSimulation() #reset the simulation
        self._client.setGravity(0, 0, -9.81)
        self._snake = UpdatedSnakeBot(self._client, "pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro")
        self._goal = Goal(self._client, 3) #insert goal in random position
        if env == "maze":
            self._env = Maze(self._client)
        elif env == "lab":
            self._env = Lab(self._client)
        self.plane = Plane(self._client) #insert plane
        self.done = False
        self.prev_dist_to_goal = self.get_dist_to_goal()
        self.joint_ob = self._snake.get_joint_observation()
        base_pos, ori = self._snake.get_base_observation()[0], self._snake.get_base_observation()[1]
        dist_to_goal = self.get_dist_to_goal()
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