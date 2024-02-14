import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
from ..resources.snakebot import Snakebot
from ..resources.goal import Goal
from ..resources.terrain import terrain
import matplotlib.pyplot as plt

#setting up the environment
class TestEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low = np.array([0,0]), high = np.array([200, 200]), dtype = np.float32)
        self.random = gym.utils.seeding.np_random()
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(0.01, self.client)
        p.setGravity(0, 0, -9.81)

        self.snake = None
        self.goal = None
        self.done = None
        self.prev_dist_to_goal = None #parameterising the distance remaining to the goal/final reward
        self.rendered_img = None
        self.reset()

    def step(self, action):
        '''
        We quantify reward here based on the remaining distance to the goal. The distance
        is calculated using get_dist_to_goal function.
        '''
        self.snake.apply_action(action)
        p.stepSimulation()
        self.render()
        snake_obs = self.snake.get_joint_observation()
        dist_to_goal = self.get_dist_to_goal()
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)

    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return np.linalg.norm(self.goal - self.snake.get_position())
    
    def reset(self):
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.snake = Snakebot(self.client)
        self.goal = self.random.rand(2) * 200 #insert goal in random position
        self.done = False
        self.dist_to_goal = self.get_dist_to_goal()
        #add visual element of goal
        Goal(self.client, self.goal)
        snake_joint_ob = self.snake.get_joint_observation()
        snake_base_ob = self.snake.get_base_observation()
        self.prev_dist_to_goal = self.get_dist_to_goal()
        return snake_joint_ob, snake_base_ob

    def render(self, mode = 'human'):
        '''
        Function to render the environment
        '''
        snake_id, client_id = self.snake.get_id()
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = self.snake.get_position(), distance = 6, yaw = 0, pitch = -10, roll = 0, upAxisIndex = 2, physicsClientId = client_id)
        proj_matrix = p.computeProjectionMatrixFOV(fov = 60, aspect = 1.0, nearVal = 0.1, farVal = 100.0)
        pos, ori = p.getBasePositionAndOrientation(snake_id, client_id)

        #rotate the camera to match values
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori).reshape(3, 3))
        camera_vector = np.matmul(rot_matrix, np.array([1, 0, 0]))
        camera_up = np.matmul(rot_matrix, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vector, camera_up)

        #get images from the simulation
        frame = p.getCameraImage(width = 256, height = 256, viewMatrix = view_matrix, projectionMatrix = proj_matrix, renderer = p.ER_BULLET_HARDWARE_OPENGL, physicsClientId = client_id)
        frame = np.reshape(frame[2], (256, 256, 4))
        self.rendered_img = frame
        plt.draw()
        plt.pause(0.001)




    def close(self):
        '''
        Closes simulation by disconnecting from Pybullet client
        '''
        p.disconnect(self.client)