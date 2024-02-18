import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
# from ..resources.goal import Goal
# from ..resources.terrain import terrain
from ..snakebot_description.snakebot_class_implementation import Snakebot
import matplotlib.pyplot as plt

#setting up the environment
class TestEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        #initialise environment
        self.action_space = gym.spaces.Discrete(4) #Forward, backwards, left and right - four main directions/actions
        #Observation space - 10 dimensional continuous array - x,y,z position and orientation of base, velocity of snakebot, and x,y,z positions of the target to be reached
        self.observation_space = gym.spaces.Box(
            low = np.array([-15, -15, -15, -1, -1, -1, 0, -15, -15, -15]),
            high = np.array([15, 15, 15, 1, 1, 1, 1, 15, 15, 15]), 
            dtype = np.float32
            )
        self.random, _ = gym.utils.seeding.np_random() #setting the seed for the RL environment
        self.client = p.connect(p.DIRECT) #use direct client for now - some issues with GUI client.
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #get the plane.urdf and other URDF files available on bullet3/examples.
        p.setTimeStep(0.01, self.client)
        p.setGravity(0, 0, -9.81)

        #Additional Params for the environment.
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
        #TODO: set condition for done
        '''
        self.snake.apply_action(action)
        p.stepSimulation()
        self.render() 
        snake_obs = self.snake.get_joint_observation() #here we primarily want the joint positions, not velocities
        dist_to_goal = self.get_dist_to_goal()
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        return [reward, snake_obs[0], 0]

    def seed(self, seed = None):
        #Generate seed for the environment
        pass
        
    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return np.linalg.norm(self.goal - self.snake.get_position())
    
    def reset(self, seed = None):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.81)
        self.snake = Snakebot(self.client)
        self.goal = np.random.rand(3) * 200 #insert goal in random position
        self.done = False
        self.prev_dist_to_goal = self.get_dist_to_goal()
        #add visual element of goal
        # Goal(self.client, self.goal)
        self.joint_ob = self.snake.get_joint_observation()
        self.base_ob = self.snake.get_base_observation()
        return self.prev_dist_to_goal
    
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