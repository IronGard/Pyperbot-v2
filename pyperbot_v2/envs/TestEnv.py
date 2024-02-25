import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
# from ..resources.goal import Goal
# from ..resources.terrain import terrain
from ..snakebot_description.snakebot_class_implementation import Snakebot
from ..resources.goal import Goal
import matplotlib.pyplot as plt

#framework inspired by code provided by stable baselines.

#setting up the environment
class TestEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self):
        #initialise environment
        #action space should actually be continuous - each joint can move in between from 0 and 1 different values 
        #TODO: need to set action space for prismatic and revolute joints, separately
        self.action_space = gym.spaces.Box(low = np.array([0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236, 0, -0.0873, -0.2618, -0.5236, -0.5236]), 
                                           high = np.array([0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236, 0.02, 0.0873, 0.2618, 0.5236, 0.5236]), 
                                           dtype = np.float32) #20 joints in the snakebot (to be printed + appended to a CSV file)
        #Observation space - 8 dimensional continuous array - x,y,z position and orientation of base, remaining distance to goal, and velocity of the robot
        #we require a general number of the infomration.
        self.observation_space = gym.spaces.Box(
            low = np.array([0, 0, 0, -3.1415, -3.1415, -3.1415, 0, 0]), #first three are base position, next three are base orientation, next is velocity, last is distance to goal
            high = np.array([200, 200, 200, 3.1415, 3.1415, 3.1415, 100, 20]),
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
        self.rot_matrix = None
        self.reset()

    def step(self, action):
        '''
        We quantify reward here based on the remaining distance to the goal. The distance
        is calculated using get_dist_to_goal function. 
        #TODO: set condition for done
        '''
        self.snake.apply_action(action)
        p.stepSimulation() 
        snake_joint_obs = self.snake.get_joint_observation() #here we primarily want the joint positions, not velocities
        base_pos, base_ori = self.snake.get_base_observation()
        dist_to_goal = self.get_dist_to_goal()
        #set the reward based on improvement in distance to goal
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0)
        #if the snake runs off boundaries of the grid, self.done == True
        if (all(x < 0 for x in base_pos)) or (all(x>200 for x in base_pos)):
            self.done = True
            reward = -50
        #if the distance to goal is less than threshold, we can set self.done == True
        elif dist_to_goal < 0.5:
            self.done = True
            reward = 50
        #get the basevelocity of the snake
        base_velocity, _ = p.getBaseVelocity(self.snake.get_ids()[0], self.client)
        #calculate normalised linear velocity of snake
        linear_velocity = np.linalg.norm(base_velocity)
        observation = np.array(list(base_pos) + list(base_ori) + [linear_velocity, dist_to_goal], dtype = np.float32)
        return (observation, reward, self.done, False, {"obs": snake_joint_obs})
        #change returned observation to be a numpy array instead of a list.

    def seed(self, seed = None):
        #Generate seed for the environment
        self.random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def get_dist_to_goal(self):
        '''
        Function to calculate distance to goal using Euclidean Heuristic
        '''
        return np.linalg.norm(self.goal.get_goals()[0]) - np.linalg.norm(self.snake.get_base_observation()[0])
    
    def reset(self, seed = None):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        p.resetSimulation(self.client) #reset the simulation
        p.setGravity(0, 0, -9.81)
        self.snake = Snakebot(self.client)
        self.goal = Goal(self.client, 3) #insert goal in random position
        self.done = False
        self.prev_dist_to_goal = self.get_dist_to_goal()
        #add visual element of goal (TODO)
        # Goal(self.client, self.goal)
        self.joint_ob = self.snake.get_joint_observation()
        base_pos, ori = self.snake.get_base_observation()[0], self.snake.get_base_observation()[1]
        print(base_pos)
        print(ori)
        dist_to_goal = self.get_dist_to_goal()
        base_velocity, _ = p.getBaseVelocity(self.snake.get_ids()[0], self.client)
        linear_velocity = np.linalg.norm(base_velocity)
        print(linear_velocity)
        print(dist_to_goal)
        observation = np.array(list(base_pos) + list(ori) + [linear_velocity, dist_to_goal], dtype = np.float32)
        #TODO: fix to get the actual observation to ensure that the data returned is actually in the observation space
        return (observation, {"obs": self.joint_ob})
    
    def render(self, mode = 'human'):
        '''
        Function to render the environment
        '''
        if self.rendered_img is None:
            self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
        snake_id, client_id = self.snake.get_ids()
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition = self.snake.get_base_observation()[0], distance = 6, yaw = 0, pitch = -10, roll = 0, upAxisIndex = 2, physicsClientId = client_id)
        proj_matrix = p.computeProjectionMatrixFOV(fov = 60, aspect = 1.0, nearVal = 0.1, farVal = 100.0)
        pos, ori = p.getBasePositionAndOrientation(snake_id, client_id)

        #rotate the camera to match values
        rot_matrix = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
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