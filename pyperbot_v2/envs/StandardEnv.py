# import gym
import gymnasium as gym
import numpy as np
import math
import pybullet as p
import pybullet_data
import os

from ..snakebot_description.updated_snakebot_class import UpdatedSnakeBot
from ..snakebot_description.standard_snakebot_class import StandardSnakebot
from ..resources.goal import Goal, SoloGoal
from ..resources.plane import Plane
from ..resources.maze import Maze
from ..resources.lab import Lab
from ..resources.maze_small import MazeSmall
from ..resources.plane_small import PlaneSmall
import matplotlib.pyplot as plt
from pybullet_utils import bullet_client as bc
import logging 
import configparser
import shutil
import math

#framework inspired by code provided by stable baselines.
logger = logging.getLogger(__name__)

#setting up the environment
class StandardTestEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, **kwargs):
        self.num_envs = kwargs.get('num_envs', 1)
        self.terrain = kwargs.get('terrain', 'plane')
        #initialise environment
        #normalised action space for each joint - need to adjust the joint limits for the other snake robot
        super(StandardTestEnv, self).__init__()
        #update action space to be 8 dimensional for the 12 joints that should be moved.
        self.action_space = gym.spaces.Box(low = -1, 
                                           high = 1,
                                           shape = (20,), 
                                           #shape = (12,),
                                           #shape = (8,)
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
        self._finish_con = 1.0
        self._goal = None
        self._done = None
        self._env = None
        self.prev_dist_to_goal = 0
        self.org_dist_to_goal = 0
        self.rendered_img = None
        self.rot_matrix = None
        self.current_step = 0
        self.max_episode_steps = 20000
        self.cum_distance = 0
        self.reward = 0
        self.cum_reward = 0
        self.remark_counter = 0
        self.remark = None
        self.trial_number = 0
        self.past_rewards = []
        self.past_actions = []
        self.milestone_track = 0.1
        self.snake_init_pos = [0, 0, 0.5]
        self.termination_condition = None
        self.max_dist_to_goal = 0
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
        #increment current step
        self.current_step += 1
        print("%======================================%")
        print("Step: ", self.current_step)
        # print("Base Position: ", observation[:3])
        # print("Base Orientation: ", observation[3:6])
        snake_joint_observation = self._snake.get_joint_observation()
        dist_to_goal = self.get_dist_to_goal()
        self.cum_distance += abs(self.prev_dist_to_goal - dist_to_goal)
        distance_traveled_to_goal = self.org_dist_to_goal - dist_to_goal
        distance_traveled_in_step = self.prev_dist_to_goal - dist_to_goal
        if distance_traveled_to_goal > self.max_dist_to_goal:
            self.max_dist_to_goal = distance_traveled_to_goal
        print("Original Distance to Goal: ", self.org_dist_to_goal)
        print("Distance traveled to Goal: ", distance_traveled_to_goal)
        print("Distance to Goal: ", dist_to_goal)
        print("Cumulative Distance: ", self.cum_distance)
        print("Distance Travelled in Step: ", distance_traveled_in_step)
        #set the reward based on improvement in distance to goal
        self.reward = distance_traveled_in_step * 10000 + (distance_traveled_in_step*2000 if distance_traveled_in_step > 0 else 0)
        # print("Joint Positions: ", snake_joint_observation[0])
        # with open(f'pyperbot_v2/logs/actions/trial_{self.trial_number-2}.csv', 'a') as f:
        #     f.write(', '.join(str(x) for x in action.tolist()))
        #     f.write('\n')
        with open(f'pyperbot_v2/logs/actions/trial_{self.trial_number-2}.csv', 'a') as f:
            f.write(', '.join(str(x) for x in snake_joint_observation[0]))
            f.write('\n')
        self._done = False
        self._truncated = False
        #First termination condition - if distance to goal is less than finish condition, set reward to 10000 and end episode
        if dist_to_goal < self._finish_con:
            self.reward += 50000 + (1/self.current_step)*10000
            self.termination_condition = "goal reached"
            if self.current_step < 10000:
                self.reward += 10000
                self.termination_condition = "goal reached early"
            self._done = True
        #check x position and increment reward with milestones - scale reward positively each time it exceeds this distance
        if self._snake.get_base_observation()[0][0] > self.milestone_track:
            self.reward += 200*(self.milestone_track*10)
            self.milestone_track += 0.1
        #milestone rewards - every 500 steps, if snake reaches 1m closer to goal, increase reward by 2000
        elif (self.current_step % 100 == 0 and distance_traveled_to_goal > 0):
            self.reward += 100
        #penalise small movements
        if abs(distance_traveled_in_step) < 1e-07:
            self.reward -= 20
        #penalise movement in y direction
        #second termination condition - if snake falls off plane, set reward to -1000 and end episode
        elif self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][2] < 0:
            self.reward -= 1000
            self._done = True
            self.termination_condition = "fallen off plane"
        #if snake falls off boundaries in x and y axes, set reward to -1000 and end episode
        elif (self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0] < -20 or
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0] > 20 or 
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1] < -20 or 
            self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1] > 20):
            self.reward -= 1000
            self._done = True
            self.termination_condition = "out of bounds"
        #check truncation condition
        elif self.current_step >= self.max_episode_steps:
            self._done = True
            self._truncated = True
            self.reward -= 5000
            self.termination_condition = "truncated"
        #early termination condition: if the snake doesn't move at least 0.1 metres closer to the goal within 20% of the episode AND REWARD IS LESS THAN 1e-06
        #i.e. the snake is stuck, terminate the episode. #consider distance limiter as well for the stuck condition - or transform into an intermediate reward.
        elif (self.current_step > 0.4*self.max_episode_steps) and (self.current_step < self.max_episode_steps):
            if distance_traveled_to_goal < 1:
                self._done = True
                self.termination_condition = "poor performance"
        #penalise heavy movement in y direction
        if (abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1]) >  (abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][0]) + 0.5)
                or abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1]) > 1):
            self.reward -= (50 + abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0][1]*10))
            self.remark = 'heavy movement in y direction'
            self.remark_counter += 1
        #pernalise wasteful movement - if total distance travelled is more than 3 times the distance travelled on the way to the goal.
        # if ((self.cum_distance > 10*distance_traveled_to_goal) and (self.current_step > 1000)):
        #     self.reward -= 10
        #     self.remark = "wasteful movement"
        #     self.remark_counter += 1
        #penalise strong changes in orientation
        if (abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[1][1]) > 0.75 or
              abs(self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[1][2]) > 0.75):
            self.reward -= 20
            self.remark = "strong orientation change"
            self.remark_counter += 1
        #if moving backwards, penalise
        if ((distance_traveled_to_goal < 0) and ((self.prev_dist_to_goal - dist_to_goal) <0)):
            self.reward -= 50
            self.remark = "moving backwards"
            self.remark_counter += 1
        #stop if there are too many negative remarks (i.e. reward counter > threshold and reward <0)
        #set prev distance to goal to current distance to goal after update
        self.prev_dist_to_goal = dist_to_goal
        self.cum_reward += self.reward
        if (self.cum_reward < 0) and (self.current_step > 0.5*self.max_episode_steps or self.remark_counter > 3000):
            self.termination_condition = "too many negative remarks"
            self._done = True
        #stop if reward is too low
        if self.cum_reward < -20000:
            self.termination_condition = "negative reward threshold reached"
            self._done = True
            
        
        if self.num_envs > 1:  # Check if vectorized environments are being used
            for i in range(self.num_envs):
                config = configparser.ConfigParser()
                config['DISTANCE'] = {'distance_travelled_to_goal': distance_traveled_to_goal[i], 'distance_travelled': self.cum_distance[i], 'distance_to_goal': dist_to_goal[i],
                                    'last_step': self.prev_dist_to_goal[i]-dist_to_goal[i],
                                    'distance_in_timestep': distance_traveled_in_step[i]}
                config['REWARD'] = {'reward': self.reward[i], 'cum_reward': self.cum_reward[i]}
                config['GOAL'] = {'goal': self._goal.get_goal_pos(i)}
                config['SNAKE'] = {'snake_pos': self._client.getBasePositionAndOrientation(self._snake.get_ids()[i])[0], 'snake_ori': self._client.getBasePositionAndOrientation(self._snake.get_ids()[i])[1],
                                'snake_base_pos': self.snake_init_pos[i]}
                config['TERMINATION'] = {'termination': self.termination_condition[i] if self._done[i] else "not done"}
                config['TIMESTEPS'] = {'timestep': self.current_step[i]}
                with open(f'pyperbot_v2/logs/actions/run_results_{self.trial_number-2}_env{i}.ini', 'w') as configfile:
                    config.write(configfile)
        else:
            config = configparser.ConfigParser()
            config['DISTANCE'] = {'distance_travelled_to_goal': distance_traveled_to_goal, 'distance_travelled': self.cum_distance, 'distance_to_goal': dist_to_goal,
                                'last_step': self.prev_dist_to_goal-dist_to_goal,
                                'distance_in_timestep': distance_traveled_in_step,
                                'average_speed': self.cum_distance/self.current_step if self.current_step > 0 else 0,
                                'max_speed': self.max_dist_to_goal}
            config['REWARD'] = {'reward': self.reward, 'cum_reward': self.cum_reward}
            config['GOAL'] = {'goal': self._goal.get_goal_pos()}
            config['SNAKE'] = {'snake_pos': self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[0], 'snake_ori': self._client.getBasePositionAndOrientation(self._snake.get_ids()[0])[1],
                            'snake_base_pos': self.snake_init_pos}
            config['TERMINATION'] = {'termination': self.termination_condition if self._done else "not done"}
            config['REMARKS'] = {f'remark_{self.remark_counter}': self.remark if self.remark else "no remark yet"}
            config['TIMESTEPS'] = {'timestep': self.current_step}
            with open(f'pyperbot_v2/logs/actions/run_results_{self.trial_number-2}.ini', 'w') as configfile:
                config.write(configfile)        
        #check information needed for vectorised environments + add KL divergence for dictionary to monitor training process.
        return (observation, self.reward, self._done, self._truncated, {"obs": snake_joint_observation, "distance_to_goal": dist_to_goal, "reward": self.reward, "done": self._done, "dist_to_timestep": {"dist": dist_to_goal, "timestep": self.current_step},
                                                                        "termination": self.termination_condition})

    #step function using sparse binary rewards instead of intermediate rewards
    # def step(self, action):
    #     '''
    #     We quantify reward here based on the remaining distance to the goal. The distance
    #     is calculated using get_dist_to_goal function. 
    #     '''
    #     self._snake.apply_action(action)

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
    
    def reset(self, seed = None, env = None):
        '''
        Resets the simulation and returns the first observation. We may prescribe this to be the current dist_to_goal
        '''
        self.current_step = 0
        self.reward = 0
        self.cum_reward = 0
        self.cum_distance = 0
        self.remark_counter = 0
        self.milestone_track = 0.1
        self.remark = None
        self.termination_condition = None
        self._client = bc.BulletClient(connection_mode = p.DIRECT) #connect to pybullet client
        # self._client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._client.setRealTimeSimulation(0) 
        self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._client.resetSimulation() #reset the simulation
        self._client.setGravity(0, 0, -9.81)
        self.plane = Plane(self._client) #insert plane
        # self.snake_init_pos = [np.random.uniform(0,5), 0, 0.5]
        self._snake = StandardSnakebot(self._client, snakebot_dir = "pyperbot_v2/snakebot_description/urdf/snakebot.urdf.xacro", basePosition = [0, 0, 0.5], seed = int(self.seed_value))
        self._goal = SoloGoal(self._client, [-5, 0, 0], log_number = self.trial_number)  
        print("Goal Position: ", self._goal.get_goal_pos())  
        if env == "maze":
            self._env = Maze(self._client)
        elif env == "lab":
            self._env = Lab(self._client)
        elif env == "maze-small":
            self._env = MazeSmall(self._client)
        elif env == "plane-small":
            self._env = PlaneSmall(self._client)
        else:
            print("No environment specified. Defaulting to Plane.")
        self.done = False
        self.prev_dist_to_goal = self.get_dist_to_goal()
        self.org_dist_to_goal = self.prev_dist_to_goal
        # config = configparser.ConfigParser()
        # config['DISTANCE'] = {'distance': self.prev_dist_to_goal}
        # with open(f'pyperbot_v2/logs/trial_{self.trial_number}/goal_config.ini', 'w') as configfile:
        #     config.write(configfile)
        self.joint_ob = self._snake.get_joint_observation()
        base_pos, ori = self._snake.get_base_observation()[0], self._snake.get_base_observation()[1]
        print("Base Position: ", base_pos, "Base Orientation: ", ori)
        base_velocity, _ = self._client.getBaseVelocity(self._snake.get_ids()[0])
        linear_velocity = np.linalg.norm(base_velocity)
        observation = np.array(list(base_pos) + list(ori) + [linear_velocity], dtype = np.float32)
        self.trial_number += 1
        #TODO: fix to get the actual observation to ensure that the data returned is actually in the observation space
        return (observation, {"obs": self.joint_ob, "trial_number": self.trial_number, 'distance_to_goal': self.prev_dist_to_goal})
    
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