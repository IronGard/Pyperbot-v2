#Primary change: add physicsClientId = self.client._client to all p calls

#File to define the snakebot with a class-based implementation
import pybullet as p
import numpy as np 
import math
import os
import sys
import random

#import utils functions for loading goals and maze
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)



class Snakebot:
    def __init__(self, client):
        '''
        Constructor for the Snakebot class to define a snakebot object
        '''
        self.client = client._client
        robot_name = os.path.join(os.getcwd(), 'pyperbot_v2/snakebot_description/urdf/updated_snakebot.urdf.xacro')
        print(robot_name)
        self.robot = p.loadURDF(robot_name, [0.5, 0.5, 0], globalScaling = 2.0,
                                physicsClientId = self.client)
        self.planeID = p.loadURDF("plane.urdf", [0, 0, 0])
        #inclusion of the loaded maze
        self.maze_visual_id = p.createVisualShape(shapeType = p.GEOM_MESH,
                                                  fileName = "pyperbot_v2/snakebot_description/meshes/maze_10x10.stl",
                                                  meshScale = [1, 1, 1],
                                                  physicsClientId = self.client)
        self.maze_collision_id = p.createCollisionShape(shapeType = p.GEOM_MESH,
                                                        fileName = "pyperbot_v2/snakebot_description/meshes/maze_10x10.stl",
                                                        meshScale = [1, 1, 1],
                                                        flags = 1,
                                                        physicsClientId = self.client)
        self.mazeID = p.createMultiBody(basePosition = [0, 0, 0],
                                        baseVisualShapeIndex = self.maze_visual_id,
                                        baseCollisionShapeIndex = self.maze_collision_id,
                                        physicsClientId = self.client)
        #getting prismatic joints and revolute joints from the robot
        self.moving_joints = [p.getJointInfo(self.robot, joint, physicsClientId = self.client) for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] != p.JOINT_FIXED]
        self.moving_joints_inds = [p.getJointInfo(self.robot, joint, physicsClientId = self.client)[0] for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] != p.JOINT_FIXED]
        print(self.moving_joints_inds)
        self.prismatic_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] == p.JOINT_PRISMATIC]
        self.revolute_joints = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.moving_joints)
    def get_ids(self):
        '''
        Returns id of the robot and the plane used
        '''
        return self.robot, self.planeID
    def apply_action(self, actions):
        '''
        Function to apply actions to the snakebot
        To be further refined and implemented based on actual snake robot movement.
        * Should be more precise for handling revolute and prismatic joints in particular.
        * Need to decide how to map joint movements to the snakebot's movement.
        '''
        #action needs to be twenty dimensional
        #consider setting prismatic and revolute joints separately.
        #check why the render doesn't work with the client for some reason - make sure that the joint values are still being printed out appropriately at the end of the simulation.
        #Each time step is 1.240 of a second
        #we can calcualte the drag and friction of the car as well.
        #debug - using p.setjointMotorControl2 to set joint motor control
        moving_joint_list = [0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30]
        for i in range(len(moving_joint_list)):
            p.setJointMotorControl2(self.robot, 
                                    moving_joint_list[i], 
                                    p.POSITION_CONTROL, 
                                    targetPosition = actions[i], 
                                    force = 30, 
                                    physicsClientId = self.client)
        base_position, base_orientation = p.getBasePositionAndOrientation(self.robot, physicsClientId = self.client)
        # p.setJointMotorControlArray(self.robot,
        #                             [0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30],
        #                             p.POSITION_CONTROL,
        #                             targetPositions = actions,
        #                             forces = [30]*self.num_joints,
        #                             physicsClientId = self.client)
        # print(len([0, 1, 2, 5, 6, 8, 9, 10, 13, 14, 16, 17, 18, 21, 22, 24, 25, 26, 29, 30]))
        # print(len(actions))
    
    def gen_goals(self, num_goals):
        '''
        Function to generate number of goals to be added to the snakebot environment
        '''
        goals = []
        duck_scale = [1, 1, 1]
        shift = [0, -0.02, 0]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName="duck.obj",
                                            rgbaColor=[1, 1, 1, 1],
                                            specularColor=[0.4, 0.4, 0],
                                            visualFramePosition=shift,
                                            meshScale=duck_scale,
                                            physicsClientId = self.client)

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                fileName="duck_vhacd.obj",
                                                collisionFramePosition=shift,
                                                meshScale=duck_scale,
                                                physicsClientId = self.client)

        #Generate random coordinates for the goal
        x, y = [], []
        for i in range(num_goals):
            x.append(random.randrange(-9, 9, 2))
            y.append(random.randrange(1, 19, 2))
            p.createMultiBody(baseMass=1,
                            baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collisionShapeId,
                            baseVisualShapeIndex=visualShapeId,
                            baseOrientation=[0, 45, 45, 0],
                            basePosition=[x[i], y[i], 0],
                            physicsClientId = self.client)
            goals.append([x[i], y[i]])
        return goals
    
    def get_joint_observation(self):
        #get num moving joints and moving joint indices
        moving_joints_inds = [joint for joint in range(p.getNumJoints(self.robot, physicsClientId = self.client)) if p.getJointInfo(self.robot, joint, physicsClientId = self.client)[2] != p.JOINT_FIXED]
        num_moving_joints = len(moving_joints_inds)
        joint_states = p.getJointStates(self.robot, moving_joints_inds, physicsClientId = self.client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        return [joint_positions, joint_velocities]
    
    def get_base_observation(self):
        '''
        Obtain the position and orientation of the robot and use these parameters as an observation of the state of the environment.
        Compute the sum to return an individual float as the observation value.
        '''
        pos, ang = p.getBasePositionAndOrientation(self.robot, physicsClientId = self.client)
        ori = p.getEulerFromQuaternion(ang)
        return [pos, ori]
    
    def make_sin_wave(self):
        '''
        Function to generate sine wave along body of the snake.
        '''
        pass
    
