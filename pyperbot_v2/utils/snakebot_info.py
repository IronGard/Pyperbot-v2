import pybullet as p
import pandas as pd

class Info():
    '''
    Class to obtain/ display information related to the snake robot.
    Attributes:
        robot_id: int
            the unique id of the robot.
    Methods:
        joint_info()
            classify the joints.
        moving_joint_info()
            obtain a list of moving joints.
        base_info()
            obtain position and orientation of the robot base.
        joint_position()
            obtain the current and past joint positions.
    '''
    def __init__(self, robot_id):
        '''
        Setup robot id.

        Parameter:
            robot_id: int
                the unique id of the robot.
        '''
        self._robot_id = robot_id

    def joint_info(self):
        '''
        Classify the joints and provide corresponding information.

        Returns:
            revolute_df: Dataframe
                table of revolute joints info
            prismatic_df: Dataframe
                table of prismatic joints info
            fixed_df: Dataframe
                table of fixed joints info
            others_df: Dataframe
                table of other joints info
        '''
        revolute, prismatic, fixed, others = [], [], [], []
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            joint_dict = {"joint_index": joint_info[0],
                          "joint_name": joint_info[1],
                          "joint_type": joint_info[2],
                          "link_name": joint_info[12],
                          "parent_link": joint_info[16]}
            if joint_info[2] == 0:
                revolute.append(joint_dict)
            elif joint_info[2] == 1:
                prismatic.append(joint_dict)
            elif joint_info[2] == 4:
                fixed.append(joint_dict)
            else:
                others.append(joint_dict)
            
        revolute_df = pd.DataFrame(revolute)
        prismatic_df = pd.DataFrame(prismatic)
        fixed_df = pd.DataFrame(fixed)
        others_df = pd.DataFrame(others)

        return revolute_df, prismatic_df, fixed_df, others_df

    def moving_joint_info(self):
        '''
        Retrieve all moving joints ids.

        Return:
            moving_joints_ids: list
                a list of ids of moving joints.
        '''
        moving_joints_ids = []
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            if joint_info[2] != 4:
                moving_joints_ids.append(joint_info[0])

        return moving_joints_ids
    
    def base_info(self):
        '''
        Obtain current position and orientation of the base. Note that the orientation has been converted to euler format.

        Return:
            pos: list
                list of position in [x,y,z]
            ori: list
                list of orientation in [x,y,z]
        '''
        pos, ori = p.getBasePositionAndOrientation(self._robot_id)
        ori = p.getEulerFromQuaternion(ori)

        return pos, ori

    def joint_position(self, past_pos=[]):
        '''
        Obtain joint positions.

        Parameter:
            past_pos: list
                all past joint positions (default empty).
        
        Returns:
            current_pos: list
                the joint positions at current time step
            past_pos: list
                all past joint positions (current included).
        '''
        current_pos= []
        for i in self.moving_joint_info():
            current_pos.append(p.getJointState(self._robot_id, i)[0])
        past_pos.append(current_pos)

        return current_pos, past_pos
