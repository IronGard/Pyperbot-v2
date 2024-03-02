
import pybullet as p
import os
import time
import pybullet_data
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)

from snakebot_sim_argparse import parse_args
from utils.structure_loader import Loader
from utils.gaits import Gaits
#from utils.snakebot_info import Info

# Arguements
args = parse_args()
sim_env = args.env
gait_type = args.gait
num_goals = args.num_goals
timesteps = args.timesteps
camera = args.cam

def main():
    pyb_setup = Loader("--mp4=results/videos/training_video.mp4")
    pyb_setup.plane()
    
    if sim_env == "lab":
        pyb_setup.lab("pyperbot_v2/snakebot_description/meshes/lab_floor_plan.stl")
    elif sim_env == "maze":
        pyb_setup.maze("pyperbot_v2/snakebot_description/meshes/maze_10x10.stl")
        pyb_setup.goal(num_goals)
    else:
        print('No selected environment.')
    
    robot_id = pyb_setup.robot("pyperbot_v2/snakebot_description/urdf/full_snakebot_no_macro.urdf.xacro")
    moving_joints = [p.getJointInfo(robot_id, joint) for joint in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED]
    moving_joint_ids = [p.getJointInfo(robot_id, joint)[0] for joint in range(p.getNumJoints(robot_id)) if p.getJointInfo(robot_id, joint)[2] != p.JOINT_FIXED]
    num_moving_joints = len(moving_joints)
    pyb_gaits = Gaits(robot_id)

    joint_pos = []
    for i in range(args.timesteps):
        if gait_type == "concertina_locomotion":
            pyb_gaits.concertina_locomotion()
        else:
            pyb_gaits.lateral_undulation()

        if camera == 1:
            pyb_setup.camera()
            
        pos, ori = p.getBasePositionAndOrientation(robot_id)
        euler = p.getEulerFromQuaternion(ori)
        print("Base position: ", pos)
        #save joint positions to csv
        
        for j in range(len(moving_joint_ids)):
            joint_pos.append(p.getJointState(robot_id, moving_joint_ids[j])[0])
        p.stepSimulation()
        time.sleep(1/240)
    
    joint_pos = np.array(joint_pos)
    df = pd.DataFrame(joint_pos)
    df.to_csv('pyperbot_v2/results/csv/joint_positions.csv', index = False)

if __name__ == "__main__":
    main()

    