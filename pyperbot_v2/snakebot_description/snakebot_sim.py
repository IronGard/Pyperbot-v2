
import pybullet as p
import os
import time
import pybullet_data
import sys

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

    pyb_gaits = Gaits(robot_id)

    while True:
        if gait_type == "concertina_locomotion":
            pyb_gaits.concertina_locomotion()
        else:
            pyb_gaits.lateral_undulation()

        if camera == 1:
            pyb_setup.camera()
            
        p.stepSimulation()
        time.sleep(1/240)
    

if __name__ == "__main__":
    main()

    