# Rough Terrain Navigation of a Snake Robot using Model-Based Reinforcement Learning
Most snake robots developed for search-and-rescue-tasks have relied on preprogrammed instructions in order for the robot to navigate. This project proposes to use Model-Based reinforcement learning (MBRL) to enable the robot to achieve forward locomotion and be able to navigate complex environments autonomously.

## Prerequisites
Please install the latest version of swig and ffmpeg and add it to your PATH. This will be needed for the gymnasium[box2d] command. Please also install Microsoft Visual C++ Build Tools, required for PyBullet, which may be found here: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022. Lastly, it would also be highly beneficial to create a virtual environment to install the requirements.

## Supported Python Version
This project uses python 3.11.7.

## Requirements
To install the requirements, run:
```
pip install -r requirements.txt
```
You may need to remove the +cu121 at the end of the torch, torchaudio and torchvision imports in order to be able to install all necessary requirements.

## Running the file
At the moment, in order to visualise the snakebot, please run snakebot.py as follows:
```
python snakebot.py
```

## Repository Structure
The repository currently follows the following structure:
```
| .dev_container -- (used to store Dockerfile and create container (TBC))
| agents -- (stores RL agents e.g. TD3, PPO and DDPG models)
| envs -- (stores environments for navigation tasks)
| models -- (stores saved models)
| pipeline -- (contains python files for data analysis and preprocessing tasks)
| results -- (stores training results)
|__ graphs -- (stores graphed results)
|__ videos -- (stores video results)
| ros_components -- (stores code from previous ROS builds)
| snakebot_description -- (stores URDF files and meshes)
|__meshes -- (stores the meshes used as STL files)
|__urdf -- (stores the xacro files for the snake robot)
```