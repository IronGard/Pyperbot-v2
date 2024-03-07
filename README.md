# Rough Terrain Navigation of a Snake Robot using Model-Based Reinforcement Learning
Most snake robots developed for search-and-rescue-tasks have relied on preprogrammed instructions in order for the robot to navigate. This project proposes to use Model-Based reinforcement learning (MBRL) to enable the robot to achieve forward locomotion and be able to navigate complex environments autonomously.

## Prerequisites
Please install the latest version of swig and ffmpeg and add it to your PATH. This will be needed for the gymnasium[box2d] command. Please also install Microsoft Visual C++ Build Tools, required for PyBullet, which may be found here: https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022. Please also install VCXSRV from the following link: https://github.com/ArcticaProject/vcxsrv/releases/tag/1.17.0.0-3. For VCXSRV, please also launch XLaunch and select the "Disable Access Control" option, saving this as a config to use for each other run. 

## Supported Python Version
This project supports python versions from 3.11.0-3.11.7.

## Requirements
To install the requirements, run:
```
pip install -r requirements.txt
```
You may need to remove the +cu121 at the end of the torch, torchaudio and torchvision imports in order to be able to install all necessary requirements.

## Running from docker container
To pull the image from dockerhub, do:
```
docker image pull isaaccheung0930/pyperbotv2
```
Alternatively, to build the image on host pc, do:
```
docker build --pull --rm -f "Dockerfile" -t pyperbotv2:v2 "."
```
To run the container (assuming currently at the root directory of the project), do:
```
docker run -it --rm -v .:/pyperbot_v2 pyperbotv2:v2
```
To enable X-server forwarding, do:
```
export DISPLAY=<your_ip>:0.0
```
To run the snakebot.py for manual control visualisation, do:
```
python pyperbot_v2/snakebot_description/snakebot_sim.py.
```
For the main snakebot with reinforcement learning, run:
```
python main.py
```
with or without arguments. See the snakebot_description folder for more details on the files and arguments that may be passed into the function.

## Repository Structure
The repository currently follows the following structure:
```
|pyperbot_v2 -- contains most assets, resources, agents and environments used in the project
|__ agents -- contains the reinforcement learning (RL) agents for the training of the snake environment task
|__ config -- contains configuration files for environments and mazes
|__ envs -- contains the environments used for training the snake robot
|__ Pi_Code_folder -- contains code for interfacing with the raspberry pi to control the snake robot
|__ pipeline -- contains code for data analysis and preprocessing (as required)
|__ resources -- contains the code for urdf and meshes for loading different objects/mazes into the environment
|____ urdf -- urdf files
|____ meshes -- STL meshes for mazes, goals etc.
|__ slam_code -- code for simultaneous localisation and mapping (SLAM)
|__ snakebot_description -- contains the primary simulation code and implementation (both manual and RL) approaches to programming and controlling the snake gait.
|__ tensorboard_logs -- folder to store results from tensorboard from training runs
|__ utils -- utility functions for running RL models and to supplement other parts of the code.
|__ wrappers -- Wrappers for the snakebot environment (e.g. custom monitor to get desired results, TimeLimitEnv to restrict number of timesteps)
```
