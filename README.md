# Rough Terrain Navigation of a Snake Robot using Model-Based Reinforcement Learning

Project Members: Hahn Lon Lam, Harry Softley-Graham, Gene Dumnernchanvanich, Wing Hei Cheung

Supervisor: Dr Chow Yin Lai

Most snake robots developed for search-and-rescue-tasks have relied on preprogrammed instructions in order for the robot to navigate. This project proposes to use Model-Based reinforcement learning (MBRL) to enable the robot to achieve forward locomotion and be able to navigate complex environments autonomously.

## Prerequisites
**Gynmnaisum**  
Install the latest version of swig and ffmpeg and add them to your PATH.  
**Pybullet**  
Install the Microsoft Visual C++ Build Tools. This can be found at:  
https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022  
**Docker Visual Server**  
To run Pybullet simulation in docker container, install Vcxsrv from the following link and select "Disable Access Control" option at launch. This config can be saved for each other run.  
https://github.com/ArcticaProject/vcxsrv/releases/tag/1.17.0.0-3  
**CUDA**  
To enable GPU processing of the RL model, install Nvidia CUDA toolkit from the following link.  
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64  

## Supported Python Version
This project supports python versions from 3.11.0-3.11.7.

## Requirements
To install the requirements, run:
```
pip install -r requirements.txt
```
You may need to remove the +cu121 at the end of the torch, torchaudio and torchvision imports in order to be able to install all necessary requirements. Tensorflow does not support python version >3.11 at the moment. 

## Running from docker container
The Docker image can be pull from Dockerhub. Tag v2 and onwards support CUDA.
```
docker image pull isaaccheung0930/pyperbotv2:<tag>
```
Alternatively, to build the image on host pc:
```
docker build --pull --rm -f "Dockerfile" -t pyperbotv2:v2 "."
```
To run the container, execute container_setup.bat or 
run the following commands (assuming currently at root directory):
```
docker run -it --rm --gpus all -v .:/pyperbot_v2 pyperbotv2:v2
export DISPLAY=<your_ip>:0.0
```
Within the container, for manual control visualisation, do:
```
python3 pyperbot_v2/snakebot_description/snakebot_sim.py.
```
For the main snakebot with reinforcement learning, run:
```
python3 main.py
