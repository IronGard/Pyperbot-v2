#-----------------------------------------------------------------------------------------
# Prerequisites 1:
#   For GUI ouput of pybullet, visual server must be installed in the host.
#   The server Vcxsrv can be installed from:
#       https://github.com/ArcticaProject/vcxsrv/releases/tag/1.17.0.0-3
#   or if chocolaty is available:
#       choco install vcxsrv
#   Select the "Disable access control" option when setting up Vcxsrv.
#
# This image only supports pybullet GUI if Vcxsrc is installed. 
# Change the code in snakebot.py to p.connect(p.DIRECT) instead if Vcxsrc is not available.
#
# Prerequisites 2:
#   To enable CUDA support, the host pc must have CUDA compatiable GPU and have installed the toolkit.
#   The installation version can be verified by:
#       nvcc --version
# The image only supports CPU if CUDA toolkit is not found.
# 
# To pull the image from dockerhub, do:
#   docker image pull isaaccheung0930/pyperbotv2
# To build the image on host pc, do:
#   docker build --pull --rm -f "Dockerfile" -t pyperbotv2:v2 "."
# To run the container (assuming currently at the root directory of the project):
#   1. execute the contain_setup.bat, or use the following command
#   2. docker run -it --rm --gpus all -v .:/pyperbot_v2 pyperbotv2:v2
# To enable X-server forwarding, do:
#   export DISPLAY=<your_ip>:0.0
# To run the snakebot.py, use the following command within the container:
#   python3 pyperbot_v2/snakebot_description/snakebot.py
#-------------------------------------------------------------------------------------------

# Use the python 3.11 base image (tensorflow cannot be installed if python version>3.11)
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

# Define the root directory inside the container
WORKDIR /pyperbot_v2

# Update apt-get, then install ffmpeg and swig libraries
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Add ffmpeg and swig to PATH
ENV PATH="/usr/bin:${PATH}"

# Copy the requirements file into the container
COPY requirements.txt .

# Update pip
RUN pip install --upgrade pip

# Install packages from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install other necessary peripherals
RUN pip install --no-cache-dir optuna-dashboard SQLAlchemy plotly 

# Set the entry point to run the Python script
CMD ["bash"]
