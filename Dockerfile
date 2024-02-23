#-----------------------------------------------------------------------------------------
# Prerequisites:
#   For GUI ouput of pybullet, visual server must be installed in the host.
#   The server Vcxsrv can be installed from:
#       https://github.com/ArcticaProject/vcxsrv/releases/tag/1.17.0.0-3
#   or if chocolaty is available:
#       choco install vcxsrv
#   Select the "Disable access control" option when setting up Vcxsrv.
#
# This container only supports pybullet GUI if Vcxsrc is installed. 
# Change the code in snakebot.py to p.connect(p.DIRECT) instead if Vcxsrc is not available.
# 
# To pull the image from dockerhub, do:
#   docker image pull isaaccheung0930/pyperbotv2
# To build the image on host pc, do:
#   docker build --pull --rm -f "Dockerfile" -t pyperbotv2:v2 "."
# To run the container (assuming currently at the root directory of the project), do:
#   docker run -it --rm -v .:/pyperbot_v2 pyperbotv2:v2
# To enable X-server forwarding, do:
#   export DISPLAY=<your_ip>:0.0
# To run the snakebot.py, do:
#   python pyperbot_v2/snakebot_description/snakebot.py
#-------------------------------------------------------------------------------------------

# Use the python 3.11 base image (tensorflow cannot be installed if python version>3.11)
FROM python:3.11

# Define the root directory inside the container
WORKDIR /pyperbot_v2

# Update apt-get, then install ffmpeg and swig libraries
RUN apt-get update && \
    apt-get install -y \
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

# Set the entry point to run the Python script
CMD ["bash"]
