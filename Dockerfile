#------------------------------------
# To build the image, do:
#   docker build --pull --rm -f "Dockerfile" -t pyperbotv2:v2 "."
# To run the container, do:
#   docker run -it --rm -v .:/pyperbot_v2 pyperbotv2:v2
# To run the snakebot.py, do:
#   python pyperbot_v2/snakebot_description/snakebot.py
# The container currently does not support pybullet GUI, use p.connect(p.DIRECT) instead.
#------------------------------------
# Use a base image with necessary dependencies for graphics
FROM python:3.11

# Define the root directory inside the container
WORKDIR /pyperbot_v2

# Update apt-get, then install ffmpeg, swig, and X11 libraries
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
