# Use a base image with necessary dependencies for graphics
FROM python:3.11

# Define the root directory inside the container
WORKDIR /Pyperbot-V2

# Update apt-get, then install ffmpeg, swig, and X11 libraries
RUN apt-get update && \
    apt-get install -y ffmpeg swig libx11-dev libxtst-dev libxrandr-dev libxrender-dev libxi-dev && \
    rm -rf /var/lib/apt/lists/*

# Add ffmpeg and swig to PATH
ENV PATH="/usr/bin:${PATH}"

# Set environment variable for X11
ENV DISPLAY=:99

# Set up Xvfb
RUN Xvfb :99 -screen 0 1024x768x16 &

# Copy the requirements file into the container
COPY requirements.txt .

# Update pip
RUN pip install --upgrade pip

# Install packages from the requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point to run the Python script
CMD ["python", "snakebot_description/snakebot.py"]
