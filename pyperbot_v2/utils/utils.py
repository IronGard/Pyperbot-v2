import matplotlib.pyplot as plt
from shutil import rmtree
from os import path, mkdir

def plot_gait(gait):
    '''
    Function to plot the gait of the snakebot.

    Parameters:
        x: list
            the x-coordinates of the snakebot.
        y: list
            the y-coordinates of the snakebot.
        z: list
            the z-coordinates of the snakebot.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(gait[0], gait[1], gait[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def file_clear(file_path):
    '''
    Function to clear the file at the given path.
    If the file does not exist, a new file is created.

    Parameter:
        file_path: str
            the path of the file to be cleared.
    '''
    if path.exists(file_path):
        rmtree(file_path, ignore_errors=True)
    else:
        mkdir(file_path)
