import matplotlib.pyplot as plt
from shutil import rmtree
from os import path, mkdir
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy

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


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()