import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorboard as tb
import tensorflow as tf

#plotting results provided from the CSV files and getting information out
def plot_results():
    '''
    Function to plot the results from the CSV files.
    '''
    #read the CSV files
    df = pd.read_csv('results.csv')
    #plot the results
    plt.plot(df['episode'], df['reward'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.show()

def get_info():
    '''
    Function to get the information from the CSV files.
    '''
    df = pd.read_csv('results.csv')
    print(df.head())
    print(df.describe())



