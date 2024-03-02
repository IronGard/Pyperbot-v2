import argparse

def parse_args():
    '''
    Function to accept arguments for controlling the simulation environment.
    '''
    parser = argparse.ArgumentParser(description='snakebot_sim')
    parser.add_argument('-e', '--env', type=str, default='none', help='Sim env: none, maze, lab.')
    parser.add_argument('-g', '--gait', type=str, default='lateral_undulation', help='snake gaits: lateral undulation, concertina_locomotion')
    parser.add_argument('-ng', '--num_goals', type = int, default = 3, help = 'Number of goals (maze env only).')
    parser.add_argument('-ts', '--timesteps', type = int, default = 2400, help = 'Number of timesteps for the simulation.')
    args = parser.parse_args()

    return args