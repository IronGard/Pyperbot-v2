import argparse

def parse_args():
    '''
    Function to accept arguments for controlling the simulation environment.
    '''
    parser = argparse.ArgumentParser(description='snakebot_sim')
    parser.add_argument('-e', '--env', type=str, default='none', help='Sim env: none, maze, lab, terrain.')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the environment.')
    parser.add_argument('-lc', '--load_config', type = bool, default = False, help = 'Load a configuration file for the simulation')
    parser.add_argument('-g', '--gait', type=str, default='lateral_undulation', help='Gaits: lateral undulation, concertina_locomotion')
    parser.add_argument('-ng', '--num_goals', type=int, default=3, help='Number of goals (maze env only).')
    parser.add_argument('-c', '--cam', type=int, default=0, help='Attach head camera: 0, 1.')
    parser.add_argument('-ts', '--timesteps', type=int, default=2400, help='Number of timesteps for the simulation.')
    parser.add_argument('-s', '--server', type=bool, default=False, help='Running code to read data output directly to server')
    args = parser.parse_args()

    return args