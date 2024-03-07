from setuptools import setup

setup(
    name = "Pyperbot-v2",
    version = "0.0.1",
    author = "Hahn Lam",
    author_email = 'hahnlon.lam@gmail.com',
    install_requires =  ['gymnasium', 
                         'pybullet', 
                         'stable_baselines3', 
                         'torch', 
                         'rl_zoo3', 
                         'panda_gym',
                         'numpy',
                         'opencv_python',
                         'matplotlib'],
)

