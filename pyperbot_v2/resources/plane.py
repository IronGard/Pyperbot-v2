import pybullet as p
import pybullet_data

class Plane():
    def __init__(self, client):
        self._client = client
        self._client.loadURDF("plane.urdf", 
                   basePosition = [0, 0, 0])