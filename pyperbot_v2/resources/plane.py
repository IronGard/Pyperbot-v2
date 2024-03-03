import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

class Plane():
    def __init__(self, client):
        self._physicsClient = client
        p.loadURDF("plane.urdf", 
                   basePosition = [0, 0, 0])