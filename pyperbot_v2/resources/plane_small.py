import pybullet as p
import pybullet_data

class PlaneSmall():
    '''
    Instantiates the smaller plane in the environment.
    '''
    def __init__(self, client, terrain_dir):
        self._client = client
        self._terrain_scale = [0.5, 0.5, 0.5]
        terrain_visual_id = self._client.createVisualShape(
            shapeType=self._client.GEOM_MESH,
            fileName=terrain_dir,
            meshScale=self._terrain_scale)
        terrain_collision_id = self._client.createCollisionShape(
            shapeType=self._client.GEOM_MESH,
            fileName=terrain_dir,
            meshScale=self._terrain_scale,
            flags=1)
        self.terrain = self._client.createMultiBody(
            basePosition=[0, 0, 0],
            baseVisualShapeIndex=terrain_visual_id,
            baseCollisionShapeIndex=terrain_collision_id,
            physicsClientId=self._client._physicsClientId)

        