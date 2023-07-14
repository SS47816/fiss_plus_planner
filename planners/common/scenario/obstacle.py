from shapely.geometry import Polygon
from commonroad.scenario.state import State, TraceState
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle

class Obstacle(object):
    """ Obstacle (other agents)

    Used internally by the Planner class to represent an obstacle

    Attributes
    ------
        `timestamp`
        `id` (`int`): id of the obstacle
        `type` (`ObstacleType`): 
        `lane_id` (`int`): id of the lane that the trajectory was sampled on
        `type` (`float`): difference cost term
        
        ...
    """
    def __init__(self, obstacle: DynamicObstacle, time_step: int, time_resolution: float = 0.1):
        self.timestamp: float = time_step*time_resolution
        self.id: int = obstacle.obstacle_id
        self.type: ObstacleType = obstacle.obstacle_type
        self.polygon: Polygon = obstacle.obstacle_shape.shapely_object
        self.state: TraceState = obstacle.state_at_time(time_step)
        # self.pos = self.state.position
        # self.yaw = self.state.orientation
    
    # def __init__(self, id = None, type = None, timestamp = None, polygon = None, pos = None, yaw = None, prediction = None):
    #     self.id = id
    #     self.type = type
    #     self.timestamp = timestamp
    #     self.polygon = polygon
    #     self.pos = pos
    #     self.yaw = yaw
    #     self.prediction = prediction

    # def cr2irl(self, obs: DynamicObstacle, t: int):
    #     self.id = obs.obstacle_id
    #     self.type = obs.obstacle_type
    #     self.timestamp = t
    #     self.polygon = copy.deepcopy(obs.obstacle_shape.vertices)
    #     self.state = obs.state_at_time(t)
    #     self.pos = obs.state_at_time(t).position
    #     self.yaw = obs.state_at_time(t).orientation
    #     self.prediction = obs.prediction
        