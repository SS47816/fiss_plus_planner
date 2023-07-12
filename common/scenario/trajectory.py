from common.scenario.lane import LaneType

class Pose(object):
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        
class PoseStamped(object):
    def __init__(self):
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

class Trajectory(object):
    """ Trajectory

    Used internally by the Planner class to generate the trajectory in both frenet frame / world frame
    Do not use this class for the planning outputs

    Attributes
    ------
        `lane_type` (`LaneType`): Left / Middle / Right
        `lane_id` (`int`): id of the lane that the trajectory was sampled on
        `id` (`int`): id of the trajectory itself (i.e. 0-100)
        `cd` (`float`): difference cost term
        `cv` (`float`): velocity cost term
        `cf` (`float`): final total cost
        
        ...
    """
    def __init__(self):
        self.lane_type = LaneType.UNDEFINED # Left / Middle / Right
        self.lane_id = -1 # id of the lane that the trajectory was sampled on
        self.id = -1 # id of the trajectory itself (i.e. 0-100)
        
        self.t = []
        self.poses = []
        
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0
