class Scenario:
    """ Trajectory

    Used internally by the Planner class to generate the trajectory in both frenet frame / world frame
    Do not use this class for the planning outputs

    Attributes
    ------
        `lane_type` (`LaneType`): Left / Middle / Right
        `lane_id` (`int`): id of the lane that the trajectory was sampled on
        `id` (`int`): id of the trajectory itself (i.e. 0-100)
        cd (`float`): difference cost term
        cv (`float`): velocity cost term
        cf (`float`): final total cost
        
        ...
    """
    def __init__(self):
        # time resolution between two planned waypoints
        self.tick_t = 0.2  # time tick [s]
        self.ego_lane = 
        self.ego_lane = 
        self.ego_lane = 
        self.obstacles = [] #