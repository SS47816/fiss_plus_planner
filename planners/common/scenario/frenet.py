import copy
import numpy as np
from planners.common.scenario.lane import LaneType
from planners.common.geometry.math_utils import unifyAngleRange

class State(object):
    def __init__(self, t: float = 0.0, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, v: float = 0.0, a: float = 0.0):
        self.t = t
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a
        
class FrenetState(object):
    def __init__(self, t: float = 0.0,
                       s: float = 0.0, s_d: float = 0.0, s_dd: float = 0.0, s_ddd: float = 0.0,
                       d: float = 0.0, d_d: float = 0.0, d_dd: float = 0.0, d_ddd: float = 0.0):
        self.t = t
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.s_ddd = s_ddd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_ddd = d_ddd
    
    def __str__(self):
        return f'FrenetState with d={self.d:.2f}, s_d={self.s_d:.2f}, t={self.t:.2f}'
    
    def from_state(self, state: State, polyline: np.ndarray):
        
        def find_nearest_point_idx(state: State, polyline: np.ndarray):
            distances = np.hypot(polyline[:,0] - state.x, polyline[:,1] - state.y)
            return np.argmin(distances)

        def find_next_point_idx(state: State, polyline: np.ndarray):
            nearest_idx = find_nearest_point_idx(state, polyline)
            heading = np.arctan2(polyline[nearest_idx, 1] - state.y, polyline[nearest_idx, 0]- state.x)
            angle = abs(state.yaw - heading)
            angle = min(2*np.pi - angle, angle)

            if angle > np.pi/2:
                next_wp_id = nearest_idx + 1
            else:
                next_wp_id = nearest_idx
                
            # if it is behind the start of the waypoint list
            if next_wp_id < 1:
                next_wp_id = 1
            # if it reaches the end of the waypoint list
            elif next_wp_id >= polyline.shape[0]:
                next_wp_id = polyline.shape[0] - 1
            
            return next_wp_id
        
        # Get the previous and the next waypoint ids
        next_wp_id = find_next_point_idx(state, polyline)
        prev_wp_id = max(next_wp_id - 1, 0)
        # vector n from previous waypoint to next waypoint
        n_x = polyline[next_wp_id, 0] - polyline[prev_wp_id, 0]
        n_y = polyline[next_wp_id, 1] - polyline[prev_wp_id, 1]
        # vector x from previous waypoint to current position
        x_x = state.x - polyline[prev_wp_id, 0]
        x_y = state.y - polyline[prev_wp_id, 1]
        x_yaw = np.arctan2(x_y, x_x)
        # find the projection of x on n
        # print(f"numerator: {(x_x * n_x + x_y * n_y)}")
        # print(f"demominator: {(n_x * n_x + n_y * n_y)}")
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

        # calculate d value
        self.d = np.hypot(x_x - proj_x, x_y - proj_y)

        wp_yaw = polyline[prev_wp_id, 2]
        delta_yaw = unifyAngleRange(state.yaw - wp_yaw)

        # if wp_yaw > x_yaw: 
        if wp_yaw <= x_yaw: # CommonRoad
            self.d *= -1

        # calculate s value
        self.s = 0
        for i in range(prev_wp_id):
            self.s += np.hypot(polyline[i+1, 0] - polyline[i, 0], polyline[i+1, 1] - polyline[i, 1])

        # calculate s_d and d_d
        self.t = state.t
        self.s_d = state.v * np.cos(delta_yaw)
        self.s_dd = 0.0
        self.s_ddd = 0.0
        self.d_d = state.v * np.sin(delta_yaw)
        self.d_dd = 0.0
        self.d_ddd = 0.0
        
        return state

class FrenetTrajectory(object):
    """ FrenetTrajectory

    Used internally by the Planner class to generate the trajectory in both frenet frame / world frame
    Do not use this class for the planning outputs

    Attributes
    ------
        `lane_type` (`LaneType`): Left / Middle / Right
        `lane_id` (`int`): id of the lane that the trajectory was sampled on
        `id` (`int`): id of the trajectory itself (i.e. 0-100)
        ...
    """
    def __init__(self):
        self.idx = np.array([-1, -1, -1])   # id of the trajectory itself (i.e. 0-100)
        self.lane_id = -1                   # id of the lane that the trajectory was sampled on
        self.lane_type = LaneType.UNDEFINED # Left / Middle / Right
        
        self.is_generated = False
        self.is_searched= False
        self.constraint_passed = False
        self.collision_passed = False
        self.end_state = None
        
        self.cost_fix = 0.0
        self.cost_dyn = 0.0
        self.cost_heu = 0.0
        self.cost_est = 0.0
        self.cost_final = 0.0
        
        self.t = []
        
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        
        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.c_d = []
        self.c_dd = []
        
    def __eq__(self, other):
        return (self.cost_final == other.cost_final)

    def __ne__(self, other):
        return (self.cost_final != other.cost_final)

    def __lt__(self, other):
        return (self.cost_final < other.cost_final)

    def __le__(self, other):
        return (self.cost_final <= other.cost_final)

    def __gt__(self, other):
        return (self.cost_final > other.cost_final)

    def __ge__(self, other):
        return (self.cost_final >= other.cost_final)

    def __repr__(self):
        return "%f" % (self.cost_final)
    
    def __str__(self):
        return f'FrenetTrajectory with cost_final={self.cost_final:.2f},  d={self.end_state.d:.2f}, s_d={self.end_state.s_d:.2f}, t={self.end_state.t:.2f}'

    def state_at_time_step(self, t: int) -> State:
        """
        Function to get the state of a trajectory at a specific time instance.

        :param time_step: considered time step
        :return: state of the trajectory at time_step
        """
        assert t < len(self.s) and t >= 0

        return State(self.t[t], self.x[t], self.y[t], self.yaw[t], self.s_d[t], self.s_dd[t])
        
    def frenet_state_at_time_step(self, t: int) -> FrenetState:
        """
        Function to get the state of a trajectory at a specific time instance.

        :param time_step: considered time step
        :return: state of the trajectory at time_step
        """
        assert t < len(self.s) and t >= 0

        return FrenetState(self.t[t], 
                           self.s[t], self.s_d[t], self.s_dd[t], self.s_ddd[t],
                           self.d[t], self.d_d[t], self.d_dd[t], self.d_ddd[t])

    def forward_t_steps(self, steps: int):
        
        if steps < 0 or steps >= len(self.t):
            return None
        
        new_traj = copy.deepcopy(self)
        
        new_traj.t = new_traj.t[steps:]
        
        new_traj.s = new_traj.s[steps:]
        new_traj.s_d = new_traj.s_d[steps:]
        new_traj.s_dd = new_traj.s_dd[steps:]
        new_traj.s_ddd = new_traj.s_ddd[steps:]
        new_traj.d = new_traj.d[steps:]
        new_traj.d_d = new_traj.d_d[steps:]
        new_traj.d_dd = new_traj.d_dd[steps:]
        new_traj.d_ddd = new_traj.d_ddd[steps:]
        
        new_traj.x = new_traj.x[steps:]
        new_traj.y = new_traj.y[steps:]
        new_traj.yaw = new_traj.yaw[steps:]
        new_traj.ds = new_traj.ds[steps:]
        new_traj.c = new_traj.c[steps:]
        
        return new_traj