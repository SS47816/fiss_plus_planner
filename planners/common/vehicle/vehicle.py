import numpy as np
from shapely.geometry import Polygon
from planners.common.geometry.math_utils import kph2mps

class Vehicle(object):
    """ class that stores vehicle parameters 

    Used internally by the Planner class 

    Attributes
    ------
        
        ...
    """
    def __init__(self, vehicle_params, safety_factor: float = 1.0):
        # vehicle dimensions
        self.l: float = vehicle_params.l * safety_factor
        self.w: float = vehicle_params.w * safety_factor
        self.h: float = 1.5 * safety_factor
        self.bbox_size: np.ndarray = np.array([self.l, self.w, self.h])         # bounding box size [m]
        # self.baselink_coordinate: np.ndarray = np.array([-1.35, 0.0, -0.55])    # base_link 's coordinates wrt bounding box center [m]
        
        # footprint coordinates (in clockwise direction)
        self.corners: list[tuple[float, float]] = [
                        (self.l/2, self.w/2),                                   # front left corner's coordinates in box_center frame [m]
                        (self.l/2, -self.w/2),                                  # front right corner's coordinates in box_center frame [m]
                        (-self.l/2, -self.w/2),                                 # rear right corner's coordinates in box_center frame [m]
                        (-self.l/2, self.w/2),                                  # rear left corner's coordinates in box_center frame [m]
                        (self.l/2, self.w/2),                                   # front left corner's coordinates in box_center frame [m] (to enclose the polygon)
                        ]
        self.polygon: Polygon = Polygon(self.corners)
        
        # kinematic parameters
        self.a = vehicle_params.a                                               # distance from base_link to the CoG [m]
        self.b = vehicle_params.b                                               # distance from the CoG to front_link [m]
        self.L = self.a + self.b                                                # wheelbase distance [m]
        self.T_f = vehicle_params.T_f                                           # front track w [m]
        self.T_r = vehicle_params.T_r                                           # rear track w [m]
        self.max_speed = vehicle_params.longitudinal.v_max                      # maximum speed [m/s]
        self.max_accel = vehicle_params.longitudinal.a_max                      # maximum acceleration [m/ss]
        self.deccel = - self.max_speed / 5.0                                    # maximum decceleration [m/ss]
        self.max_steering_angle = vehicle_params.steering.max                   # maximum steering angle [rad]
        self.max_steering_rate = vehicle_params.steering.v_max                  # maximum steering rate [rad/s]
        self.max_curvature = np.sin(self.max_steering_angle)/self.L             # maximum curvature [1/m]
        self.max_kappa_d = vehicle_params.steering.kappa_dot_max                # maximum curvature change rate [1/m]
        self.max_kappa_dd = vehicle_params.steering.kappa_dot_dot_max           # maximum curvature change rate rate [1/m]