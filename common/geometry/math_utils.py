# import math
import numpy as np
from scipy.spatial.transform import Rotation

def mps2kph(x):
    return x*3.6

def kph2mps(x):
    return x/3.6

def euler_to_quat(roll, pitch, yaw, degrees=False):
    rot = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return rot.as_quat()

def quate_to_euler(quat, degrees=False):
    rot = Rotation.from_quat(quat)
    return rot.as_euler('xyz', degrees=degrees)

def yaw_to_quat(yaw, degrees=False):
    rot = Rotation.from_euler('z', yaw, degrees=degrees)
    return rot.as_quat()

def quate_to_yaw(quat, degrees=False):
    rot = Rotation.from_quat(quat)
    euler = rot.as_euler('zyx', degrees=degrees)
    return euler[0]

def unifyAngleRange(angle):
    new_angle = angle
    while(new_angle > np.pi):
        new_angle -= 2*np.pi
    while(new_angle < -np.pi):
        new_angle += 2*np.pi
    return new_angle

# def clamp(value, lower_bound, upper_bound):
#     return max(min(value, upper_bound), lower_bound)
# np.clip(value, lower_bound, upper_bound)

# def distance(x1, y1, x2, y2):
#     return ((x2 - x1) ** 2+(y2 - y1) ** 2) ** 0.5
# np.hypot()

# def pose_distance(a, b):
#     return distance(a.position.x, a.position.y, b.position.x, b.position.y)

# def magnitude(x, y, z):
#     return (x**2 + y**2 + z**2)**0.5

# def isLegal(x):
#     return False if (math.isinf(x) or math.isnan(x)) else True