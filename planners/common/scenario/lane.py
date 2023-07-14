from enum import Enum
import numpy as np
# import nav_msgs.msg._Path
# import geometry_msgs.msg
# from tf.transformations import euler_from_quaternion
# from common.geometry.math_utils import 

class LaneType(Enum):
    UNDEFINED = 0
    LEFT = 1
    EGO = 2
    RIGHT = 3
    
class LanePoint(object):
    def __init__(self, x, y, yaw, s=0, width=0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.s = s
        self.width = width

class Lane(object):
    def __init__(self, positions: np.ndarray, orientations: np.ndarray, widths: np.ndarray, s: np.ndarray):
        self.positions = positions
        self.orientations = orientations
        self.widths = widths
        self.s = s

# class Waypoint(object):
#     def Init1(self, x, y, yaw, s=0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw
#         self.s = s
    
#     def InitPose(self, pose, s=0):
#         self.x = pose.x
#         self.y = pose.y
#         self.yaw = euler_from_quaternion(pose.orientation)[2]
#         self.s = 0.0

# class LanePoint(object):
#     def initWithWaypoint(self, point, left_width, right_width, far_left_width, far_right_width):
#         self.left_width = left_width
#         self.right_width = right_width
#         self.far_left_width = far_left_width
#         self.far_right_width = far_right_width
#         self.point = point

#     def initWithPose(self, pose, left_width, right_width, far_left_width, far_right_width):
#         self.left_width = left_width
#         self.right_width = right_width
#         self.far_left_width = far_left_width
#         self.far_right_width = far_right_width
#         wp = Waypoint()
#         wp.InitPose(pose)
#         self.point = wp

# class Lane(object):
#     def __init__(self, ref_path, left_width, right_width, far_left_width, far_right_width):
#         s_total = 0
#         #nav_msgs.msg._Path.Path.poses is an array of PoseStamped[] 
#         self.points = []
#         LP = LanePoint()
#         LP.initWithPose(ref_path.poses[0].pose, left_width, right_width, far_left_width, far_right_width)
#         # Use append() to replace emplace_back()
#         self.points.append(LP)
#         for i in range(1,len[ref_path.poses]):
#             dist = math_utils.distance(ref_path.poses[i-1].pose, ref_path.poses[i].pose)
#             if(dist > 0.01):
#                 s_total += dist
#                 LP = LanePoint()
#                 LP.initWithPose(ref_path.poses[i].pose, left_width, right_width, far_left_width, far_right_width)
#             self.points.append(LP)

#     def clear(self):
#         self.points.clear()

# class Path(object):
#     def __init__(self):
#         self.x = []
#         self.y = []
#         self.yaw = []
#         self.v = []
#     def clear(self):
#         self.x.clear()
#         self.y.clear()
#         self.yaw.clear()
#         self.v.clear()

# def closestWaypoint_Path(current_state, path):
#     closest_dist = 100000.0
#     closest_waypoint = 0 

#     for i in range(len(path.x)):
#         dist = math_utils.distance(current_state.x, current_state.y, path.x[i], path.y[i])
#         if dist < closest_dist:
#             closest_dist = dist
#             closest_waypoint = i
    
#     return closest_waypoint

# def closestWaypoint_Lane(current_state, lane):
#     closest_dist = 100000.0
#     closest_waypoint = 0 

#     for i in range(len(lane.points)):
#         dist = math_utils.distance(current_state.x, current_state.y, lane.points[i].point.x, lane.points[i].point.y)
#         if dist < closest_dist:
#             closest_dist = dist
#             closest_waypoint = i
    
#     return closest_waypoint  

# def nextWaypoint_Path(current_state, path):
#     closestWaypoint = closestWaypoint_Path(current_state, path)
#     heading = math.atan2((path.y[closestWaypoint] - current_state.y), (path.x[closestWaypoint] - current_state.x))
#     angle = abs(current_state.yaw - heading)
#     angle = min(2*math.pi-angle, angle)

#     if angle > math.pi/2:
#         closestWaypoint += 1

#     return closestWaypoint

# def nextWaypoint_Lane(current_state, lane):
#     closestWaypoint = closestWaypoint_Lane(current_state, lane)
#     heading = math.atan2((lane.points[closestWaypoint].point.y - current_state.y), (lane.points[closestWaypoint].point.x - current_state.x))
#     angle = abs(current_state.yaw - heading)
#     angle = min(2*math.pi-angle, angle)

#     if angle > math.pi/2:
#         closestWaypoint += 1

#     return closestWaypoint

# def lastWaypoint_Path(current_state, path):
#     return nextWaypoint_Path(current_state, path) - 1

# def lastWaypoint_Path(current_state, lane):
#     return nextWaypoint_Lane(current_state, lane) - 1



