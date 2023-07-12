import os
import copy
import math
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.state import CustomState

from common.primitive.trajectory import Trajectory as FrenetPath
from common.geometry.cubic_spline import CubicSpline2D
from common.geometry.polynomial import QuinticPolynomial
from common.geometry.polynomial import QuarticPolynomial
from common_road.global_planner import GlobalPlanner
from vehiclemodels import parameters_vehicle3
from common.collision.new_checker import SATCollisionChecker
from common.collision.obstacle import obstacle


checker = SATCollisionChecker()
ego = obstacle()
ego.pos = np.array([ 298.00639876, -322.73473531])
ego.polygon = np.array([[-2.25, -1],[-2.25, 1],[ 2.25, 1],[ 2.25, -1 ],[-2.25, -1 ]])
ego.yaw = 0.8571511181269675

obs = obstacle()
obs.pos = np.array([ 294.11753, -321.17582])
obs.polygon = np.array([[-2.15105678, -0.8109728 ]
 ,[-2.15105678,  0.8109728 ]
 ,[ 2.15105678,  0.8109728 ]
 ,[ 2.15105678, -0.8109728 ]
 ,[-2.15105678, -0.8109728 ]])
obs.yaw = 0.91369281

print(checker.check_collision(ego,obs))

# Collision between [ 298.00639876 -322.73473531] and [ 294.11753 -321.17582]!!
# ego.poly = [[-2.25 -1.  ]
#  [-2.25  1.  ]
#  [ 2.25  1.  ]
#  [ 2.25 -1.  ]
#  [-2.25 -1.  ]], obs.poly = [[-2.15105678 -0.8109728 ]
#  [-2.15105678  0.8109728 ]
#  [ 2.15105678  0.8109728 ]
#  [ 2.15105678 -0.8109728 ]
#  [-2.15105678 -0.8109728 ]]
# ego.yaw = 0.8571511181269675, obs.yaw = 0.91369281

vehicle3 = parameters_vehicle3.parameters_vehicle3()
ego_vehicle_shape = Rectangle(length=vehicle3.l, width=vehicle3.w)

print(ego_vehicle_shape)
    