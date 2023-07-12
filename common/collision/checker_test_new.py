
import os
import numpy    
import matplotlib.pyplot as plt
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.state import CustomState
import sat_collision_checker
import new_checker
import common.scenario.obstacle as Obstacle

# load the CommonRoad scenario that has been created in the CommonRoad tutorial
file_path = os.path.join(os.getcwd(), 'data/test/ZAM_Tutorial-1_2_T-1.xml')

scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

# generate the static obstacle according to the specification, refer to API for details of input parameters
static_obstacle_id = scenario.generate_object_id()
static_obstacle_type = ObstacleType.PARKED_VEHICLE
static_obstacle_shape = Rectangle(width = 2.0, length = 4.5)
static_obstacle_initial_state = CustomState(position = numpy.array([40.0, 0.34]), orientation = 0.02, time_step = 0)

# feed in the required components to construct a static obstacle
static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape, static_obstacle_initial_state)

# add the static obstacle to the scenario
scenario.add_objects(static_obstacle)


ego_initialstate = planning_problem_set.find_planning_problem_by_id(100).initial_state.position

s_obstacle = scenario.static_obstacles[1]
# print(s_obstacle.initial_state.position)

d_obstacle0 = scenario.dynamic_obstacles[0]
d_obstacle0Poly = scenario.dynamic_obstacles[0].obstacle_shape.vertices

d_obstacle1 = scenario.dynamic_obstacles[1]
d_obstacle1Poly = scenario.dynamic_obstacles[1].obstacle_shape.vertices

# plot the scenario for each time step
for i in range(0, 40):
    if i != 1:
        continue
    plt.figure(figsize=(25, 10))
    rnd = MPRenderer()
    rnd.draw_params.time_begin = i
    scenario.draw(rnd)
    rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "g"
    planning_problem_set.draw(rnd)
    rnd.render()
    plt.show()

CHECKER = new_checker.SATCollisionChecker()
result = False
for t in range(0,40):
    if t%5 != 1:
        continue
    obs0 = obstacle.obstacle()
    obs0.cr2irl(d_obstacle0, t)
    obs1 = obstacle.obstacle()
    obs1.cr2irl(s_obstacle, t)

    if (CHECKER.check_collision(obs0, obs1) == True):
        print(f"obs0.pos = {numpy.shape(obs0.pos)}")
        print(f"obs1.pos = {numpy.shape(obs1.pos)}")
        print(f"obs0.polygon = {numpy.shape(obs0.polygon)}")
        print(f"obs1.polygon = {numpy.shape(obs1.polygon)}")
        print(f"Collision happened at time step {t}")
        result = True
    
if result == False:
    print("No collision checked")

print("Check done!")