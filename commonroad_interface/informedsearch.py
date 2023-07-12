import os
import sys
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.obstacle import StaticObstacle, ObstacleType
from commonroad.scenario.state import CustomState

from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.utility import plot_primitives

# type_motion_planner = MotionPlannerType.UCS
type_motion_planner = MotionPlannerType.GBFS
# type_motion_planner = MotionPlannerType.ASTAR
# type_motion_planner = MotionPlannerType.STUDENT_EXAMPLE

# your own motion planner can be called by uncommenting next line
# type_motion_planner = MotionPlannerType.STUDENT

# read in scenario and planning problem set

path = 'data/test/DEU_Flensburg-1_1_T-1.xml'
#path = '/home/jidan/AVWorkSpace/CommonRoad/commonroad-search/scenarios/exercise/USA_Lanker-1_2_T-1.xml'
scenario, planning_problem_set = CommonRoadFileReader(path).open()
# retrieve the first planning problem in the problem set
planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]

# # generate the static obstacle according to the specification, refer to API for details of input parameters
# static_obstacle_id = scenario.generate_object_id()
# static_obstacle_type = ObstacleType.PARKED_VEHICLE
# static_obstacle_shape = Rectangle(width = 2.0, length = 4.5)
# static_obstacle_initial_state = CustomState(position = numpy.array([0, -18]), orientation = 0.02, time_step = 0)

# # feed in the required components to construct a static obstacle
# static_obstacle = StaticObstacle(static_obstacle_id, static_obstacle_type, static_obstacle_shape, static_obstacle_initial_state)

# # add the static obstacle to the scenario
# scenario.add_objects(static_obstacle)

# plot the scenario for each time step
for i in range(0, 50):
    if i != 1:
        continue
    plt.figure(figsize=(10, 10))
    renderer = MPRenderer()
    
    # uncomment the following line to visualize with animation
    clear_output(wait=True)
    
    # plot the scenario for each time step
    renderer.draw_params.time_begin = i
    
    scenario.draw(renderer)
    # plot the planning problem set
    planning_problem_set.draw(renderer)
    
    renderer.render()
    plt.show()

# load the xml with stores the 524 motion primitives
name_file_motion_primitives = 'V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
# generate automaton
automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)
# plot motion primitives
plot_primitives(automaton.list_primitives)

# load the xml with stores the 167 motion primitives
name_file_motion_primitives = 'V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
# generate automaton
automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)
# plot motion primitives
plot_primitives(automaton.list_primitives)

# construct motion planner
motion_planner = MotionPlanner.create(scenario=scenario, 
                                      planning_problem=planning_problem,
                                      automaton=automaton, 
                                      motion_planner_type=type_motion_planner)

# solve for solution
print(f"begin")
list_paths_primitives, _, _ = motion_planner.execute_search()
print(f"end")
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import KSState
from SMP.motion_planner.utility import create_trajectory_from_list_states
from commonroad.common.solution import Solution, PlanningProblemSolution, \
                                       VehicleModel, VehicleType, CostFunction
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

vehicle_type = VehicleType.BMW_320i
params = VehicleParameterMapping[vehicle_type.name].value

trajectory_solution = create_trajectory_from_list_states(list_paths_primitives, params.b)

from SMP.motion_planner.utility import visualize_solution
visualize_solution(scenario, planning_problem_set, trajectory_solution)

# create PlanningProblemSolution object
kwarg = {'planning_problem_id': planning_problem.planning_problem_id,
         'vehicle_model':VehicleModel.KS,                            # used vehicle model, change if needed
         'vehicle_type':VehicleType.BMW_320i,                        # used vehicle type, change if needed
         'cost_function':CostFunction.SA1,                           # cost funtion, DO NOT use JB1
         'trajectory':trajectory_solution}

planning_problem_solution = PlanningProblemSolution(**kwarg)

# create Solution object
kwarg = {'scenario_id':scenario.scenario_id,
         'planning_problem_solutions':[planning_problem_solution]}

solution = Solution(**kwarg)

# checking validity
from commonroad_dc.feasibility.solution_checker import valid_solution
print(valid_solution(scenario, planning_problem_set, solution))

# saving planning results
from commonroad.common.solution import CommonRoadSolutionWriter

dir_output = "data/solution/"

# create directory if not exists
if not os.path.exists(os.path.dirname(dir_output)):
    print("Please change the output path!")
    # os.makedirs(dir_output, exist_ok=True)

# write solution to a CommonRoad XML file
csw = CommonRoadSolutionWriter(solution)
csw.write_to_file(output_path=dir_output, overwrite=True)