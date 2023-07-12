import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import signal
import time

from omegaconf import DictConfig
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import PMState, KSState, CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad_interface.global_planner import GlobalPlanner

from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.utility import plot_primitives
from SMP.motion_planner.utility import create_trajectory_from_list_states

from common.vehicle.vehicle import Vehicle
from common.scenario.frenet import State, FrenetState
from planners.frenet_optimal_planner import Stats, FrenetOptimalPlannerSettings, FrenetOptimalPlanner
from planners.fop_plus_planner import FopPlusPlanner
from planners.fiss_planner import FissPlannerSettings, FissPlanner
from planners.fiss_plus_planner import FissPlusPlannerSettings, FissPlusPlanner

def frenet_optimal_planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig, method: str, num_samples: tuple):
    # Plan a global route
    global_planner = GlobalPlanner()
    global_plan = global_planner.plan_global_route(scenario, planning_problem)
    ego_lane_pts = global_plan.concat_centerline
    
    # Goal
    goal_region = planning_problem.goal

    if goal_region.state_list[0].has_value("velocity"):
        speed_interval = goal_region.state_list[0].velocity
        min_speed = speed_interval.start
        max_speed = speed_interval.end
        print(f"    Speed interval {min_speed}, {max_speed} m/s")
    else:
        min_speed = 0.0
        max_speed = 13.5
        print(f"    Scenario has no speed interval, using {min_speed}, {max_speed} m/s")
        
    goal_lanelet_idx = goal_region.lanelets_of_goal_position[0][0]
    goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(goal_lanelet_idx)
    center_vertices = goal_lanelet.center_vertices
    mid_idx = int((center_vertices.shape[0] - 1)/2)
    goal_center = center_vertices[mid_idx]
    # goal_polygon = goal_lanelet.polygon().shapely_object
    # print("goal_center", goal_center)
        
    # Obstacle lists
    obstacles_static = scenario.static_obstacles
    obstacles_dynamic = scenario.dynamic_obstacles
    obstacles_all = obstacles_static + obstacles_dynamic
    
    obstacle_positions = []
    final_time_step = scenario.dynamic_obstacles[0].prediction.final_time_step
    for t_step in range(final_time_step):
        frame_positions = []
        # frame_obstacles = []
        for obstacle in obstacles_all:
            if obstacle.state_at_time(t_step) is not None:
                frame_positions.append(obstacle.state_at_time(t_step).position)
        obstacle_positions.append(frame_positions)
    
    # Initialize local planner
    vehicle = Vehicle(vehicle_params)
    num_width, num_speed, num_t = num_samples
    
    if method == 'FOP':
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        planner = FrenetOptimalPlanner(planner_settings, vehicle, scenario)
    elif method == 'FOP\'':
        planner_settings = FrenetOptimalPlannerSettings(num_width, num_speed, num_t)
        planner = FopPlusPlanner(planner_settings, vehicle, scenario)
    elif method == 'FISS':
        planner_settings = FissPlannerSettings(num_width, num_speed, num_t)
        planner = FissPlanner(planner_settings, vehicle, scenario)
    elif method == 'FISS+':
        planner_settings = FissPlusPlannerSettings(num_width, num_speed, num_t)
        planner = FissPlusPlanner(planner_settings, vehicle, scenario)
    else:
        print("ERROR: Planning method entered is not recognized!")
        raise ValueError
        
    csp_ego, ref_ego_lane_pts = planner.generate_frenet_frame(ego_lane_pts)

    # Initial state
    initial_state = planning_problem.initial_state
    start_state = State(t=0.0, x=initial_state.position[0], y=initial_state.position[1], 
                        yaw=initial_state.orientation, v=initial_state.velocity, a=initial_state.acceleration)
    current_frenet_state = FrenetState()
    current_frenet_state.from_state(start_state, ref_ego_lane_pts)

    # Start planning in simulation (matplotlib)
    show_animation = False
    area = 20.0  # animation area length [m]

    processing_time = 0
    num_cycles = 0
    state_list = []
    time_list = []
    stats = Stats()
    goal_reached = False
    for i in range(final_time_step):
        num_cycles += 1
        
        # Plan!
        start_time = time.time()
        best_traj_ego = planner.plan(current_frenet_state, max_speed, obstacles_all, i)
        end_time = time.time()
        processing_time += (end_time - start_time)
        stats += planner.stats
        
        if best_traj_ego is None:
            # print("No solution available for problem:", file)
            break
        # Update and record the vehicle's trajectory
        next_step_idx = 1
        current_state = best_traj_ego.state_at_time_step(next_step_idx)
        current_frenet_state = best_traj_ego.frenet_state_at_time_step(next_step_idx)
        state = CustomState(**{'time_step': i,
                               'position': np.array([current_state.x, current_state.y]),
                               'orientation': current_state.yaw,
                               'velocity': current_frenet_state.s_d,
                               'velocity_y': current_frenet_state.d_d,
                               # 'steering_angle': None
                               })
        state_list.append(state)
        time_list.append(end_time - start_time)
        
        # Verify if the goal has been reached
        if goal_region.is_reached(state):
            print("Goal Reached")
            goal_reached = True
            break
        # if goal_polygon.contains_properly()
        elif np.hypot(state.position[0] - goal_center[0], state.position[1] - goal_center[1]) <= vehicle.l/2:
            print("    Goal Reached")
            goal_reached = True
            break
        elif np.hypot(state.position[0] - ref_ego_lane_pts[-1, 0], state.position[1] - ref_ego_lane_pts[-1, 1]) <= 3.0:
            print("    Reaching End of the Map, Stopping, Goal Not Reached")
            goal_reached = True
            break
        
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(ref_ego_lane_pts[:, 0], ref_ego_lane_pts[:, 1])
            if len(obstacle_positions) > i:
                # print('total time steps:', len(obstacle_positions), '# of obstacle:', obstacle_markers.shape[0])
                obstacle_markers = np.array(obstacle_positions[i])
                plt.plot(obstacle_markers[:, 0], obstacle_markers[:, 1], "X")
                plt.plot(best_traj_ego.x[next_step_idx:], best_traj_ego.y[next_step_idx:], "-or")
                plt.plot(best_traj_ego.x[next_step_idx], best_traj_ego.y[next_step_idx], "vc")
                plt.xlim(best_traj_ego.x[next_step_idx] - area, best_traj_ego.x[next_step_idx] + area)
                plt.ylim(best_traj_ego.y[next_step_idx] - area, best_traj_ego.y[next_step_idx] + area)

                plt.title("v[km/h]:" + str(best_traj_ego.s_d[next_step_idx] * 3.6)[0:4])
                plt.grid(True)
                plt.pause(0.0001)
            # else:
            #     print("Error!")
            #     raise BaseException
            
    # print("Success!")
    avg_processing_time = processing_time / num_cycles
    stats.average(num_cycles)
    
    if show_animation and best_traj_ego is not None:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()

    # create the planned trajectory starting at time step 0
    if state_list:
        ego_vehicle_traj = Trajectory(initial_time_step=0, state_list=state_list)
    else: 
        ego_vehicle_traj = None
    
    return goal_reached, ego_vehicle_traj, avg_processing_time, time_list, stats, planner.all_trajs

def timeout_handler(signum, frame):
    raise BaseException("Program exceeded 10 seconds")

def informed_planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    # load the xml with stores the 524 motion primitives
    name_file_motion_primitives = 'V_0.0_20.0_Vstep_2.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # load the xml with stores the 167 motion primitives
    name_file_motion_primitives = 'V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # construct motion planner
    type_motion_planner = MotionPlannerType.GBFS # UCS, ASTAR, STUDENT_EXAMPLE
    motion_planner = MotionPlanner.create(scenario=scenario, 
                                        planning_problem=planning_problem,
                                        automaton=automaton, 
                                        motion_planner_type=type_motion_planner)

    # solve for solution
    start_time = time.time()
    list_paths_primitives, _, _ = motion_planner.execute_search()
    end_time = time.time()
    processing_time = end_time - start_time

    ego_vehicle_trajectory = create_trajectory_from_list_states(list_paths_primitives, vehicle_params.b)
    return True, ego_vehicle_trajectory, processing_time, None

def planning(scenario: Scenario, planning_problem: PlanningProblem, vehicle_params: DictConfig, method: str, num_samples: tuple):
    if method == 'informed':
        return informed_planning(scenario, planning_problem, vehicle_params)
    else:
        return frenet_optimal_planning(scenario, planning_problem, vehicle_params, method, num_samples)