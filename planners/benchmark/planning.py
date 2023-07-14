import os
import signal
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import VehicleType
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from matplotlib.collections import LineCollection
from omegaconf import DictConfig
from PIL import Image

from planners.common.scenario.frenet import FrenetState, State
from planners.common.vehicle.vehicle import Vehicle
from planners.commonroad_interface.global_planner import GlobalPlanner
from planners.fiss_planner import FissPlanner, FissPlannerSettings
from planners.fiss_plus_planner import FissPlusPlanner, FissPlusPlannerSettings
from planners.fop_plus_planner import FopPlusPlanner
from planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from SMP.motion_planner.utility import create_trajectory_from_list_states


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
        print(
            f"    Scenario has no speed interval, using {min_speed}, {max_speed} m/s")

    goal_lanelet_idx = goal_region.lanelets_of_goal_position[0][0]
    goal_lanelet = scenario.lanelet_network.find_lanelet_by_id(
        goal_lanelet_idx)
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
        planner_settings = FrenetOptimalPlannerSettings(
            num_width, num_speed, num_t)
        planner = FrenetOptimalPlanner(planner_settings, vehicle, scenario)
    elif method == 'FOP+':
        planner_settings = FrenetOptimalPlannerSettings(
            num_width, num_speed, num_t)
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
        best_traj_ego = planner.plan(
            current_frenet_state, max_speed, obstacles_all, i)
        end_time = time.time()
        processing_time += (end_time - start_time)
        stats += planner.stats

        if best_traj_ego is None:
            # print("No solution available for problem:", file)
            break
        # Update and record the vehicle's trajectory
        next_step_idx = 1
        current_state = best_traj_ego.state_at_time_step(next_step_idx)
        current_frenet_state = best_traj_ego.frenet_state_at_time_step(
            next_step_idx)
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
                plt.plot(best_traj_ego.x[next_step_idx:],
                         best_traj_ego.y[next_step_idx:], "-or")
                plt.plot(best_traj_ego.x[next_step_idx],
                         best_traj_ego.y[next_step_idx], "vc")
                plt.xlim(best_traj_ego.x[next_step_idx] -
                         area, best_traj_ego.x[next_step_idx] + area)
                plt.ylim(best_traj_ego.y[next_step_idx] -
                         area, best_traj_ego.y[next_step_idx] + area)

                plt.title(
                    "v[km/h]:" + str(best_traj_ego.s_d[next_step_idx] * 3.6)[0:4])
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
        ego_vehicle_traj = Trajectory(
            initial_time_step=0, state_list=state_list)
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
    automaton = ManeuverAutomaton.generate_automaton(
        name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # load the xml with stores the 167 motion primitives
    name_file_motion_primitives = 'V_0.0_20.0_Vstep_4.0_SA_-1.066_1.066_SAstep_0.18_T_0.5_Model_BMW_320i.xml'
    # generate automaton
    automaton = ManeuverAutomaton.generate_automaton(
        name_file_motion_primitives)
    # plot motion primitives
    # plot_primitives(automaton.list_primitives)

    # construct motion planner
    type_motion_planner = MotionPlannerType.GBFS  # UCS, ASTAR, STUDENT_EXAMPLE
    motion_planner = MotionPlanner.create(scenario=scenario,
                                          planning_problem=planning_problem,
                                          automaton=automaton,
                                          motion_planner_type=type_motion_planner)

    # solve for solution
    start_time = time.time()
    list_paths_primitives, _, _ = motion_planner.execute_search()
    end_time = time.time()
    processing_time = end_time - start_time

    ego_vehicle_trajectory = create_trajectory_from_list_states(
        list_paths_primitives, vehicle_params.b)
    return True, ego_vehicle_trajectory, processing_time, None


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def planning(cfg: dict, output_dir: str, input_dir: str, file: str) -> None:
    # Global benchmark settings
    method = cfg['PLANNER']  # 'informed', 'FOP', 'FOP+', 'FISS', 'FISS+'
    num_samples = (cfg['N_W_SAMPLE'], cfg['N_S_SAMPLE'], cfg['N_W_SAMPLE'])
    save_gif = cfg['SAVE_GIF']

    vehicle_type = VehicleType.VW_VANAGON  # FORD_ESCORT, BMW_320i, VW_VANAGON
    vehicle_params = VehicleParameterMapping[vehicle_type.name].value

    ##################################################### Planning #########################################################
    # Read the Commonroad scenario
    file_path = os.path.join(input_dir, file)
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
    planning_problem = list(
        planning_problem_set.planning_problem_dict.values())[0]
    initial_state = planning_problem.initial_state

    try:
        # Plan!
        if method == 'informed':
            _, ego_vehicle_trajectory, _, time_list = informed_planning(
                scenario, planning_problem, vehicle_params)
        else:
            _, ego_vehicle_trajectory, _, time_list, _, fplist = frenet_optimal_planning(
                scenario, planning_problem, vehicle_params, method, num_samples)

        if ego_vehicle_trajectory is None:
            print("No ego vehicle trajectory found")
            raise RuntimeError

        # The ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_shape = Rectangle(
            length=vehicle_params.l, width=vehicle_params.w)
        ego_vehicle_prediction = TrajectoryPrediction(
            trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)
        ego_vehicle_type = ObstacleType.CAR
        ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                      obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                                      prediction=ego_vehicle_prediction)

    except RuntimeError:
        print("   ", f"{file} not feasible!")
        return

    ##################################################### Visualization #########################################################
    if save_gif and fplist:
        images = []
        # For each
        for i in range(len(fplist)):
            plt.figure(figsize=(25, 10))
            mpl.rcParams['font.size'] = 20
            rnd = MPRenderer()
            rnd.draw_params.time_begin = i
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "g"
            ego_vehicle.draw(rnd)
            planning_problem_set.draw(rnd)
            rnd.render()
            costs = []
            xs = []
            ys = []
            for fp in fplist[i]:
                costs.append(fp.cost_final)
                xs.append(fp.x[1:])
                ys.append(fp.y[1:])
            lc = multiline(xs, ys, costs, ax=rnd.ax,
                           cmap='RdYlGn_r', lw=2, zorder=20)
            plt.colorbar(lc)

            x_coords = [state.position[0]
                        for state in ego_vehicle_trajectory.state_list]
            y_coords = [state.position[1]
                        for state in ego_vehicle_trajectory.state_list]
            x_coords_p = [state.position[0]
                          for state in ego_vehicle_trajectory.state_list[0:i]]
            y_coords_p = [state.position[1]
                          for state in ego_vehicle_trajectory.state_list[0:i]]
            x_coords_f = [state.position[0]
                          for state in ego_vehicle_trajectory.state_list[i:]]
            y_coords_f = [state.position[1]
                          for state in ego_vehicle_trajectory.state_list[i:]]
            dx_ego_f = np.diff(x_coords_f)
            dy_ego_f = np.diff(y_coords_f)
            rnd.ax.plot(x_coords_p, y_coords_p, color='#9400D3',
                        alpha=1,  zorder=25, lw=1)
            rnd.ax.plot(x_coords_f, y_coords_f, color='#AFEEEE',
                        alpha=1,  zorder=25, lw=1)
            rnd.ax.quiver(x_coords_f[:-1:5], y_coords_f[:-1:5], dx_ego_f[::5], dy_ego_f[::5],
                          scale_units='xy', angles='xy', scale=1, width=0.009, color='#AFEEEE', zorder=26)

            x_min = min(x_coords)-8
            x_max = max(x_coords)+8
            y_min = min(y_coords)-8
            y_max = max(y_coords)+8
            l = max(x_max-x_min, y_max-y_min)

            if l == x_max - x_min:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min - (l-(y_max-y_min))/2,
                         y_max + (l-(y_max-y_min))/2)
            else:
                plt.xlim(x_min - (l-(x_max-x_min))/2,
                         x_max + (l-(x_max-x_min))/2)
                plt.ylim(y_min, y_max)

            for obs in scenario.dynamic_obstacles:
                t = 0
                obs_traj_x = []
                obs_traj_y = []
                while obs.state_at_time(t) is not None:
                    obs_traj_x.append(obs.state_at_time(t).position[0])
                    obs_traj_y.append(obs.state_at_time(t).position[1])
                    t += 1
                dx = np.diff(obs_traj_x)
                dy = np.diff(obs_traj_y)
                obs_traj_x = obs_traj_x[:-1]
                obs_traj_y = obs_traj_y[:-1]
                rnd.ax.quiver(obs_traj_x[:i:5], obs_traj_y[:i:5], dx[:i:5], dy[:i:5],
                              scale_units='xy', angles='xy', scale=1, width=0.006, color='#BA55D3', zorder=25)
                rnd.ax.quiver(obs_traj_x[i::5], obs_traj_y[i::5], dx[i::5], dy[i::5],
                              scale_units='xy', angles='xy', scale=1, width=0.006, color='#1d7eea', zorder=25)
                rnd.ax.plot(obs_traj_x[0:i], obs_traj_y[0:i],
                            color='#BA55D3', alpha=0.8,  zorder=25, lw=0.6)
                rnd.ax.plot(obs_traj_x[i:], obs_traj_y[i:],
                            color='#1d7eea', alpha=0.8,  zorder=25, lw=0.6)
            time_list.append(0)

            plt.title("{method}: {time}s".format(
                method=method, time=round(time_list[i], 3)))
            scenario_id = os.path.splitext(file)[0]
            plt.suptitle(f'Scenario ID: {scenario_id}',
                         fontsize=20, x=0.59, y=0.06)

            # Write the figure into a jpg file
            result_path = os.path.join(
                output_dir, 'gif_cache', method, scenario_id)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print("Target directory: {} Created".format(result_path))
            fig_path = os.path.join(
                result_path, "{time_step}.jpg".format(time_step=i))
            plt.savefig(fig_path, dpi=200, bbox_inches='tight')
            print("Fig saved to:", fig_path)

            # plt.show()
            plt.close()

            images.append(Image.open(fig_path))

        # Genereate a gif file from the previously saved jpg files
        gif_dirpath = os.path.join(output_dir, 'gif/', method)
        if not os.path.exists(gif_dirpath):
            os.makedirs(gif_dirpath)
            print("Target directory: {} Created".format(gif_dirpath))
        gif_filepath = os.path.join(gif_dirpath, f"{scenario_id}.gif")
        images[0].save(gif_filepath, save_all=True,
                       append_images=images[1:], optimize=True, duration=100, loop=0)
        print("Gif saved to:", gif_filepath)

    return
