import logging
import logging.handlers
import multiprocessing
import os
import warnings
from datetime import datetime
from typing import Tuple, Union, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
from commonroad.common.solution import VehicleModel, VehicleType, CostFunction
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction, SetBasedPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib.animation import FuncAnimation

from SMP.batch_processing.process_scenario import ResultType, ResultText, SearchResult
from SMP.batch_processing.scenario_loader import ScenarioLoader
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.motion_planner import MotionPlannerType


def load_config_file(filename) -> dict:
    # open config file
    with open(filename, 'r') as stream:
        try:
            configs = yaml.load(stream, Loader=yaml.BaseLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return configs


def release_logger(logger):
    """
    Releases the logger
    :param logger: the logger to be released
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def initialize_logger(logger_name, config_file) -> logging.Logger:
    # create logger
    logger = logging.getLogger(logger_name)
    release_logger(logger)
    logger.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # add console handler to logger
    # logger.addHandler(console_handler)

    # create and handle log file
    if config_file['logging']['log_to_file'] == 'True':
        log_file_dir = config_file['logging']['log_file_dir']
        date_time_string = ''
        if config_file['logging']['add_timestamp_to_log_file'] == 'True':
            now = datetime.now()  # current date and time
            date_time_string = now.strftime("_%Y_%m_%d_%H-%M-%S")

        # if directory not exists create it
        os.makedirs(log_file_dir, exist_ok=True)

        log_file_path = os.path.join(log_file_dir, config_file['logging']['log_file_name'] + date_time_string + ".log")
        file_handler = logging.handlers.RotatingFileHandler(log_file_path)
        # set the level of logging to file
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    message = "Config file loaded and logger created."
    logger.info(message)
    print(message)
    return logger


def get_last_time_step_in_scenario(scenario: Scenario):
    time_steps = [len(obs.prediction.occupancy_set) for obs in scenario.dynamic_obstacles]
    return max(time_steps)


def get_plot_limits(trajectory: Trajectory, frame_count, zoom=30):
    """
    The plot limits track the center of the ego vehicle.
    """
    num_time_step_trajectory = len(trajectory.state_list)

    dict_plot_limits = dict()
    for i in range(frame_count):
        if i < num_time_step_trajectory:
            state = trajectory.state_list[i]
        else:
            state = trajectory.state_list[num_time_step_trajectory - 1]

        dict_plot_limits[i] = [state.position[0] - zoom,
                               state.position[0] + zoom,
                               state.position[1] - zoom,
                               state.position[1] + zoom]

    return dict_plot_limits


def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def get_default_motion_primitive_base_name_by_veh_id(veh_type_id: int, config_dict):
    # the primitives vary for different vehicle models 1, 2 and 3.
    assert veh_type_id in (1, 2, 3), "Input vehicle type id is not valid! Must be either 1, 2 or 3."

    def_automaton_file = None

    if veh_type_id == 1:
        def_automaton_file = config_dict["default_automaton_files"]["FORD_ESCORT"]
    elif veh_type_id == 2:
        def_automaton_file = config_dict["default_automaton_files"]["BMW_320i"]
    elif veh_type_id == 3:
        def_automaton_file = config_dict["default_automaton_files"]["VW_VANAGON"]

    return def_automaton_file


def get_default_motion_primitive_file_by_veh_id(veh_type_id: int, config_dict):
    def_automaton_file = get_default_motion_primitive_base_name_by_veh_id(veh_type_id, config_dict)

    if def_automaton_file is not None:
        return def_automaton_file + os.path.extsep + 'xml'
    else:
        return None


def get_default_automaton_by_veh_id(veh_type_id: int, config_dict):
    def_automaton_file = get_default_motion_primitive_base_name_by_veh_id(veh_type_id, config_dict)

    if def_automaton_file is not None:
        return def_automaton_file + os.path.extsep + ManeuverAutomaton.extension
    else:
        return None


def parse_vehicle_model(model: str) -> VehicleModel:
    if model == 'PM':
        cr_model = VehicleModel.PM
    elif model == 'ST':
        cr_model = VehicleModel.ST
    elif model == 'KS':
        cr_model = VehicleModel.KS
    elif model == 'MB':
        cr_model = VehicleModel.MB
    else:
        raise ValueError('Selected vehicle model is not valid: {}.'.format(model))
    return cr_model


def parse_vehicle_type(vehicle_type: str) -> Tuple[VehicleType, int]:
    if vehicle_type == 'FORD_ESCORT':
        cr_type = VehicleType.FORD_ESCORT
        cr_type_id = 1
    elif vehicle_type == 'BMW_320i':
        cr_type = VehicleType.BMW_320i
        cr_type_id = 2
    elif vehicle_type == 'VW_VANAGON':
        cr_type = VehicleType.VW_VANAGON
        cr_type_id = 3
    else:
        raise ValueError('Selected vehicle type is not valid: {}.'.format(vehicle_type))

    return cr_type, cr_type_id


def parse_cost_function(cost: str) -> CostFunction:
    if cost == 'JB1':
        cr_cost = CostFunction.JB1
    elif cost == 'SA1':
        cr_cost = CostFunction.SA1
    elif cost == 'WX1':
        cr_cost = CostFunction.WX1
    elif cost == 'SM1':
        cr_cost = CostFunction.SM1
    elif cost == 'SM2':
        cr_cost = CostFunction.SM2
    elif cost == 'SM3':
        cr_cost = CostFunction.SM3
    else:
        raise ValueError('Selected cost function is not valid: {}.'.format(cost))
    return cr_cost


def parse_processes_number(process_count: int) -> int:
    # this fixes the number of worker processes to the interval [1, cpu_count],
    # such that it cannot be run with more cores than available or illegal numbers
    return max(min(process_count, multiprocessing.cpu_count()), 1)


def parse_motion_planner_type(motion_planner_type: str) -> MotionPlannerType:
    if motion_planner_type == MotionPlannerType.BFS.value:
        return MotionPlannerType.BFS
    elif motion_planner_type == MotionPlannerType.DFS.value:
        return MotionPlannerType.DFS
    elif motion_planner_type == MotionPlannerType.UCS.value:
        return MotionPlannerType.UCS
    elif motion_planner_type == MotionPlannerType.GBFS.value:
        return MotionPlannerType.GBFS
    elif motion_planner_type == MotionPlannerType.ASTAR.value:
        return MotionPlannerType.ASTAR
    elif motion_planner_type == MotionPlannerType.DLS.value:
        return MotionPlannerType.DLS
    elif motion_planner_type == MotionPlannerType.STUDENT.value:
        return MotionPlannerType.STUDENT
    elif motion_planner_type == MotionPlannerType.STUDENT_EXAMPLE.value:
        return MotionPlannerType.STUDENT_EXAMPLE
    else:
        raise ValueError(f'Motion planner type <{motion_planner_type}> is not valid!')


def parse_max_tree_depth(tree_depth: int) -> int:
    def_tree_depth = 100
    if tree_depth <= 0:
        warnings.warn(f"Tree depth is equal or less than zero, set the default value {def_tree_depth}")
        return def_tree_depth
    else:
        return tree_depth


def parse_timeout(timeout: int) -> int:
    def_timeout_in_sec = 50
    if timeout <= 0:
        warnings.warn(f"Timeout is equal or less than zero, set the default value: {def_timeout_in_sec} sec")
        return def_timeout_in_sec
    else:
        return timeout


def parse_scenario_config(configuration, scenario, planning_problem_set):
    """
    parse the scenario file and return scenario, planning problem set, vehicle model, etc.
    """
    # default configuration
    vehicle_model = parse_vehicle_model(configuration['default']['vehicle_model'])
    vehicle_type, vehicle_type_id = parse_vehicle_type(configuration['default']['vehicle_type'])
    cost_function = parse_cost_function(configuration['default']['cost_function'])
    planning_problem_idx = int(configuration['default']['planning_problem_idx'])
    planner_id = int(configuration['default']['planner'])
    max_tree_depth = int(configuration['default']['max_tree_depth'])

    # get configuration for each scenario
    if scenario.benchmark_id in configuration.keys():
        # configuration for specific scenario
        try:
            vehicle_model = parse_vehicle_model(configuration[scenario.benchmark_id]['vehicle_model'])
        except KeyError:
            pass
        try:
            vehicle_type, vehicle_type_id = parse_vehicle_type(configuration[scenario.benchmark_id]['vehicle_type'])
        except KeyError:
            pass
        try:
            cost_function = parse_cost_function(configuration[scenario.benchmark_id]['cost_function'])
        except KeyError:
            pass
        try:
            planning_problem_idx = int(configuration[scenario.benchmark_id]['planning_problem_idx'])
        except KeyError:
            pass
        try:
            planner_id = int(configuration[scenario.benchmark_id]['planner'])
        except KeyError:
            pass
        try:
            max_tree_depth = int(configuration[scenario.benchmark_id]['max_tree_depth'])
        except KeyError:
            pass

    return (
        scenario, planning_problem_set, vehicle_type_id, vehicle_type, vehicle_model, cost_function,
        planning_problem_idx,
        planner_id, max_tree_depth)


result_tuple_list = [
    (ResultType.SUCCESS, ResultText.SUCCESS),
    (ResultType.INVALID_SOLUTION, ResultText.INVALID_SOLUTION),
    (ResultType.FAILURE, ResultText.FAILURE),
    (ResultType.EXCEPTION, ResultText.EXCEPTION),
    (ResultType.TIMEOUT, ResultText.TIMEOUT)
]


def log_detailed_scenario_results(logger: logging.Logger, result_dict, verbose: bool = False):
    message = "=" * 30 + f"{'Search Results':^30}" + "=" * 30
    logger.info(message)
    print(message)
    for scenario_id in result_dict["scenarios_to_process"]:
        try:
            if isinstance(result_dict[scenario_id], SearchResult):
                search_result: SearchResult = result_dict[scenario_id]

                if search_result.result == ResultType.TIMEOUT:
                    message = f"[{scenario_id:<30}]\t<{search_result.result.value}>\t Time [ms]:\t{search_result.search_time_ms * 1000:6}"
                    logger.info(message)
                    print(message)
                elif search_result.result == ResultType.EXCEPTION:
                    # TODO format error message to fit on one line ?
                    error_msg = search_result.error_msg if verbose else ""
                    message = f"[{scenario_id:<30}]\t<{search_result.result.value}>\t{error_msg}"
                    logger.info(message)
                    print(message)

                elif search_result.result == ResultType.FAILURE:
                    message = f"[{scenario_id:<30}]\t<{search_result.result.value}>\t Time [ms]:\t{int(search_result.search_time_ms):6}"
                    logger.info(message)
                    print(message)

                else:
                    message = f"[{scenario_id:<30}]\t<{search_result.result.value}>\t Time [ms]:\t{int(search_result.search_time_ms):6}\t"
                    logger.info(message)
                    print(message)
        except KeyError:
            continue


def log_statistics(logger: logging.Logger, result_dict, verbose: bool = False):
    if verbose:
        for result_type, result_text in result_tuple_list:
            str_scenario_list = ""
            for scenario_id in result_dict[result_type]:
                str_scenario_list += f"\n\t- {scenario_id}"

            if str_scenario_list != "":
                message = "=" * 30 + f"{result_text[:-1]:^30}" + "=" * 30 + str_scenario_list
                logger.info(message)
                print(message)

    message = "\n" + "=" * 30 + f"{'Result Summary':^30}" + "=" * 30
    logger.info(message)
    print(message)

    message = f"Total number of scenarios:  \t{result_dict['num_scenarios']:>10}"
    logger.info(message)
    print(message)

    print_current_status(result_dict, logger)


def print_current_status(result_dict, logger: Union[logging.Logger, None] = None):
    if logger is not None:
        for res_type, res_text in result_tuple_list:
            message = f"{res_text:<30}\t{len(result_dict[res_type]):>10}"
            logger.info(message)
            print(message)
    else:
        print("\n" + "=" * 50)
        print(f"Scenario being processed:\t{result_dict['started_processing']:>10}")
        for res_type, res_text in result_tuple_list:
            print(f"{res_text:<30}\t{len(result_dict[res_type]):>10}")

        print("=" * 50)


def init_processing(logger_name: str, for_multi_processing: bool = True):
    # load config file
    configuration = load_config_file(os.path.join(os.path.dirname(__file__), 'batch_processing_config.yaml'))

    # create logger
    logger = initialize_logger(logger_name, configuration)

    # create scenario loader which loads all desired scenarios
    scenario_loader = ScenarioLoader(configuration['setting']['input_path'], configuration, logger)

    # generate automaton
    def_veh_type, def_veh_type_id = parse_vehicle_type(configuration['default']['vehicle_type'])
    def_automaton_file = get_default_automaton_by_veh_id(def_veh_type_id, configuration)
    def_automaton = ManeuverAutomaton.load_automaton(def_automaton_file)
    def_automaton.serialize()
    message = f"Automaton loaded for vehicle type: {def_veh_type.name}"
    logger.info(message)
    print(message)

    message = f"Motion planner: {configuration['default']['planner']}"
    logger.info(message)
    print(message)

    if for_multi_processing:
        manager = multiprocessing.Manager()
        result_dict = manager.dict()
    else:
        result_dict = dict()

    result_dict[ResultType.SUCCESS] = list()
    result_dict[ResultType.INVALID_SOLUTION] = list()
    result_dict[ResultType.FAILURE] = list()
    result_dict[ResultType.EXCEPTION] = list()
    result_dict[ResultType.TIMEOUT] = list()
    result_dict["scenarios_to_process"] = scenario_loader.scenario_ids
    result_dict["num_scenarios"] = len(scenario_loader.scenario_ids)
    result_dict["started_processing"] = 0

    return configuration, logger, scenario_loader, def_automaton, result_dict


# def save_gif(scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory, output_path: str) -> None:
#     print(f"Generating GIF for {scenario.scenario_id} ...")
#
#     # create the ego vehicle prediction using the trajectory and the shape of the obstacle
#     dynamic_obstacle_initial_state = trajectory.state_list[0]
#     dynamic_obstacle_shape = Rectangle(width=1.8, length=4.3)
#     # dynamic_obstacle_prediction = TrajectoryPrediction(trajectory, dynamic_obstacle_shape)
#     dynamic_obstacle_prediction = TrajectoryPrediction(Trajectory(1, trajectory.state_list[1:]), dynamic_obstacle_shape)
#
#     # generate the dynamic obstacle according to the specification
#     dynamic_obstacle_id = scenario.generate_object_id()
#     dynamic_obstacle_type = ObstacleType.CAR
#     dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
#                                        dynamic_obstacle_type,
#                                        dynamic_obstacle_shape,
#                                        dynamic_obstacle_initial_state,
#                                        dynamic_obstacle_prediction)
#
#     # configuration for plotting
#     fig_num = os.getpid()
#     figsize = (15, 15)
#
#     fig = plt.figure(fig_num, figsize=figsize)
#     (ln,) = plt.plot([], [], animated=True)
#
#     frame_count = len(trajectory.state_list)
#     # delay between frames in milliseconds, 1 second * dt to get actual time in ms
#     interval = (1000 * scenario.dt)
#     # add short padding to create a short break before the loop (1 sec)
#     frame_count += int(0.5 / scenario.dt)
#     # a dictionary that holds the plot limits at each time step
#     dict_plot_limits = get_plot_limits(trajectory, frame_count)
#
#     # helper functions for plotting
#     def init_plot():
#         fig.gca().axis('equal')
#         ax = fig.gca()
#         draw_object(scenario, plot_limits=dict_plot_limits[0], ax=ax)
#         draw_object(planning_problem, plot_limits=dict_plot_limits[0], ax=ax)
#         draw_object(dynamic_obstacle, plot_limits=dict_plot_limits[0], ax=ax,
#                     draw_params={'time_begin': 0,
#                                  'dynamic_obstacle': {'shape': {'facecolor': 'green'}}})
#         fig.tight_layout()
#         return (ln,)
#
#     def animate_plot(frame):
#         fig.clf()
#         fig.gca().axis('equal')
#         ax = fig.gca()
#         draw_object(scenario, plot_limits=dict_plot_limits[frame], ax=ax, draw_params={'time_begin': frame})
#         draw_object(planning_problem, plot_limits=dict_plot_limits[frame], ax=ax)
#         draw_object(dynamic_obstacle, plot_limits=dict_plot_limits[frame], ax=ax,
#                     draw_params={'time_begin': frame,
#                                  'dynamic_obstacle': {'shape': {'facecolor': 'green'}}})
#         fig.tight_layout()
#         return (ln,)
#
#     anim = FuncAnimation(fig, animate_plot, frames=frame_count, init_func=init_plot, blit=True, interval=interval)
#
#     file_name = scenario.scenario_id + os.extsep + 'gif'
#     anim.save(os.path.join(output_path, file_name), dpi=30, writer="imagemagick")
#     plt.close(fig)
#     # print(f"{scenario.benchmark_id} GIF saved.")


def save_gif2(scenario: Scenario, planning_problem: PlanningProblem, trajectory: Trajectory, output_path: str) -> None:
    print(f"Generating GIF for {scenario.scenario_id} ...")

    # create the ego vehicle prediction using the trajectory and the shape of the obstacle
    dynamic_obstacle_initial_state = trajectory.state_list[0]
    # todo: use real vehicle size
    dynamic_obstacle_shape = Rectangle(width=1.8, length=4.3)
    dynamic_obstacle_prediction = TrajectoryPrediction(trajectory, dynamic_obstacle_shape)

    # generate the dynamic obstacle according to the specification
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = ObstacleType.CAR
    dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                       dynamic_obstacle_type,
                                       dynamic_obstacle_shape,
                                       dynamic_obstacle_initial_state,
                                       dynamic_obstacle_prediction)

    # configuration for plotting
    fig_num = os.getpid()
    figsize = (15, 15)

    fig = plt.figure(fig_num, figsize=figsize)
    (ln,) = plt.plot([], [], animated=True)

    frame_count = len(trajectory.state_list)
    # delay between frames in milliseconds, 1 second * dt to get actual time in ms
    interval = (1000 * scenario.dt)
    # add short padding to create a short break before the loop (1 sec)
    frame_count += int(0.5 / scenario.dt)
    # a dictionary that holds the plot limits at each time step
    dict_plot_limits = get_plot_limits(trajectory, frame_count)

    fig = plt.figure(fig_num, figsize=figsize)
    (ln,) = plt.plot([], [], animated=True)

    def init_plot():
        fig.gca().axis('equal')
        ax = fig.gca()
        renderer = MPRenderer(plot_limits=dict_plot_limits[0], ax=ax)

        scenario.draw(renderer)
        planning_problem.draw(renderer)
        renderer.draw_params.time_begin = 0
        renderer.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "green"
        dynamic_obstacle.draw(renderer)

        fig.tight_layout()
        renderer.render()
        return (ln,)

    def animate_plot(frame):
        renderer = MPRenderer(plot_limits=dict_plot_limits[frame])
        renderer.draw_params.time_begin = frame
        scenario.draw(renderer)
        planning_problem.draw(renderer)
        renderer.draw_params.time_begin = frame
        renderer.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "green"
        dynamic_obstacle.draw(renderer)

        renderer.render()
        return (ln,)

    anim = FuncAnimation(fig, animate_plot, frames=frame_count, init_func=init_plot, blit=True, interval=interval)

    file_name = str(scenario.scenario_id) + os.extsep + 'gif'
    anim.save(os.path.join(output_path, file_name), dpi=30, writer="imagemagick")
    plt.close(fig)


def str2bool(bool_string: str):
    return bool_string.lower() in ('yes', 'true', 't', 'y', '1')


def get_frame_count(scenario: Scenario) -> int:
    """
    Calculates frame count for a scenario. This is the number of time steps of the longest moving obstacle.
    """
    frame_count = 1
    obstacles = scenario.dynamic_obstacles
    for o in obstacles:
        if type(o.prediction) == SetBasedPrediction:
            frame_count = max(frame_count, len(o.prediction.occupancy_set))
        elif type(o.prediction) == TrajectoryPrediction:
            frame_count = max(frame_count, len(o.prediction.trajectory.state_list))
    return frame_count


def redraw_dynamic_obstacles(dyn_obst_list: List[DynamicObstacle], handles,
                             figure_handle,
                             draw_params=None, plot_limits: Union[List[Union[int, float]], None] = None,
                             draw: bool = True) -> None:
    """
    This function is used for fast updating dynamic obstacles of an already drawn plot. Saves on average about 80% time
    compared to a complete plot.
    Deletes all dynamic obstacles which are specified in handles and draws dynamic obstacles of a scenario.

    :param dyn_obst_list: dynamic obstacles
    :param handles: dict of obstacle_ids and corresponding patch handles (generated by draw_object function)
    :param draw_params:
    :param plot_limits: axis limits for plot [x_min, x_max, y_min, y_max]
    :param figure_handle: figure handle of current plot
    :param draw: if True, updates are displayed directly in figure
    :return: None
    """
    # remove dynamic obstacle from current plot
    for handles_i in handles.values():
        for handle in handles_i:
            if handle is not None:
                handle.remove()
    handles.clear()

    renderer = MPRenderer(plot_limits=plot_limits)
    # redraw dynamic obstacles
    for obs in dyn_obst_list:
        obs.draw(renderer, draw_params=draw_params)
    # draw_object(dyn_obst_list, draw_params=draw_params, plot_limits=plot_limits, handles=handles)

    # update plot
    if draw is True:
        ax = figure_handle.gca()
        for handles_i in handles.values():
            for handle in handles_i:
                if handle is not None:
                    ax.draw_artist(handle)

        if mpl.get_backend() == 'TkAgg':
            figure_handle.canvas.draw()

        elif mpl.get_backend() == 'Qt5Agg':
            figure_handle.canvas.update()
        else:
            try:
                figure_handle.canvas.update()
            except:
                raise Exception(
                    '<plot_helper/redraw_dynamic_obstacles> Backend for matplotlib needs to be \'Qt5Agg\' or \'TkAgg\' but is'
                    '\'%s\'' % mpl.get_backend())

        renderer.render()
        figure_handle.canvas.flush_events()
