__author__ = "Anna-Katharina Rettinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["CoPlan"]
__version__ = "0.1"
__maintainer__ = "Anna-Katharina Rettinger"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Beta"

import enum
from typing import List, Dict, Tuple, Type

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from IPython.display import display
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem, PlanningProblemSet
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import ObstacleType, DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import KSState, InitialState, CustomState
from commonroad.visualization.draw_params import DynamicObstacleParams
# import CommonRoad-io modules
from commonroad.visualization.mp_renderer import MPRenderer
from ipywidgets import widgets
from matplotlib.lines import Line2D

# import Motion Automaton modules
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.plot_config import PlotConfig

list_states_nodes = None


@enum.unique
class MotionPrimitiveStatus(enum.Enum):
    IN_FRONTIER = 0
    INVALID = 1
    CURRENTLY_EXPLORED = 2
    EXPLORED = 3
    SOLUTION = 4


def plot_legend(plotting_config: Type[PlotConfig]):
    if hasattr(plotting_config, 'LABELS'):
        node_status, labels = plotting_config.LABELS
    else:
        node_status = [status.value for status in MotionPrimitiveStatus]
        labels = ['Frontier', 'Collision', 'Currently Exploring', 'Explored', 'Final Solution']

    custom_lines = []
    for value in node_status:
        custom_lines.append(Line2D([0], [0], color=plotting_config.PLOTTING_PARAMS[value][0],
                                   linestyle=plotting_config.PLOTTING_PARAMS[value][1],
                                   linewidth=plotting_config.PLOTTING_PARAMS[value][2]))
    legend = plt.legend(handles=custom_lines, labels=labels, loc='lower left',
                        bbox_to_anchor=(0.02, 0.51), prop={'size': 18})
    legend.set_zorder(30)
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.shadow"] = True


def plot_search_scenario(scenario, initial_state, ego_shape, planning_problem, config: Type[PlotConfig]):
    plt.figure(figsize=(22.5, 4.5))
    plt.axis('equal')
    renderer = MPRenderer(plot_limits=[55, 100, -2.5, 5.5])
    renderer.draw_params.lanelet_network.lanelet.facecolor = '#F8F8F8'
    scenario.draw(renderer)
    ego_vehicle = DynamicObstacle(obstacle_id=scenario.generate_object_id(), obstacle_type=ObstacleType.CAR,
                                  obstacle_shape=ego_shape,
                                  initial_state=initial_state)
    ego_vehicle.draw(renderer)
    planning_problem.draw(renderer)

    if config.PLOT_LEGEND:
        plot_legend(plotting_config=config)

    renderer.render()


def initial_visualization(scenario, initial_state, ego_shape, planning_problem, config: Type[PlotConfig], path_fig):
    # only plot if run with python script
    if not config.JUPYTER_NOTEBOOK and config.DO_PLOT:
        plot_search_scenario(scenario, initial_state, ego_shape, planning_problem, config)
        if path_fig:
            plt.rcParams['svg.fonttype'] = 'none'
            plt.axis('off')
            try:
                plt.savefig(path_fig + 'initial_scenario.' + config.OUTPUT_FORMAT, format=config.OUTPUT_FORMAT,
                            bbox_inches='tight')
            except:
                print('Saving was not successful')
        else:
            plt.show(block=False)


def plot_state(state: KSState, color='red'):
    plt.plot(state.position[0], state.position[1], color=color, marker='o', markersize=6)


def plot_motion_primitive(mp: MotionPrimitive, color='red'):
    """
    Plots an object of class MotionPrimitive with marker at initial state and end state
    @param mp: object of class Motion Primitive
    @param color: color for plotting Motion Primitive
    @return:
    """
    plot_state(state=mp.trajectory.state_list[0])
    plot_state(state=mp.trajectory.state_list[-1])
    x = []
    y = []
    for state in mp.trajectory.state_list:
        x.append(state.position[0])
        y.append(state.position[1])
    plt.plot(x, y, color=color, marker="")


def plot_primitive_path(mp: List[KSState], status: MotionPrimitiveStatus, plotting_params):
    plt.plot(mp[-1].position[0] + 1.42 * np.cos(mp[-1].orientation),
             mp[-1].position[1] + 1.42 * np.sin(mp[-1].orientation),
             color=plotting_params[status.value][0],
             marker='o',  markersize=8, zorder=27)
    x = []
    y = []
    for state in mp:
        x.append(state.position[0] + 1.42 * np.cos(state.orientation))
        y.append(state.position[1] + 1.42 * np.sin(state.orientation))
    plt.plot(x, y, color=plotting_params[status.value][0], marker="", linestyle=plotting_params[status.value][1],
             linewidth=plotting_params[status.value][2], zorder=25)


def update_visualization(primitive: List[KSState], status: MotionPrimitiveStatus, dict_node_status: Dict[int, Tuple],
                         path_fig, config, count, time_pause=0.4):
    assert isinstance(status, MotionPrimitiveStatus), "Status is not of type MotionPrimitiveStatus."
    # print("primitive")
    # print(primitive)
    dict_node_status.update({hash(primitive[-1]): (primitive, status)})
    # print("test")
    # print(dict_node_status)
    # only plot if run with python script
    if not config.JUPYTER_NOTEBOOK and config.DO_PLOT:
        plot_primitive_path(mp=primitive, status=status, plotting_params=config.PLOTTING_PARAMS)
        if path_fig:
            plt.axis('off')
            plt.rcParams['svg.fonttype'] = 'none'
            try:
                plt.savefig(path_fig + 'solution_step_' + str(count) + '.' + config.OUTPUT_FORMAT,
                            format=config.OUTPUT_FORMAT, bbox_inches='tight')
            except:
                print('Saving was not successful')
        else:
            plt.pause(time_pause)

    return dict_node_status


def show_scenario(scenario_data: Tuple[Scenario, InitialState, Rectangle, PlanningProblem], node_status: Dict[int, Tuple],
                  config):
    plot_search_scenario(scenario=scenario_data[0], initial_state=scenario_data[1], ego_shape=scenario_data[2],
                         planning_problem=scenario_data[3], config=config)
    for node in node_status.values():
        plot_primitive_path(node[0], node[1], config.PLOTTING_PARAMS)

    plt.show()


def display_steps(scenario_data, config, algorithm, **args):
    def slider_callback(iteration):
        # don't show graph for the first time running the cell calling this function
        try:
            show_scenario(scenario_data, node_status=list_states_nodes[iteration], config=config)

        except:
            # better add some error/print some information
            pass

    def visualize_callback(Visualize):
        if Visualize is True:
            button.value = False
            global list_states_nodes

            if 'limit_depth' in args:
                path, primitives, list_states_nodes = algorithm(limit_depth=args['limit_depth'])
            else:
                path, primitives, list_states_nodes = algorithm()

            slider.max = len(list_states_nodes) - 1

            for i in range(slider.max + 1):
                slider.value = i
                # time.sleep(.5)

    slider = widgets.IntSlider(min=0, max=1, step=1, value=0)
    slider_visual = widgets.interactive(slider_callback, iteration=slider)
    # noinspection PyTypeChecker
    display(slider_visual)

    button = widgets.ToggleButton(value=False)
    button_visual = widgets.interactive(visualize_callback, Visualize=button)
    # noinspection PyTypeChecker
    display(button_visual)


def plot_primitives(list_primitives, figsize=(12, 3)):
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    for primitive in list_primitives:
        list_x = [state.position[0] for state in primitive.trajectory.state_list]
        list_y = [state.position[1] for state in primitive.trajectory.state_list]

        list_x = [primitive.state_initial.x] + list_x + [primitive.state_final.x]
        list_y = [primitive.state_initial.y] + list_y + [primitive.state_final.y]

        plt.plot(list_x, list_y)

    ax.set_xticks(np.arange(-5, 20, 0.5))
    ax.set_yticks(np.arange(-5, 5., 0.5))

    plt.axis('equal')
    plt.grid(alpha=0.5)
    plt.show()


def create_trajectory_from_list_states(list_paths_primitives: List[List[KSState]], rear_ax_dist) -> Trajectory:
    # turns the solution (list of lists of states) into a CommonRoad Trajectory
    # positions of the states have to be shifted to the vehicle center since the Motion Primitives positions refer
    # to the reference point (rear axis) of the KS Model
    list_states = list()

    for path_primitive in list_paths_primitives:
        for state in path_primitive:
            # kwarg = {'position': state.position + np.array([rear_ax_dist * np.cos(state.orientation),
            #                                                 rear_ax_dist * np.sin(state.orientation)]),
            #          'velocity': state.velocity,
            #          'steering_angle': state.steering_angle,
            #          'orientation': state.orientation,
            #          'time_step': state.time_step}
            # list_states.append(KSState(**kwarg))
            list_states.append(CustomState(**{'position': state.position + np.array([rear_ax_dist * np.cos(state.orientation), rear_ax_dist * np.sin(state.orientation)]),
                                'orientation': state.orientation,
                                'time_step': state.time_step,
                                'velocity': state.velocity * np.cos(state.orientation),
                                'velocity_y': state.velocity * np.sin(state.orientation)
                                }))

    # remove duplicates. the primitive have 6 states, thus a duplicate appears every 6 states
    list_states = [list_states[i] for i in range(len(list_states)) if i % 6 != 1]

    # return Trajectory(initial_time_step=list_states[0].time_step, state_list=list_states)
    return Trajectory(initial_time_step=list_states[0].time_step, state_list=list_states)


def visualize_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, trajectory: Trajectory) -> None:
    from IPython import display

    num_time_steps = len(trajectory.state_list)

    # create the ego vehicle prediction using the trajectory and the shape of the obstacle
    dynamic_obstacle_initial_state = trajectory.state_list[0]
    dynamic_obstacle_shape = Rectangle(width=1.8, length=4.3)
    dynamic_obstacle_prediction = TrajectoryPrediction(trajectory, dynamic_obstacle_shape)

    # generate the dynamic obstacle according to the specification
    dynamic_obstacle_id = scenario.generate_object_id()
    dynamic_obstacle_type = ObstacleType.CAR
    ego_vehicle = DynamicObstacle(dynamic_obstacle_id,
                                       dynamic_obstacle_type,
                                       dynamic_obstacle_shape,
                                       dynamic_obstacle_initial_state,
                                       dynamic_obstacle_prediction)

    # visualize scenario
    ego_params = DynamicObstacleParams()
    ego_params.vehicle_shape.occupancy.shape.facecolor = "green"

    for i in range(0, num_time_steps):
        display.clear_output(wait=True)
        plt.figure(figsize=(25, 10))
        renderer = MPRenderer()
        renderer.draw_params.time_begin = i
        scenario.draw(renderer)

        ego_params.time_begin = i
        ego_params.trajectory.draw_trajectory = True
        ego_params.trajectory.facecolor = '#ff00ff'
        ego_params.trajectory.draw_continuous = True
        ego_params.trajectory.zorder = 60
        ego_params.trajectory.line_width = 1

        ego_vehicle.draw(renderer, draw_params=ego_params)
        planning_problem_set.draw(renderer)
        plt.gca().set_aspect('equal')
        renderer.render()
        plt.show()