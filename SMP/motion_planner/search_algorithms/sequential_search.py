import copy
import time
from abc import ABC
from typing import Tuple, Union, Dict, List, Any
import numpy as np

from commonroad.scenario.state import KSState

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.queue import FIFOQueue, LIFOQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization

import SMP.batch_processing.timeout_config

class SequentialSearch(SearchBaseClass, ABC):
    """
    Abstract class for sequential search (BFS/DFS) motion planners.
    """

    # declaration of class variables
    frontier: Union[FIFOQueue, LIFOQueue]
    path_fig: Union[str, None]

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        
    
    def execute_search(self) -> Tuple[Union[None, List[List[KSState]]], Union[None, List[MotionPrimitive]], Any]:
        """
        Implementation of BFS/DFS (tree search) using a FIFO/LIFO queue.
        The adopted frontiers are determined in children classes
        """
        # for visualization in jupyter notebook
        list_status_nodes = []
        dict_node_status: Dict[int, Tuple] = {}

        # shift initial state of planning problem from vehicle center to rear axle position
        # (reference point of motion primitives)
        new_state_initial = self.state_initial.translate_rotate(
            -np.array([self.rear_ax_dist * np.cos(self.state_initial.orientation),
                       self.rear_ax_dist * np.sin(self.state_initial.orientation)]), 0)

        # first node
        node_initial = Node(list_paths=[[new_state_initial]],
                            list_primitives=[self.motion_primitive_initial],
                            depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem,
                              self.config_plot, self.path_fig)

        # check if we have already reached the goal state
        if self.reached_goal(node_initial.list_paths[-1]):
            return self.remove_states_behind_goal(node_initial.list_paths), \
                   node_initial.list_primitives, list_status_nodes

        # we add current node to the frontier
        self.frontier.insert(node_initial)

        dict_node_status = update_visualization(primitive=node_initial.list_paths[-1],
                                                status=MotionPrimitiveStatus.IN_FRONTIER,
                                                dict_node_status=dict_node_status, path_fig=self.path_fig,
                                                config=self.config_plot,
                                                count=len(list_status_nodes))
        list_status_nodes.append(copy.copy(dict_node_status))

        start_time = time.perf_counter()
        while not self.frontier.empty():
            if SMP.batch_processing.timeout_config.use_sequential_processing:
                if (time.perf_counter() - start_time) >= SMP.batch_processing.timeout_config.timeout:
                    raise Exception('Time Out')
            # pop the deepest/shallowest node
            node_current = self.frontier.pop()

            dict_node_status = update_visualization(primitive=node_current.list_paths[-1],
                                                    status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                                    dict_node_status=dict_node_status, path_fig=self.path_fig,
                                                    config=self.config_plot,
                                                    count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_node_status))

            # check all possible successor primitives (i.e., actions) for the current node
            for primitive_successor in node_current.get_successors():

                # translate and rotate motion primitive to current position
                list_primitives_current = copy.copy(node_current.list_primitives)
                path_translated = self.translate_primitive_to_current_state(primitive_successor,
                                                                            node_current.list_paths[-1])

                # check for collision, skip if is not collision-free
                if not self.is_collision_free(path_translated):
                    list_status_nodes, dict_node_status = self.plot_colliding_primitives(current_node=node_current,
                                                                                         path_translated=path_translated,
                                                                                         node_status=dict_node_status,
                                                                                         list_states_nodes=list_status_nodes)
                    continue

                list_primitives_current.append(primitive_successor)

                # goal test
                if self.reached_goal(path_translated):
                    # goal reached
                    path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
                    path_solution = self.remove_states_behind_goal(path_new)
                    list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=dict_node_status,
                                                           list_states_nodes=list_status_nodes)
                    # return solution
                    return self.remove_states_behind_goal(path_new), list_primitives_current, list_status_nodes

                # insert the child to the frontier:
                path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
                child = Node(list_paths=path_new, list_primitives=list_primitives_current,
                             depth_tree=node_current.depth_tree + 1)
                dict_node_status = update_visualization(primitive=path_new[-1],
                                                        status=MotionPrimitiveStatus.IN_FRONTIER,
                                                        dict_node_status=dict_node_status, path_fig=self.path_fig,
                                                        config=self.config_plot,
                                                        count=len(list_status_nodes))
                list_status_nodes.append(copy.copy(dict_node_status))
                self.frontier.insert(child)

                if path_translated[-1].time_step > self.time_desired.end:
                    # prevent algorithm from running infinitely long for DFS (search failed)
                    print('Algorithm is in infinite loop and will not find a solution')
                    return None, None, list_status_nodes

            dict_node_status = update_visualization(primitive=node_current.list_paths[-1],
                                                    status=MotionPrimitiveStatus.EXPLORED,
                                                    dict_node_status=dict_node_status, path_fig=self.path_fig,
                                                    config=self.config_plot,
                                                    count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_node_status))

        return None, None, list_status_nodes


class BreadthFirstSearch(SequentialSearch):
    """
    Class for Breadth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        # using a FIFO queue
        self.frontier = FIFOQueue()

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/bfs/'
        else:
            self.path_fig = None


class DepthFirstSearch(SequentialSearch):
    """
    Class for Depth First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        # using a LIFO queue
        self.frontier = LIFOQueue()

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/dfs/'
        else:
            self.path_fig = None
