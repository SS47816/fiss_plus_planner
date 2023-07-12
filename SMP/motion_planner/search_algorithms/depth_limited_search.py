import copy
import time
from typing import Dict, Tuple, List, Any, Union
import numpy as np

from commonroad.scenario.state import KSState

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.utility import initial_visualization, update_visualization, MotionPrimitiveStatus
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass

import SMP.batch_processing.timeout_config

class DepthLimitedSearch(SearchBaseClass):
    """
    Class for Depth Limited Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        self.start_time = None
        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/dls/'
        else:
            self.path_fig = None
    
   
    def execute_search(self, limit_depth=7) -> Tuple[Union[None, List[List[KSState]]], Union[None, List[MotionPrimitive]], Any]:
        """
        Depth-Limited Search code_tree_search
        """
        # for visualization in jupyter notebook
        list_status_nodes = []
        dict_status_nodes: Dict[int, Tuple] = {}

        # shift initial state of planning problem from vehicle center to rear axle position
        # (reference point of motion primitives)
        new_state_initial = self.state_initial.translate_rotate(
            -np.array([self.rear_ax_dist * np.cos(self.state_initial.orientation),
                       self.rear_ax_dist * np.sin(self.state_initial.orientation)]), 0)

        # First node
        node_initial = Node(list_paths=[[new_state_initial]], list_primitives=[self.motion_primitive_initial],
                            depth_tree=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem, self.config_plot,
                              self.path_fig)

        dict_status_nodes = update_visualization(primitive=node_initial.list_paths[-1],
                                                 status=MotionPrimitiveStatus.IN_FRONTIER,
                                                 dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                 config=self.config_plot,
                                                 count=len(list_status_nodes))
        list_status_nodes.append(copy.copy(dict_status_nodes))
        if SMP.batch_processing.timeout_config.use_sequential_processing:
            self.start_time = time.perf_counter()
        result = self.recursive_dls(list_status_nodes, dict_status_nodes, node_initial, limit_depth)

        if result is None:
            return None, None, list_status_nodes
        else:
            path = result[0]
            list_primitives = result[1]
            list_status_nodes = result[2]

            return path, list_primitives, list_status_nodes

    def recursive_dls(self, list_status_nodes: List[Dict[int, Tuple]], dict_status_nodes: Dict[int, Tuple],
                      node_current: Node,
                      limit: int):
        """
        Recursive code_tree_search of Depth-Limited Search.
        @param list_status_nodes: List which stores the changes in the node _status for plotting
        @param dict_status_nodes: Dict with status of each node
        @param node_current: consists of path, list of primitives and the current tree_depth
        @param limit: current Limit
        @return:
        """
        if SMP.batch_processing.timeout_config.use_sequential_processing:
            current_time = time.perf_counter()
            if (current_time - self.start_time) >= SMP.batch_processing.timeout_config.timeout:
                raise Exception('Time Out')
        dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                 status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                                 dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                 config=self.config_plot,
                                                 count=len(list_status_nodes))
        list_status_nodes.append(copy.copy(dict_status_nodes))

        # Goal test
        if self.reached_goal(node_current.list_paths[-1]):
            solution_path = self.remove_states_behind_goal(node_current.list_paths)
            list_status_nodes = self.plot_solution(path_solution=solution_path, node_status=dict_status_nodes,
                                                   list_states_nodes=list_status_nodes)
            # return solution
            return solution_path, node_current.list_primitives, list_status_nodes

        elif limit == 0:
            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.EXPLORED,
                                                     dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))
            return 'cutoff'

        else:
            cutoff_occurred = False

        for primitive_successor in reversed(node_current.get_successors()):
            # translate/rotate motion primitive to current position
            list_primitives_current = copy.copy(node_current.list_primitives)
            path_translated = self.translate_primitive_to_current_state(primitive_successor,
                                                                        node_current.list_paths[-1])

            # check for collision, if is not collision free it is skipped
            if not self.is_collision_free(path_translated):
                list_status_nodes, dict_status_nodes = self.plot_colliding_primitives(current_node=node_current,
                                                                                      path_translated=path_translated,
                                                                                      node_status=dict_status_nodes,
                                                                                      list_states_nodes=list_status_nodes)
                continue

            # Continue search with child node
            list_primitives_current.append(primitive_successor)
            path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
            child = Node(list_paths=path_new, list_primitives=list_primitives_current,
                         depth_tree=node_current.depth_tree + 1)

            dict_status_nodes = update_visualization(primitive=path_new[-1], status=MotionPrimitiveStatus.IN_FRONTIER,
                                                     dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))

            result = self.recursive_dls(list_status_nodes=list_status_nodes, dict_status_nodes=dict_status_nodes,
                                        node_current=child,
                                        limit=limit - 1)

            if result == 'cutoff':
                cutoff_occurred = True

            elif result is not None:
                return result

        if cutoff_occurred:
            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.EXPLORED,
                                                     dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))

        return 'cutoff' if cutoff_occurred else None
