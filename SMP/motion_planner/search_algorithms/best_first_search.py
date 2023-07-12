import copy
import time
import numpy as np
from abc import abstractmethod, ABC
from typing import Tuple, Dict, Any, List, Union

from commonroad.scenario.state import KSState

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization
from SMP.motion_planner.queue import PriorityQueue
from SMP.motion_planner.search_algorithms.base_class import SearchBaseClass
import SMP.batch_processing.timeout_config

class BestFirstSearch(SearchBaseClass, ABC):
    """
    Abstract class for Best First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)
        self.frontier = PriorityQueue()

    @abstractmethod
    def evaluation_function(self, node_current: PriorityNode):
        """
        Function that evaluates f(n) in the inherited classes.
        @param node_current:
        @return: cost
        """
        pass

    def heuristic_function(self, node_current: PriorityNode) -> float:
        """
        Function that evaluates the heuristic cost h(n) in inherited classes.
        The example provided here estimates the time required to reach the goal state from the current node.
        @param node_current: time to reach the goal
        @return:
        """
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step

        else:
            velocity = node_current.list_paths[-1][-1].velocity

            if np.isclose(velocity, 0):
                return np.inf

            else:
                return self.calc_euclidean_distance(current_node=node_current) / velocity


    def execute_search(self) -> Tuple[Union[None, List[List[KSState]]], Union[None, List[MotionPrimitive]], Any]:
        """
        Implementation of Best First Search (tree search) using a Priority queue.
        The evaluation function of each child class is implemented individually.
        """
        # for visualization in jupyter notebook
        list_status_nodes = []
        dict_status_nodes: Dict[int, Tuple] = {}

        # shift initial state of planning problem from vehicle center to rear axle position
        # (reference point of motion primitives)
        new_state_initial = self.state_initial.translate_rotate(
            -np.array([self.rear_ax_dist * np.cos(self.state_initial.orientation),
                       self.rear_ax_dist * np.sin(self.state_initial.orientation)]), 0)

        # first node
        node_initial = PriorityNode(list_paths=[[new_state_initial]],
                                    list_primitives=[self.motion_primitive_initial], depth_tree=0, priority=0)
        initial_visualization(self.scenario, self.state_initial, self.shape_ego, self.planningProblem, self.config_plot,
                              self.path_fig)

        # add current node (i.e., current path and primitives) to the frontier
        f_initial = self.evaluation_function(node_initial)
        self.frontier.insert(item=node_initial, priority=f_initial)

        dict_status_nodes = update_visualization(primitive=node_initial.list_paths[-1],
                                                 status=MotionPrimitiveStatus.IN_FRONTIER,
                                                 dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                 config=self.config_plot,
                                                 count=len(list_status_nodes))
        list_status_nodes.append(copy.copy(dict_status_nodes))

        start_time = time.perf_counter()
        while not self.frontier.empty():
            if SMP.batch_processing.timeout_config.use_sequential_processing:
                if (time.perf_counter() - start_time) >= SMP.batch_processing.timeout_config.timeout:
                    raise Exception('Time Out')
            # pop the last node
            node_current = self.frontier.pop()

            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                                     dict_node_status=dict_status_nodes,
                                                     path_fig=self.path_fig, config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))

            # goal test
            if self.reached_goal(node_current.list_paths[-1]):
                path_solution = self.remove_states_behind_goal(node_current.list_paths)
                list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=dict_status_nodes,
                                                       list_states_nodes=list_status_nodes)
                # return solution
                return path_solution, node_current.list_primitives, list_status_nodes

            # check all possible successor primitives(i.e., actions) for current node
            for primitive_successor in node_current.get_successors():

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

                list_primitives_current.append(primitive_successor)

                path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
                node_child = PriorityNode(list_paths=path_new,
                                          list_primitives=list_primitives_current,
                                          depth_tree=node_current.depth_tree + 1,
                                          priority=node_current.priority)
                f_child = self.evaluation_function(node_current=node_child)

                # insert the child to the frontier:
                dict_status_nodes = update_visualization(primitive=node_child.list_paths[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(list_status_nodes))
                list_status_nodes.append(copy.copy(dict_status_nodes))
                self.frontier.insert(item=node_child, priority=f_child)

            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.EXPLORED,
                                                     dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))

        return None, None, list_status_nodes


class UniformCostSearch(BestFirstSearch):
    """
    Class for Uniform Cost Search (Dijkstra) algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/ucs/'
        else:
            self.path_fig = None

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of UCS is f(n) = g(n)
        """

        # calculate g(n)
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        return node_current.priority


class GreedyBestFirstSearch(BestFirstSearch):
    """
    Class for Greedy Best First Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/gbfs/'
        else:
            self.path_fig = None

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of GBFS is f(n) = h(n)
        """

        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority


class AStarSearch(BestFirstSearch):
    """
    Class for A* Search algorithm.
    """

    def __init__(self, scenario, planningProblem, automaton, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automaton,
                         plot_config=plot_config)

        if plot_config.SAVE_FIG:
            self.path_fig = '../figures/astar/'
        else:
            self.path_fig = None

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)
