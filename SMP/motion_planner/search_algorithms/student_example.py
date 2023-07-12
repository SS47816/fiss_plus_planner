import numpy as np
from SMP.motion_planner.node import PriorityNode

from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


class StudentMotionPlannerExample(GreedyBestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        # copied the implementation in GreedyBestFirstSearch
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority

    def heuristic_function(self, node_current: PriorityNode) -> float:
        # a sample heuristic function from a previous random student
        path_last = node_current.list_paths[-1]

        distStartState = self.calc_heuristic_distance(path_last[0])
        distLastState = self.calc_heuristic_distance(path_last[-1])

        if distLastState is None:
            return np.inf

        if distStartState < distLastState:
            return np.inf

        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(path_last)

        if cost_lanelet is None or final_lanelet_id[0] is None:
            return np.inf

        self.calc_path_efficiency(path_last)
        self.num_obstacles_in_lanelet_at_time_step(path_last[-1].time_step, final_lanelet_id[0])
        self.is_goal_in_lane(final_lanelet_id[0])
        if self.dict_lanelets_costs[final_lanelet_id[0]] == -1:
            return np.inf
        factor = 1
        if self.dict_lanelets_costs[final_lanelet_id[0]] > self.dict_lanelets_costs[start_lanelet_id[0]]:
            return np.inf
        if self.dict_lanelets_costs[final_lanelet_id[0]] < self.dict_lanelets_costs[start_lanelet_id[0]]:
            factor = factor * 0.1

        angleToGoal = self.calc_angle_to_goal(path_last[-1])

        orientationToGoalDiff = self.calc_orientation_diff(angleToGoal, path_last[-1].orientation)
        if final_lanelet_id[0] in self.list_ids_lanelets_goal:
            factor = factor * 0.07
        pathLength = self.calc_travelled_distance(path_last)
        cost_time = self.calc_time_cost(path_last)
        weights = np.zeros(6)
        if distLastState < 0.5:
            factor = factor * 0.00001
        # elif math.pi - abs(abs(laneletOrientationAtPosition - path[-1].orientation)

        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start +
                           self.planningProblem.goal.state_list[0].velocity.end) / 2
            dist_vel = abs(path_last[-1].velocity - v_mean_goal)
        else:
            dist_vel = 0

        weights[0] = 8.7
        weights[1] = 0.01
        weights[2] = 0.5
        weights[3] = 0.1
        weights[4] = 0.05
        weights[5] = 1
        cost = weights[0] * (cost_lanelet / len(path_last)) + \
               weights[1] * abs(orientationToGoalDiff) + \
               weights[3] * cost_time + \
               weights[2] * distLastState + \
               weights[4] * (100 - pathLength) + \
               weights[5] * dist_vel

        if cost < 0:
            cost = 0
        return cost * factor
