__author__ = "Daniel Tar, Anna-Katharina Rettinger"
__copyright__ = "TUM Cyber-Physical Systems Group"
__version__ = "2020b"
__maintainer__ = "Daniel Tar"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Beta"

import copy
import math
from abc import abstractmethod, ABC
from typing import List, Tuple, Union, Type, Optional, Any

import numpy as np
from commonroad.common.util import Interval
from commonroad.geometry.shape import Rectangle, Polygon, ShapeGroup, Circle
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.obstacle import Obstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.state import InitialState, KSState
from commonroad.scenario.trajectory import Trajectory
from commonroad_dc.boundary import boundary
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_checker, \
    create_collision_object
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from SMP.motion_planner.node import Node, PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig, PlotConfig
from SMP.motion_planner.utility import MotionPrimitiveStatus, update_visualization


class SearchBaseClass(ABC):
    """
    Abstract base class for all motion planners.
    A bunch of helper functions for computing the heuristic cost are provided for your convenience.
    """

    def __init__(self, scenario: Scenario, planningProblem: PlanningProblem, automaton: ManeuverAutomaton,
                 plot_config: Type[PlotConfig] = DefaultPlotConfig):
        """
        Initialization of class BaseMotionPlanner.
        """
        # store input parameters
        self.scenario: Scenario = scenario
        self.planningProblem: PlanningProblem = planningProblem
        self.automaton: ManeuverAutomaton = automaton
        self.shape_ego: Rectangle = automaton.shape_ego

        vehicle_params = VehicleParameterMapping[self.automaton.type_vehicle.name].value
        self.rear_ax_dist = vehicle_params.b

        # create necessary attributes
        self.lanelet_network = self.scenario.lanelet_network
        self.list_obstacles = self.scenario.obstacles
        self.state_initial: InitialState = self.planningProblem.initial_state
        self.motion_primitive_initial = automaton.create_initial_motion_primitive(planningProblem)
        self.list_ids_lanelets_initial = []
        self.list_ids_lanelets_goal = []
        self.time_desired = None
        self.position_desired = None
        self.velocity_desired = None
        self.orientation_desired = None
        self.distance_initial = None
        self.dict_lanelets_costs = {}

        # visualization parameters
        self.config_plot = plot_config
        self.path_fig = None

        # remove unnecessary attributes of the initial state
        """
        even though the attribute will be deleted in the first run,
        hasattr still return True (the absence of the attribute is 
        equivalent its value being None ). 
        In short, after attribute is deleted, it will possess value
        None, but hasattr will still return True.
        More please refer to:
        https://hynek.me/articles/hasattr/
        """
        # if hasattr(self.state_initial, 'yaw_rate'):
        if getattr(self.state_initial,"yaw_rate") != None:
            # delattr(self.state_initial, 'yaw_rate')
            del self.state_initial.yaw_rate

        # if hasattr(self.state_initial, 'slip_angle'):
        if getattr(self.state_initial,"slip_angle") != None:
            del self.state_initial.slip_angle

        # parse planning problem
        self.parse_planning_problem()
        self.initialize_lanelets_costs()

        """
        Create collision checker with the drivability checker package, add obstacles in the scenario to the checker
        For more info on the drivability checker, please refer to https://commonroad.in.tum.de/drivability_checker
        """
        self.collision_checker = create_collision_checker(self.scenario)
        # triangulate road boundary
        _, shapegroup_triangles_boundary = boundary.create_road_boundary_obstacle(scenario,
                                                                                  method='aligned_triangulation',
                                                                                  axis=2)
        # add road boundary into collision checker
        self.collision_checker.add_collision_object(shapegroup_triangles_boundary)

    def parse_planning_problem(self) -> None:
        """
        Parses the given planning problem, and computes related attributes.
        """
        assert isinstance(self.planningProblem, PlanningProblem), "Given planning problem is not valid!"

        # get lanelet id of the initial state
        self.list_ids_lanelets_initial = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.planningProblem.initial_state.position])[0]

        # get lanelet id of the goal region, which can be of different types
        self.list_ids_lanelets_goal = None
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            if hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                self.list_ids_lanelets_goal = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.center])[0]

            elif hasattr(self.planningProblem.goal.state_list[0].position, 'shapes'):
                self.list_ids_lanelets_goal = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.shapes[0].center])[0]
                self.planningProblem.goal.state_list[0].position.center = \
                    self.planningProblem.goal.state_list[0].position.shapes[0].center

        # set attributes with given planning problem
        if hasattr(self.planningProblem.goal.state_list[0], 'time_step'):
            self.time_desired = self.planningProblem.goal.state_list[0].time_step
        else:
            self.time_desired = Interval(0, np.inf)

        self.position_desired = None
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            if hasattr(self.planningProblem.goal.state_list[0].position, 'vertices'):
                self.position_desired = self.calc_goal_interval(self.planningProblem.goal.state_list[0].position.vertices)

            elif hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                x = self.planningProblem.goal.state_list[0].position.center[0]
                y = self.planningProblem.goal.state_list[0].position.center[1]
                self.position_desired = [Interval(start=x, end=x), Interval(start=y, end=y)]

        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            self.velocity_desired = self.planningProblem.goal.state_list[0].velocity
        else:
            self.velocity_desired = Interval(0, np.inf)

        if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
            self.orientation_desired = self.planningProblem.goal.state_list[0].orientation
        else:
            self.orientation_desired = Interval(-math.pi, math.pi)

        # create necessary attributes
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            self.distance_initial = SearchBaseClass.distance(self.planningProblem.initial_state.position,
                                                             self.planningProblem.goal.state_list[0].position.center)
        else:
            self.distance_initial = 0

    def initialize_lanelets_costs(self) -> None:
        """
        Initializes the heuristic costs for lanelets. The cost of a lanelet equals to the number
        of lanelets that should be traversed before reaching the goal region. The cost is set to
        -1 if it is not possible to reach the goal region from the lanelet, and 0 if it is within
        the list of goal lanelets.
        """
        # set lanelet costs to -1, except goal lanelet (0)
        for lanelet in self.scenario.lanelet_network.lanelets:
            self.dict_lanelets_costs[lanelet.lanelet_id] = -1

        if self.list_ids_lanelets_goal is not None:
            for ids_lanelet_goal in self.list_ids_lanelets_goal:
                self.dict_lanelets_costs[ids_lanelet_goal] = 0

            # calculate costs for lanelets, this is a recursive method
            for ids_lanelet_goal in self.list_ids_lanelets_goal:
                list_lanelets_visited = []
                lanelet_goal = self.scenario.lanelet_network.find_lanelet_by_id(ids_lanelet_goal)
                self.calc_lanelet_cost(lanelet_goal, 1, list_lanelets_visited)

    @abstractmethod
    def execute_search(self) -> Tuple[Union[None, List[List[KSState]]], Union[None, List[MotionPrimitive]], Any]:
        """
        The actual search algorithms are implemented in the children classes.
        """
        pass

    def plot_solution(self, path_solution, node_status, list_states_nodes):
        node_status = update_visualization(primitive=path_solution[-1], status=MotionPrimitiveStatus.SOLUTION,
                                           dict_node_status=node_status, path_fig=self.path_fig,
                                           config=self.config_plot,
                                           count=len(list_states_nodes))
        list_states_nodes.append(copy.copy(node_status))
        for prim in path_solution:
            node_status = update_visualization(primitive=prim, status=MotionPrimitiveStatus.SOLUTION,
                                               dict_node_status=node_status, path_fig=self.path_fig,
                                               config=self.config_plot,
                                               count=len(list_states_nodes))
        list_states_nodes.append(copy.copy(node_status))
        return list_states_nodes

    def plot_colliding_primitives(self, current_node: Union[Node, PriorityNode], path_translated, node_status,
                                  list_states_nodes):
        node_status = update_visualization(primitive=[current_node.list_paths[-1][-1]] + path_translated,
                                           status=MotionPrimitiveStatus.INVALID, dict_node_status=node_status,
                                           path_fig=self.path_fig, config=self.config_plot,
                                           count=len(list_states_nodes))
        if self.config_plot.PLOT_COLLISION_STEPS and self.config_plot.DO_PLOT:
            list_states_nodes.append(copy.copy(node_status))

        return list_states_nodes, node_status

    """
    the following helper functions are for your convenience. You may use them if you want.
    """

    @staticmethod
    def find_closest_vertex(centerVertices: np.ndarray, pos: np.ndarray) -> int:
        """
        Returns the index of the closest center vertex to the given position.

        :param centerVertices: the vertices of the center line of the Lanelet described as a polyline
        [[x0,x1,...,xn],[y0,y1,...,yn]]
        :param pos: the closest vertex to this position will be found
        """
        distances = []
        for vertex in centerVertices:
            distances.append(SearchBaseClass.distance(vertex, pos, 0))
        return distances.index(min(distances))

    @staticmethod
    def calc_angle_of_position(centerVertices: np.ndarray, pos: np.ndarray) -> float:
        """
        Returns the angle (in world coordinate, radian) of the line defined by 2 nearest lanelet center vertices to
        the given position.

        :param centerVertices: lanelet center vertices, whose distance to the given position is considered
        :param pos: the input position
        """
        index_closestVert = SearchBaseClass.find_closest_vertex(centerVertices, pos)
        if index_closestVert + 1 >= centerVertices.size / 2.0:
            index_closestVert = index_closestVert - 1
        return math.atan2(centerVertices[index_closestVert + 1][1] - centerVertices[index_closestVert][1],
                          centerVertices[index_closestVert + 1][0] - centerVertices[index_closestVert][0])

    @staticmethod
    def calc_dist_to_closets_point_on_line(vertexA: np.ndarray, vertexB: np.ndarray, pos: np.ndarray) -> float:
        """
        Returns the distance of the given position to a line segment (e.g. the nearest lanelet center line segment to
        the given position).

        :param pos:
        :param vertexA: the start point of the line segment
        :param vertexB: the end point of the line segment
        """

        magAB2 = (vertexB[0] - vertexA[0]) ** 2 + (vertexB[1] - vertexA[1]) ** 2
        ABdotAP = (pos[0] - vertexA[0]) * (vertexB[0] - vertexA[0]) + (pos[1] - vertexA[1]) * (vertexB[1] - vertexA[1])
        s = ABdotAP / magAB2
        if s < 0:
            return SearchBaseClass.distance(vertexA, pos, 0)
        elif s > 1:
            return SearchBaseClass.distance(vertexB, pos, 0)
        else:
            newVertex = np.empty(2)
            newVertex[0] = vertexA[0] + (vertexB[0] - vertexA[0]) * s
            newVertex[1] = vertexA[1] + (vertexB[1] - vertexA[1]) * s
            return SearchBaseClass.distance(newVertex, pos, 0)

    @staticmethod
    def calc_distance_to_nearest_point(centerVertices: np.ndarray, pos: np.ndarray) -> float:
        """
        Returns the closest euclidean distance to a polyline (e.g. defined by lanelet center vertices) according to
        the given position.

        :param centerVertices: the polyline, between which and the given position the distance is calculated
        :param pos: the input position
        """
        distances = []
        for vertex in centerVertices:
            distances.append(SearchBaseClass.distance(vertex, pos, 0))
        index_closestVert = distances.index(min(distances))

        if (index_closestVert + 1) < len(centerVertices):
            dist1 = SearchBaseClass.calc_dist_to_closets_point_on_line(centerVertices[index_closestVert],
                                                                       centerVertices[index_closestVert + 1],
                                                                       pos)
        else:
            dist1 = SearchBaseClass.distance(centerVertices[index_closestVert], pos, 0)

        if index_closestVert > 0:
            dist2 = SearchBaseClass.calc_dist_to_closets_point_on_line(centerVertices[index_closestVert - 1],
                                                                       centerVertices[index_closestVert],
                                                                       pos)
        else:
            dist2 = SearchBaseClass.distance(centerVertices[index_closestVert], pos, 0)

        return min(dist1, dist2)

    @staticmethod
    def calc_travelled_distance(path: List[KSState]) -> float:
        """
        Returns the travelled distance of the given path.

        :param: the path, whose euclidean distance is calculated

        """
        dist = 0
        for i in range(len(path) - 1):
            dist += SearchBaseClass.distance(path[i].position, path[i + 1].position)
        return dist

    @staticmethod
    def euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the euclidean distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        # TODO: check if np.sqrt((pos1-pos2).T @ (pos1-pos2)) is faster
        return np.sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]))

    @staticmethod
    def manhattan_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the manhattan distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def chebyshev_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the chebyshev distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

    @staticmethod
    def canberra_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the canberra distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return abs(pos1[0] - pos2[0]) / (abs(pos1[0]) + abs(pos2[0])) + abs(pos1[1] - pos2[1]) / (
                abs(pos1[1]) + abs(pos2[1]))

    @staticmethod
    def cosine_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the cosine distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 1 - (pos1[0] * pos2[0] + pos1[1] * pos2[1]) / (
                np.sqrt(pos1[0] ** 2 + pos2[0] ** 2) * np.sqrt(pos1[1] ** 2 + pos2[1] ** 2))

    @staticmethod
    def sum_of_squared_difference(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the squared euclidean distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    @staticmethod
    def mean_absolute_error(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns a half of the manhattan distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 0.5 * (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    @staticmethod
    def mean_squared_error(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the mean of squared difference between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 0.5 * ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    @classmethod
    def distance(cls, pos1: np.ndarray, pos2: np.ndarray = np.zeros(2), distance_type=0) -> float:
        """
        Returns the distance between 2 points, the type is specified by 'type'.

        :param pos1: the first point
        :param pos2: the second point
        :param distance_type: specifies which kind of distance is used:
            1: manhattanDistance,
            2: chebyshevDistance,
            3: sumOfSquaredDifference,
            4: meanAbsoluteError,
            5: meanSquaredError,
            6: canberraDistance,
            7: cosineDistance.
        """
        if distance_type == 0:
            return cls.euclidean_distance(pos1, pos2)
        elif distance_type == 1:
            return cls.manhattan_distance(pos1, pos2)
        elif distance_type == 2:
            return cls.chebyshev_distance(pos1, pos2)
        elif distance_type == 3:
            return cls.sum_of_squared_difference(pos1, pos2)
        elif distance_type == 4:
            return cls.mean_absolute_error(pos1, pos2)
        elif distance_type == 5:
            return cls.mean_squared_error(pos1, pos2)
        elif distance_type == 6:
            return cls.canberra_distance(pos1, pos2)
        elif distance_type == 7:
            return cls.cosine_distance(pos1, pos2)
        return math.inf

    @staticmethod
    def calc_curvature_of_polyline(polyline: np.ndarray) -> float:
        """
        Returns the curvature of the given polyline.

        :param polyline: the polyline to be calculated
        """
        dx_dt = np.gradient(polyline[:, 0])
        dy_dt = np.gradient(polyline[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvatureArray = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
        curvature = 0
        for elem in curvatureArray:
            curvature = curvature + abs(elem)
        return curvature

    @staticmethod
    def calc_orientation_diff(orientation_1: float, orientation_2: float) -> float:
        """
        Returns the orientation difference between 2 orientations in radians.

        :param orientation_1: the first orientation.
        :param orientation_2: the second orientation.
        """
        return math.pi - abs(abs(orientation_1 - orientation_2) - math.pi)

    @staticmethod
    def calc_length_of_polyline(polyline: np.ndarray) -> float:
        """
        Calculates the length of the polyline

        :param polyline: the polyline, whose length is calculated
        :returns the length of the polyline
        """

        dist = 0
        for i in range(0, len(polyline) - 1):
            dist += SearchBaseClass.distance(polyline[i], polyline[i + 1])
        return dist

    @staticmethod
    def find_closest_point_on_line(vertexA: np.ndarray, vertexB: np.ndarray, pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Finds the closest point of the given position on the line segment
            (e.g. the nearest lanelet center line segment to the given position)

        :param pos:
        :param vertexA: the start point of the line segment
        :param vertexB: the end point of the line segment
        :returns the closest point of the given position on the line segment
                 (e.g. the nearest lanelet center line segment to the given position)
        """

        if vertexA is None or vertexB is None or pos is None:
            return None
        magAB2 = (vertexB[0] - vertexA[0]) ** 2 + (vertexB[1] - vertexA[1]) ** 2
        ABdotAP = (pos[0] - vertexA[0]) * (vertexB[0] - vertexA[0]) + (pos[1] - vertexA[1]) * (vertexB[1] - vertexA[1])
        s = ABdotAP / magAB2
        if s < 0:
            return vertexA
        elif s > 1:
            return vertexB
        else:
            newVertex = np.empty(2)
            newVertex[0] = vertexA[0] + (vertexB[0] - vertexA[0]) * s
            newVertex[1] = vertexA[1] + (vertexB[1] - vertexA[1]) * s
            return newVertex

    @staticmethod
    def find_nearest_point_on_lanelet(lanelet: Lanelet, pos: np.ndarray) -> (np.ndarray, int, int):
        """
        Finds the projection of a position on a lanelet and the trailing and following index of the center vertices

        :param lanelet: the lanelet on that the projection should be applied
        :param pos: the position to be projected
        :returns the projection of a position on a lanelet and the trailing and following index of the center vertices
        """

        distances = []
        centerVertices = lanelet.center_vertices
        for vertex in centerVertices:
            distances.append(SearchBaseClass.distance(vertex, pos, 0))
        index_closestVert = distances.index(min(distances))
        if (index_closestVert + 1) < len(centerVertices):
            currVerOnLane1 = SearchBaseClass.find_closest_point_on_line(centerVertices[index_closestVert],
                                                                        centerVertices[index_closestVert + 1],
                                                                        pos)
            index_nextCenterVert1 = index_closestVert + 1
        else:
            currVerOnLane1 = centerVertices[index_closestVert]
            index_nextCenterVert1 = None

        index_prevCenterVert1 = index_closestVert
        dist1 = SearchBaseClass.distance(currVerOnLane1, pos, 0)

        if index_closestVert > 0:
            currVerOnLane2 = SearchBaseClass.find_closest_point_on_line(centerVertices[index_closestVert - 1],
                                                                        centerVertices[index_closestVert],
                                                                        pos)
            index_prevCenterVert2 = index_closestVert - 1

        else:
            currVerOnLane2 = centerVertices[index_closestVert]
            index_prevCenterVert2 = None

        index_nextCenterVert2 = index_closestVert
        dist2 = SearchBaseClass.distance(currVerOnLane2, pos, 0)

        if dist1 < dist2:
            currVerOnLane = currVerOnLane1
            # dist = dist1
            index_nextCenterVert = index_nextCenterVert1
            index_prevCenterVert = index_prevCenterVert1
        else:
            currVerOnLane = currVerOnLane2
            # dist = dist2
            index_nextCenterVert = index_nextCenterVert2
            index_prevCenterVert = index_prevCenterVert2

        return currVerOnLane, index_prevCenterVert, index_nextCenterVert

    @staticmethod
    def find_adjacent_lanelets_same_direction(curr_lanelet: Lanelet) -> List[int]:
        """
        Finds the list_lanelets_adjacent of a lanelet in the same direction

        :param curr_lanelet: the lanelet
        :returns the ids of the adjacent lanelets in the same direction
        """
        list_lanelets_adjacent = []
        if curr_lanelet.adj_left_same_direction:
            list_lanelets_adjacent += [curr_lanelet.adj_left]
        if curr_lanelet.adj_right_same_direction:
            list_lanelets_adjacent += [curr_lanelet.adj_right]
        return list_lanelets_adjacent

    @staticmethod
    def get_lanelet_width(lanelet: Lanelet, index: int) -> Union[float, None]:
        """
        Calculates the width of the lanelet in the given index

        :param lanelet: the lanelet which width will be returned
        :param index: index of the vertex,  where the width need to be calculated
        :returns the width of the given lanelet in the given index
        """

        if lanelet is None or index is None or len(lanelet.right_vertices) <= index:
            return None
        return SearchBaseClass.distance(lanelet.left_vertices[index], lanelet.right_vertices[index])

    @staticmethod
    def calc_normal_distance_from_line_with_direction(direction_vector: np.ndarray, position: np.ndarray) -> float:
        """
        Calculates the signed distance between a direction and a position
        Note: This implementation can be used only the position vector if perpendicular to the direction

        :param direction_vector: the direction to that the normal distance should be calculated
        :param position: the position which distance should be calculated
        :returns the signed normal distance
        """

        abs_distance = SearchBaseClass.distance(position)
        return np.sign(np.cross(direction_vector, position)) * abs_distance

    def map_obstacles_to_lanelets(self, time_step: int) -> dict:
        """
        Find all obstacles that are located in every lanelet at time step t and returns a dictionary where obstacles
        are stored according to lanelet id.

        :param time_step: The time step in which the obstacle is in the current lanelet network.
        :Return type: dict[lanelet_id]
        """
        mapping = {}
        for lanelet in self.lanelet_network.lanelets:
            # map obstacles to current lanelet
            mapped_objs = self.get_obstacles(lanelet, time_step)
            # check if mapping is not empty
            if len(mapped_objs) > 0:
                mapping[lanelet.lanelet_id] = mapped_objs
        return mapping

    def get_obstacles(self, laneletObj: Lanelet, time_step: int) -> List[Obstacle]:
        """
        Returns the subset of obstacles, which are located in the given lanelet.

        :param laneletObj: specify the lanelet object to get its obstacles
        :param time_step: the time step in which the occupancy of obstacles is checked

        """
        # output list
        res = list()

        # look at each obstacle
        for o in self.list_obstacles:
            if o.occupancy_at_time(time_step) is not None:
                o_shape = o.occupancy_at_time(time_step).shape

                # vertices to check
                vertices = list()

                # distinguish between shape and shapegroup and extract vertices
                if isinstance(o_shape, ShapeGroup):
                    for sh in o_shape.shapes:
                        # distinguish between type of shape (circle has no vertices)
                        if isinstance(sh, Circle):
                            vertices.append(sh.center)
                        elif isinstance(sh, Rectangle) or isinstance(sh, Polygon):
                            vertices.append(sh.vertices)
                else:
                    # distinguish between type of shape (circle has no vertices)
                    if isinstance(o_shape, Circle):
                        vertices = o_shape.center
                    elif isinstance(o_shape, Rectangle) or isinstance(o_shape, Polygon):
                        vertices = o_shape.vertices

                # check if obstacle is in lane
                if any(laneletObj.contains_points(np.array(vertices))):
                    res.append(o)
        return res

    def calc_lanelet_cost(self, lanelet_current: Lanelet, dist: int, list_lanelets_visited: List[int]) -> None:
        """
        Calculates distances of all lanelets which can be reached through recursive adjacency/predecessor relationship
        by the current lanelet. This is a recursive implementation.

        :param lanelet_current: the current lanelet object (Often set to the goal lanelet).
        :param dist: the initial distance between 2 adjacent lanelets (Often set to 1). This value will increase
        recursively during the execution of this function.
        :param list_lanelets_visited: list of visited lanelet id. In the iterations, visited lanelets will not be
        considered. This value changes during the recursive implementation.
        """
        if lanelet_current.lanelet_id in list_lanelets_visited:
            return
        else:
            list_lanelets_visited.append(lanelet_current.lanelet_id)

        if lanelet_current.predecessor is not None:
            for pred in lanelet_current.predecessor:
                if self.dict_lanelets_costs[pred] == -1 or self.dict_lanelets_costs[pred] > dist:
                    self.dict_lanelets_costs[pred] = dist

            for pred in lanelet_current.predecessor:
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(pred), dist + 1, list_lanelets_visited)

        if lanelet_current.adj_left is not None and lanelet_current.adj_left_same_direction:
            if self.dict_lanelets_costs[lanelet_current.adj_left] == -1 or \
                    self.dict_lanelets_costs[lanelet_current.adj_left] > dist:
                self.dict_lanelets_costs[lanelet_current.adj_left] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(lanelet_current.adj_left), dist + 1,
                                       list_lanelets_visited)

        if lanelet_current.adj_right is not None and lanelet_current.adj_right_same_direction:
            if self.dict_lanelets_costs[lanelet_current.adj_right] == -1 or \
                    self.dict_lanelets_costs[lanelet_current.adj_right] > dist:
                self.dict_lanelets_costs[lanelet_current.adj_right] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(lanelet_current.adj_right), dist + 1,
                                       list_lanelets_visited)

    def calc_lanelet_orientation(self, lanelet_id: int, pos: np.ndarray) -> float:
        """
        Returns lanelet orientation (angle in radian, counter-clockwise defined) at the given position and lanelet id.

        :param lanelet_id: id of the lanelet, based on which the orientation is calculated
        :param pos: position, where orientation is calculated. (Often the position of the obstacle)

        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        return SearchBaseClass.calc_angle_of_position(laneletObj.center_vertices, pos)

    def calc_angle_to_goal(self, state: KSState) -> float:
        """
        Returns the orientation of the goal (angle in radian, counter-clockwise defined) with respect to the position
        of the state.

        :param state: the angle between this state and the goal will be calculated

        """

        curPos = state.position
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            goalPos = self.planningProblem.goal.state_list[0].position.center
            return math.atan2(goalPos[1] - curPos[1], goalPos[0] - curPos[0])
        else:
            return 0

    def lanelets_of_position(self, lanelets: List[int], state: KSState, diff: float = math.pi / 5) -> List[int]:
        """
        Returns all lanelets, whose angle to the orientation of the input state are smaller than pi/5.

        :param lanelets: potential lanelets
        :param state: the input state
        :param diff: acceptable angle difference between the state and the lanelet

        """

        correctLanelets = []
        for laneletId in lanelets:
            laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(laneletId)
            laneletOrientationAtPosition = SearchBaseClass.calc_angle_of_position(laneletObj.center_vertices,
                                                                                  state.position)
            if math.pi - abs(abs(laneletOrientationAtPosition - state.orientation) - math.pi) < diff:
                correctLanelets.append(laneletId)

        while len(correctLanelets) > 0:
            if self.dict_lanelets_costs[correctLanelets[0]] == -1:
                correctLanelets.pop(0)
            else:
                break

        return correctLanelets

    def calc_dist_to_closest_obstacle(self, lanelet_id: int, pos: np.ndarray, time_step: int) -> float:
        """
        Returns distance between the given position and the center of the closest obstacle in the given lanelet
        (specified by lanelet id).

        :param lanelet_id: the id of the lanelet where the distance to obstacle is calculated
        :param pos: current input position
        :param time_step: current time step

        """

        obstacles_in_lanelet = self.get_obstacles(self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id),
                                                  time_step)
        shortestDist = math.inf
        for obstacleObj in obstacles_in_lanelet:
            shape_obs = obstacleObj.occupancy_at_time(time_step).shape
            if isinstance(shape_obs, Circle):
                if SearchBaseClass.distance(pos, shape_obs.center) < shortestDist:
                    shortestDist = SearchBaseClass.distance(pos, shape_obs.center)
        return shortestDist

    def num_obstacles_in_lanelet_at_time_step(self, time_step: int, lanelet_id: int) -> int:
        """
        Returns the number of obstacles in the given lanelet (specified by lanelet id) at time step t.

        :param time_step: time step
        :param lanelet_id: id of the lanelet whose obstacles are considered

        """
        obstacles_in_lanelet = self.get_obstacles(self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id),
                                                  time_step)
        return len(obstacles_in_lanelet)

    def is_adjacent(self, start_lanelet_id: int, final_lanelet_id: int) -> bool:
        """
        Returns true if the the final lanelet is adjacent to the start lanelet.

        :param start_lanelet_id: id of the first lanelet (start lanelet).
        :param final_lanelet_id: id of the second lanelet (final lanelet).
        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(start_lanelet_id)
        if laneletObj.adj_left is not None and laneletObj.adj_left_same_direction:
            if laneletObj.adj_left == final_lanelet_id:
                return True

        if laneletObj.adj_right is not None and laneletObj.adj_right_same_direction:
            if laneletObj.adj_right == final_lanelet_id:
                return True
        return False

    def is_successor(self, start_lanelet_id: int, final_lanelet_id: int) -> bool:
        """
        Returns true if the the final lanelet is a successor of the start lanelet.

        :param start_lanelet_id: id of the first lanelet (start lanelet).
        :param final_lanelet_id: id of the second lanelet (final lanelet).
        Return type: bool
        """

        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(start_lanelet_id)
        if laneletObj.successor is not None:
            for suc in laneletObj.successor:
                if suc == final_lanelet_id:
                    return True
        return False

    def is_goal_in_lane(self, lanelet_id: int, traversed_lanelets=None) -> bool:
        """
        Returns true if the goal is in the given lanelet or any successor (including all successors of successors) of
        the given lanelet.

        :param lanelet_id: the id of the given lanelet.
        :param traversed_lanelets: helper variable which stores potential path (a list of lanelet id) to goal lanelet.
        Initialized to None.

        """
        if traversed_lanelets is None:
            traversed_lanelets = []
        if lanelet_id not in traversed_lanelets:
            traversed_lanelets.append(lanelet_id)
        else:
            return False
        reachable = False
        if lanelet_id in self.list_ids_lanelets_goal:
            return True
        laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
        if laneletObj.successor is not None:
            for suc in laneletObj.successor:
                if self.dict_lanelets_costs[suc] >= 0:
                    reachable = self.is_goal_in_lane(suc, traversed_lanelets)
                    if reachable:
                        return True
        return reachable

    @staticmethod
    def calc_time_cost(path: List[KSState]) -> int:
        """
        Returns time cost (number of time steps) to perform the given path.

        :param path: the path whose time cost is calculated
        """
        return path[-1].time_step - path[0].time_step

    def calc_path_efficiency(self, path: List[KSState]) -> float:
        """
        Returns the path efficiency = travelled_distance / time_cost

        :param path: the path whose efficiency is calculated
        """
        cost_time = self.calc_time_cost(path)
        if np.isclose(cost_time, 0):
            return np.inf
        else:
            return SearchBaseClass.calc_travelled_distance(path) / cost_time

    def calc_heuristic_distance(self, state: KSState, distance_type=0) -> float:
        """
        Returns the heuristic distance between the current state and the goal state.

        :param state: the state, whose heuristic distance to the goal is calculated
        :param distance_type: default is euclidean distance. For other distance type please refer to the function
        "distance(pos1: np.ndarray, pos2: np.ndarray, type=0)"
        """

        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            curPos = state.position
            goalPos = self.planningProblem.goal.state_list[0].position.center
            return SearchBaseClass.distance(curPos, goalPos, distance_type)
        else:
            return 0

    def calc_heuristic_lanelet(self, path: List[KSState]) -> Union[Tuple[None, None, None], Tuple[float, list, list]]:
        """
        Calculates the distance between every individual state of the path and the centers of the path's corresponding
        lanelets and sum them up.

        :param path: the path, whose heuristics is calculated

        Returns the heuristic distance of the path (float), id of the end lanelet of the given path (list) and the start
         lanelet id (list).
        End lanelet means the lanelet where the last state of the path is in, start lanelet means the lanelet
        corresponding to the first state of the path.

        """
        end_lanelet_id = None
        dist = 0
        start_lanelet_id = self.scenario.lanelet_network.find_lanelet_by_position([path[0].position])[
            0]  # returns id of the start lanelet
        if not start_lanelet_id:
            return None, None, None
        for i in range(len(path)):
            lanelets_of_pathSegment = self.lanelets_of_position(
                self.scenario.lanelet_network.find_lanelet_by_position([path[i].position])[0], path[i])
            if not lanelets_of_pathSegment:
                return None, None, None  # return none if path element is not in a lanelet with correct orientation
            laneletObj = self.scenario.lanelet_network.find_lanelet_by_id(lanelets_of_pathSegment[0])
            dist = dist + SearchBaseClass.calc_distance_to_nearest_point(laneletObj.center_vertices,
                                                                         path[i].position)  # distance to center line
            end_lanelet_id = lanelets_of_pathSegment
        return dist, end_lanelet_id, start_lanelet_id

    @staticmethod
    def calc_goal_interval(vertices) -> List[Interval]:
        """
        Calculate the maximum Intervals of the goal position given as vertices.
        @param: vertices: vertices which describe the goal position.
        """
        min_x = np.inf
        max_x = -np.inf

        min_y = np.inf
        max_y = -np.inf
        for vertex in vertices:
            if vertex[0] < min_x:
                min_x = vertex[0]
            if vertex[0] > max_x:
                max_x = vertex[0]
            if vertex[1] < min_y:
                min_y = vertex[1]
            if vertex[1] > max_y:
                max_y = vertex[1]
        return [Interval(start=min_x, end=max_x), Interval(start=min_y, end=max_y)]

    def calc_euclidean_distance(self, current_node: PriorityNode) -> float:
        """
        Calculates the euclidean distance of the vehicle center to the desired goal position. The attribute
        self.position_desired is extracted from the planning problem (see method self.parse_planning_problem() )

        @param current_node:
        @return: euclidean distance
        """
        current_node_state = current_node.list_paths[-1][-1]    # get last state in current path
        pos = current_node_state.position                       # get (rear axis) position of last state
        # get positions of vehicle center (node state refers to reference point of Motion Primitives, i.e., rear axis)
        pos_veh_center = pos + np.array([self.rear_ax_dist * np.cos(current_node_state.orientation),
                                         self.rear_ax_dist * np.sin(current_node_state.orientation)])

        if self.position_desired[0].contains(pos_veh_center[0]):
            delta_x = 0.0
        else:
            delta_x = min([abs(self.position_desired[0].start - pos_veh_center[0]),
                           abs(self.position_desired[0].end - pos_veh_center[0])])
        if self.position_desired[1].contains(pos_veh_center[1]):
            delta_y = 0
        else:
            delta_y = min([abs(self.position_desired[1].start - pos_veh_center[1]),
                           abs(self.position_desired[1].end - pos_veh_center[1])])

        return np.sqrt(delta_x ** 2 + delta_y ** 2)

    def reached_goal(self, path: List[KSState]) -> bool:
        """
        Goal-test every state of the path and returns true if one of the state satisfies all conditions for the goal
        region: position, orientation, velocity, time.

        :param path: the path to be goal-tested

        """
        for i in range(len(path)):
            # check if center of vehicle is within goal region
            state = path[i]
            kwarg = {'position': state.position + np.array([self.rear_ax_dist * np.cos(state.orientation),
                                                            self.rear_ax_dist * np.sin(state.orientation)]),
                     'velocity': state.velocity,
                     'steering_angle': state.steering_angle,
                     'orientation': state.orientation,
                     'time_step': state.time_step}
            state_shifted = KSState(**kwarg)
            if self.planningProblem.goal.is_reached(state_shifted):
                return True
        return False

    def remove_states_behind_goal(self, path: List[List[KSState]]) -> List[List[KSState]]:
        """
        Removes all states that are behind the state which satisfies goal state conditions and returns the pruned path.

        :param path: the path to be pruned
        """

        for i in range(len(path[-1])):
            # check if center of vehicle is within goal region
            state = path[-1][i]
            kwarg = {'position': state.position + np.array([self.rear_ax_dist * np.cos(state.orientation),
                                                            self.rear_ax_dist * np.sin(state.orientation)]),
                     'velocity': state.velocity,
                     'steering_angle': state.steering_angle,
                     'orientation': state.orientation,
                     'time_step': state.time_step}
            state_shifted = KSState(**kwarg)
            if self.planningProblem.goal.is_reached(state_shifted):
                for j in range(i + 1, len(path[-1])):
                    path[-1].pop()
                return path
        return path

    def is_collision_free(self, path: List[KSState]) -> bool:
        """
        Checks if path collides with an obstacle. Returns true for no collision and false otherwise.

        :param path: The path you want to check
        """
        # positions of states in input path have to be shifted to the center of the vehicle, since the positions
        # refer to the rear axis (reference point) when using motion primitives for the KS model
        # the collision checker requires the center point position to create the collision object

        new_state_list = list()

        for state in path:
            new_state = state.translate_rotate(np.array([self.rear_ax_dist * math.cos(state.orientation),
                                                         self.rear_ax_dist * math.sin(state.orientation)]), 0)
            new_state_list.append(new_state)

        trajectory = Trajectory(path[0].time_step, new_state_list)

        # create a TrajectoryPrediction object consisting of the trajectory and the shape of the ego vehicle
        traj_pred = TrajectoryPrediction(trajectory=trajectory, shape=self.shape_ego)

        # create a collision object using the trajectory prediction of the ego vehicle
        collision_object = create_collision_object(traj_pred)

        # check collision for motion primitive. a primitive is collision-free if none of its states at different time
        # steps is colliding
        if self.collision_checker.collide(collision_object):
            return False
        return True

    @staticmethod
    def translate_primitive_to_current_state(primitive: MotionPrimitive, path_current: List[KSState]) -> List[KSState]:
        """
        Uses the trajectory defined in the given primitive, translates it towards the last state of current path and
        returns the list of new path.
        In the newly appended part (created through translation of the primitive) of the path, the position,
        orientation and time step are changed, but the velocity is not changed.
        Attention: The input primitive itself will not be changed after this operation.

        :param primitive: the primitive to be translated
        :param path_current: the path whose last state is the goal state for the translation
        """
        return primitive.attach_trajectory_to_state(path_current[-1])

    @staticmethod
    def append_path(path_current: List[KSState], newPath: List[KSState]) -> List[KSState]:
        """
        Appends a new path to the current path and returns the whole path.

        :param path_current: current path which is to be extended
        :param newPath: new path which is going to be added to the current path
        """
        path = path_current[:]
        path.extend(newPath)
        return path
