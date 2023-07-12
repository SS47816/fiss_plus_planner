import fnmatch
import os
import pickle
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import List

from commonroad.common.solution import VehicleType
from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblem
from tqdm.notebook import tqdm

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive, MotionPrimitiveParser


class ManeuverAutomaton(object):
    """
    Class for motion automaton, which holds and manipulates motion primitives for motion planning.
    """
    __create_key = object()
    extension = "automaton"

    def __init__(self, create_key, type_vehicle: VehicleType):
        """
        Initialization of a motion automaton.
        """
        assert (create_key == ManeuverAutomaton.__create_key), \
            "MotionAutomaton objects must be created using MotionAutomaton.generateAutomaton or" \
            "MotionAutomaton.loadAutomaton"

        assert type_vehicle in (VehicleType.FORD_ESCORT, VehicleType.BMW_320i, VehicleType.VW_VANAGON), \
            f"Input vehicle type <{type_vehicle}> is not valid! Must be either FORD_ESCORT, BMW_320i or VW_VANAGON."

        self._primitive_id_counter = int(-1)
        self.num_primitives = 0
        self.list_primitives = []

        self.sets_tuples_primitives = set()
        self.dict_primitives = dict()
        self.type_vehicle = type_vehicle


        self.shape_ego = None
        if self.type_vehicle == VehicleType.FORD_ESCORT:
            self.shape_ego = Rectangle(length=4.298, width=1.674)
        elif self.type_vehicle == VehicleType.BMW_320i:
            self.shape_ego = Rectangle(length=4.508, width=1.610)
        elif self.type_vehicle == VehicleType.VW_VANAGON:
            self.shape_ego = Rectangle(length=4.569, width=1.844)

    @classmethod
    def assert_file_extension(cls, name_file: str, extension: str) -> None:
        """
        Asserts whether a file has a given extension.
        """
        assert fnmatch.fnmatch(name_file, '*' + os.extsep + extension), \
            f"The given file is not a type of <{extension}>"

    @classmethod
    def assert_file_is_xml(cls, name_file: str) -> None:
        """
        Asserts whether a file is an xml file.
        """
        cls.assert_file_extension(name_file, "xml")

    @classmethod
    def get_vehicle_type_from_filename(cls, name_file: str) -> VehicleType:
        """
        Gets the type of vehicle from the given file name.
        """
        vehicle_type = None

        if "BMW320i" in name_file or "BMW_320i" in name_file:
            vehicle_type = VehicleType.BMW_320i
        elif "FORD_ESCORT" in name_file:
            vehicle_type = VehicleType.FORD_ESCORT
        elif "VW_VANAGON" in name_file:
            vehicle_type = VehicleType.VW_VANAGON

        assert vehicle_type is not None, f"Vehicle type is unidentifiable in file: {name_file}"

        return vehicle_type

    @classmethod
    def parse_vehicle_type(cls, veh_type: str) -> VehicleType:
        """
        Parses the type of vehicle from the given type.
        """
        vehicle_type = None

        if "BMW_320i" == veh_type:
            vehicle_type = VehicleType.BMW_320i
        elif "FORD_ESCORT" == veh_type:
            vehicle_type = VehicleType.FORD_ESCORT
        elif "VW_VANAGON" == veh_type:
            vehicle_type = VehicleType.VW_VANAGON

        assert vehicle_type is not None, f"Unknown vehicle type: {veh_type}"

        return vehicle_type

    @classmethod
    def get_vehicle_type(cls, file_motion_primitive) -> VehicleType:
        """
        Gets the type of vehicle from the given motion primitive file.
        """
        cls.assert_file_is_xml(file_motion_primitive)
        # parse XML file
        xml_tree = ElementTree.parse(file_motion_primitive).getroot()
        veh_type = xml_tree.find("VehicleType").text

        return cls.parse_vehicle_type(veh_type)

    @classmethod
    def _create_automaton(cls, file_motion_primitive) -> 'ManeuverAutomaton':
        """
        Creates an automaton with the given motion primitive file.
        """
        type_veh = cls.get_vehicle_type(file_motion_primitive)
        automaton = ManeuverAutomaton(cls.__create_key, type_vehicle=type_veh)
        automaton.read_primitives_from_xml(file_motion_primitive)

        return automaton

    @staticmethod
    def generate_automaton(file_motion_primitive) -> 'ManeuverAutomaton':
        """
        Wrapper to generate an automaton with the given motion primitive file.
        """
        print(f"Reading motion primitives from file {file_motion_primitive}")
        try:
            automaton = ManeuverAutomaton._create_automaton(file_motion_primitive)
        except FileNotFoundError:
            # if the file is not found then look into GSMP/motion_automation/primitives directory
            path_motion_automaton = os.path.dirname(os.path.abspath(__file__))
            prefix = os.path.join(path_motion_automaton, "primitives")

            try:
                automaton = ManeuverAutomaton._create_automaton(os.path.join(prefix, file_motion_primitive))
            except FileNotFoundError:
                raise FileNotFoundError(f"Motion Primitive file not found at location: {file_motion_primitive}")

        automaton.examine_connectivity()
        print("Automaton created.")
        print('Number of loaded primitives: ' + str(len(automaton.list_primitives)))

        return automaton

    def save_automaton(self, file_automaton: str) -> None:
        """
        Saves automaton by simple object serialization.
        """
        self.assert_file_extension(file_automaton, ManeuverAutomaton.extension)

        with open(file_automaton, 'wb') as f:
            self.serialize()
            pickle.dump(self, f)

    @classmethod
    def load_automaton(cls, file_automaton) -> 'ManeuverAutomaton':
        """
        Loads automaton by simple object deserialization.
        """
        cls.assert_file_extension(file_automaton, ManeuverAutomaton.extension)

        if not os.path.exists(file_automaton):
            path_file_python = os.path.abspath(__file__)
            path_motion_automaton = os.path.dirname(path_file_python)
            prefix = os.path.join(path_motion_automaton, "automata")
            file_automaton = os.path.join(prefix, file_automaton)

        with open(file_automaton, 'rb') as f:
            automaton: ManeuverAutomaton = pickle.load(f)
            # restore the Primitives and Successors of primitives
            automaton.deserialize()

        return automaton

    @staticmethod
    def create_pickle(file_motion_primitive, dir_save) -> None:
        """
        Creates automaton pickle object for the given motion primitive file.

        :param file_motion_primitive: the motion primitive xml file
        :param dir_save: the directory where the created automaton will be saved
        """
        automaton = ManeuverAutomaton.generate_automaton(file_motion_primitive=file_motion_primitive)

        # cut the extension
        name_file_motion_primitive = Path(file_motion_primitive).stem

        # create new filename
        name_file_automaton = name_file_motion_primitive + os.extsep + ManeuverAutomaton.extension

        # join directory and filename
        path_automaton = os.path.join(dir_save, name_file_automaton)

        # save automaton
        automaton.save_automaton(path_automaton)

    def serialize(self):
        """
        Removes circulating references by clearing primitive.list_successors and automaton.list_primitives
        """
        for primitive in self.list_primitives:
            primitive.list_successors = []
        self.list_primitives = []

    def deserialize(self):
        """
        Restores primitive.list_successors and automaton.list_primitives from the primitive dictionary
        """
        self.list_primitives.extend(self.dict_primitives.values())
        for primitive in self.list_primitives:
            primitive.list_successors.extend(
                [self.dict_primitives.get(key) for key in self.dict_primitives.get(primitive.id).list_ids_successors])

    def read_primitives_from_xml(self, file_motion_primitive: str) -> None:
        """
        Reads all motion primitives from the given file, and stores them in the primitives array and a dictionary.

        :param file_motion_primitive: the name of the xml file which contains the motion primitives
        """
        self.assert_file_is_xml(file_motion_primitive)

        # parse XML file
        xml_tree = ElementTree.parse(file_motion_primitive).getroot()

        # get all trajectories
        list_nodes_trajectories = xml_tree.find("Trajectories").findall("Trajectory")

        for node_trajectory in list_nodes_trajectories:
            motion_primitive = MotionPrimitiveParser.create_from_node(node_trajectory)
            self.append_primitive(motion_primitive)

        self.num_primitives = len(self.dict_primitives)

        self.set_vehicle_type_for_primitives()

    def sort_primitives(self) -> None:
        """
        Sorts the primitives according to the final states y coordinate
        """
        self.list_primitives.sort(key=lambda x: x.state_final.y, reverse=False)

    def _get_unique_primitive_id(self) -> int:
        """
        Generates a new unique ID for the primitive

        :return: a unique primitive ID
        """
        self._primitive_id_counter += 1
        return self._primitive_id_counter

    def append_primitive(self, primitive: MotionPrimitive) -> None:
        """
        Appends the given primitive to the automaton if the primitive does not already exist.

        :param primitive: primitive to be added
        """
        sl = primitive.state_final
        sf = primitive.state_initial

        # optimized tuple order for faster filtering
        tuple_primitive = (sl.time_step, sl.x, sl.y, sl.orientation, sl.steering_angle, sl.velocity,
                           sf.time_step, sf.x, sf.y, sf.orientation, sf.steering_angle, sf.velocity)

        # if it is not a duplicate then add
        if tuple_primitive not in self.sets_tuples_primitives:
            self.sets_tuples_primitives.add(tuple_primitive)
            primitive.id = self._get_unique_primitive_id()
            self.dict_primitives[primitive.id] = primitive
            self.list_primitives.append(primitive)

    def get_primitives_ids_without_successor(self) -> List[int]:
        """
        Finds all primitives which has no successor.

        :return: a list of the primitives without successors
        """
        list_ids_primitives_without_successor = []
        for id_primitive, primitive in self.dict_primitives.items():
            if len(primitive.list_ids_successors) == 0:
                list_ids_primitives_without_successor.append(id_primitive)

        return list_ids_primitives_without_successor

    def prune_primitives_without_successor(self, list_ids: List[int]) -> None:
        """
        Removes primitives by the given list of ids.

        :param list_ids: the list of IDs of primitives to be removed
        """
        for id_primitive in list_ids:
            primitive_popped = self.dict_primitives.pop(id_primitive)
            self.list_primitives.remove(primitive_popped)

        set_ids_primitives_to_be_removed = set(list_ids)
        for primitive in self.dict_primitives.values():
            primitive.list_ids_successors = list(
                set(primitive.list_ids_successors).difference(set_ids_primitives_to_be_removed))

    def examine_primitive_connectivity(self, primitive_predecessor: MotionPrimitive) -> None:
        """
        Creates the successor list for a single primitive and stores them in a successor list of the given primitive.

        :param primitive_predecessor:
        """
        for primitive_successor in self.list_primitives:
            if primitive_predecessor.is_connectable(primitive_successor):
                primitive_predecessor.list_successors.append(primitive_successor)
                primitive_predecessor.list_ids_successors.append(primitive_successor.id)

    def examine_connectivity(self, verbose=False) -> None:
        """
        Creates a connectivity list for every primitive, which includes all valid successors of the primitive.
        """

        time_start = time.perf_counter()
        for primitive in tqdm(self.list_primitives):
            self.examine_primitive_connectivity(primitive)

        time_elapsed = (time.perf_counter() - time_start) * 1000
        if verbose:
            print("Connectivity examination took\t{:10.4f}\tms".format(time_elapsed))
            print(f"Primitives before pruning: {self.num_primitives}")

        list_ids_primitives_without_successor = self.get_primitives_ids_without_successor()

        while len(list_ids_primitives_without_successor) != 0:
            self.prune_primitives_without_successor(list_ids_primitives_without_successor)
            list_ids_primitives_without_successor = self.get_primitives_ids_without_successor()

        self.num_primitives = len(self.dict_primitives)
        if verbose:
            print(f"Primitives after pruning: {self.num_primitives}")

    def get_closest_initial_velocity(self, velocity_initial) -> float:
        """
        Gets the velocity among initial states that is the closest to the given initial velocity

        :param velocity_initial: the initial velocity
        :return: the closest start state velocity in the automaton
        """
        diff_velocity_min = float('inf')
        velocity_closest = None

        for primitive in self.list_primitives:
            diff_velocity = abs(velocity_initial - primitive.state_initial.velocity)
            if diff_velocity < diff_velocity_min:
                diff_velocity_min = diff_velocity
                velocity_closest = primitive.state_initial.velocity

        assert velocity_closest is not None, "Closest velocity to the planning problem not found!"

        return velocity_closest

    def set_vehicle_type_for_primitives(self) -> None:
        """
        Assigns vehicle type id to all primitives
        """
        for primitive in self.list_primitives:
            # primitive.id_type_vehicle = self.type_vehicle
            primitive.type_vehicle = self.type_vehicle

    def create_initial_motion_primitive(self, planning_problem: PlanningProblem,
                                        initial_steering_angle=0.0) -> MotionPrimitive:
        """
        Creates the initial motion primitive from the planning problem with the given initial steering angle.

        :param planning_problem: the planning problem
        :param initial_steering_angle: the initial steering angle
        :return: Initial Motion Primitive, which contains the possible successors
        """

        initial_state = planning_problem.initial_state
        initial_state.steering_angle = initial_steering_angle

        # the initial velocity of the planning problem may be any value, we need to obtain the closest velocity to it
        # from state_initials of the primitives in order to get the feasible successors of the planning problem
        initial_state.velocity = self.get_closest_initial_velocity(planning_problem.initial_state.velocity)

        # turn initial state into a motion primitive to check for connectivity to subsequent motion primitives
        state_final = MotionPrimitive.PrimitiveState(x=initial_state.position[0],
                                                     y=initial_state.position[1],
                                                     steering_angle=initial_state.steering_angle,
                                                     velocity=initial_state.velocity,
                                                     orientation=initial_state.orientation,
                                                     time_step=initial_state.time_step)

        # create a dummy initial primitive for creating and holding successors from the initial state
        # noinspection PyTypeChecker
        primitive_initial = MotionPrimitive(state_initial=None, state_final=state_final, trajectory=None,
                                            length_time_step=0)

        # create connectivity list for this imaginary motion primitive
        self.examine_primitive_connectivity(primitive_initial)

        return primitive_initial
