import copy
import logging
import os
import time
import traceback
from enum import Enum, unique
from multiprocessing import Semaphore
from multiprocessing.context import Process
from typing import Tuple, List
import numpy as np

from commonroad.common.solution import PlanningProblemSolution, Solution, \
    CommonRoadSolutionWriter
from commonroad.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.scenario.state import KSState
from commonroad_dc.feasibility.solution_checker import valid_solution
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping

import SMP.batch_processing.helper_functions as hf
from SMP.batch_processing.scenario_loader import ScenarioLoader, ScenarioConfig
from SMP.maneuver_automaton.maneuver_automaton import ManeuverAutomaton
from SMP.motion_planner.motion_planner import MotionPlanner, MotionPlannerType
from SMP.motion_planner.plot_config import StudentScriptPlotConfig
import SMP.batch_processing.timeout_config


@unique
class ResultType(Enum):
    SUCCESS = 'SUCCESS'
    INVALID_SOLUTION = 'INVALID'
    FAILURE = 'FAILURE'
    EXCEPTION = 'EXCEPTION'
    TIMEOUT = 'TIMEOUT'


class ResultText:
    SUCCESS = 'Solution found:'
    INVALID_SOLUTION = 'Solution found but invalid:'
    FAILURE = 'Solution not found:'
    EXCEPTION = 'Exception occurred:'
    TIMEOUT = 'Time out:'


class SearchResult:

    def __init__(self, scenario_benchmark_id: str, result: ResultType, search_time_ms: float,
                 motion_planner_type: MotionPlannerType, error_msg: str = "",
                 list_of_list_of_states: List[List[KSState]] = None):
        self.scenario_id = scenario_benchmark_id
        self.result = result
        self.search_time_ms = search_time_ms
        self.motion_planner_type = motion_planner_type
        self.error_msg = error_msg
        self.list_of_list_of_states = list_of_list_of_states

    @property
    def search_time_sec(self) -> float:
        return self.search_time_ms / 1000

    @staticmethod
    def compute_solution_trajectory(list_of_list_of_states, rear_ax_dist) -> Trajectory:
        # add initial state - in the initial state it is quite important to only keep these 5 parameters, because a
        # trajectory can have only states with same attributes
        # the further states are coming from motion primitives so they have only these 5 attributes, so they can be
        # easily added to the list
        # the positions of states need to be shifted from the rear axis to the center point of the vehicle to match
        # the defined convention of the position in a CommonRoad Trajectory
        state = list_of_list_of_states[0][0]
        kwarg = {'position': state.position + np.array([rear_ax_dist * np.cos(state.orientation),
                                                        rear_ax_dist * np.sin(state.orientation)]),
                 'velocity': state.velocity,
                 'steering_angle': state.steering_angle,
                 'orientation': state.orientation,
                 'time_step': state.time_step}
        list_states = [KSState(**kwarg)]

        for state_list in list_of_list_of_states:
            # in the current version the first state of the list is the last state of the previous list, hence
            # duplicated, so we have to remove it
            state_list.pop(0)
            for state in state_list:
                kwarg = {'position': state.position + np.array([rear_ax_dist * np.cos(state.orientation),
                                                                rear_ax_dist * np.sin(state.orientation)]),
                         'velocity': state.velocity,
                         'steering_angle': state.steering_angle,
                         'orientation': state.orientation,
                         'time_step': state.time_step}
                list_states.append(KSState(**kwarg))

        return Trajectory(initial_time_step=list_states[0].time_step, state_list=list_states)


def get_planning_problem_and_id(planning_problem_set: PlanningProblemSet, planning_problem_idx) -> \
        Tuple[PlanningProblem, int]:
    return list(planning_problem_set.planning_problem_dict.values())[planning_problem_idx], \
           list(planning_problem_set.planning_problem_dict.keys())[planning_problem_idx]


def solve_scenario(scenario, planning_problem, automaton, config: ScenarioConfig, result_dict) -> SearchResult:
    scenario_id = str(scenario.scenario_id)

    error_msg = ""
    list_of_list_of_states = None

    def get_search_time_in_sec(start_time):
        return time.perf_counter() - start_time

    def get_search_time_in_ms(start_time):
        return get_search_time_in_sec(start_time) * 1000

    time1 = time.perf_counter()

    try:
        motion_planner = MotionPlanner.create(scenario, planning_problem, automaton=automaton,
                                              plot_config=StudentScriptPlotConfig(DO_PLOT=False),
                                              motion_planner_type=config.motion_planner_type)
        list_of_list_of_states, list_of_motion_primitives, _ = motion_planner.execute_search()
    except Exception as err:
        if str(err) == 'Time Out':
            search_time_ms = get_search_time_in_ms(time1)
            error_msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))

            result = ResultType.TIMEOUT
        # TODO consider giving -1 back because evaluating it out with excel then will be easier
        else:
            error_msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
            result = ResultType.EXCEPTION
        
            search_time_ms = get_search_time_in_ms(time1)
    else:
        if list_of_list_of_states is None:
            search_time_ms = get_search_time_in_ms(time1)
            result = ResultType.FAILURE
        else:
            search_time_ms = get_search_time_in_ms(time1)
            result = ResultType.SUCCESS

    result_dict[scenario_id] = SearchResult(scenario_id, result, search_time_ms, config.motion_planner_type, error_msg,
                                            list_of_list_of_states)

    return result_dict[scenario_id]


def save_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, planning_problem_id: int,
                  config: ScenarioConfig,
                  computation_time_in_sec: float, list_of_list_of_states: List[List[KSState]], output_path: str = './',
                  overwrite: bool = False, validate_solution: bool = True, save_gif: bool = False,
                  output_path_gif: str = './gifs', logger: logging.Logger = logging.getLogger()) -> bool:
    """

    :param scenario:
    :param planning_problem_set:
    :param planning_problem_id:
    :param config:
    :param computation_time_in_sec:
    :param list_of_list_of_states:
    :param output_path:
    :param overwrite:
    :param validate_solution:
    :param save_gif:
    :param output_path_gif:
    :param logger:
    :return: Return True if validate_solution set to False, otherwise respectively
    """

    vehicle_params = VehicleParameterMapping[config.vehicle_type.name].value

    # create solution object for benchmark
    pps = PlanningProblemSolution(planning_problem_id=planning_problem_id,
                                  vehicle_type=config.vehicle_type,
                                  vehicle_model=config.vehicle_model,
                                  cost_function=config.cost_function,
                                  trajectory=SearchResult.compute_solution_trajectory(list_of_list_of_states,
                                                                                      vehicle_params.b))

# The commentted line below uses a static methode of class to compute 
# the solution trajectory, which is shorter than the trajectory generated 
# by method create_trajectory_from_list_states() in SMP.motion_planner.utility

                                #   trajectory=SearchResult.compute_solution_trajectory(list_of_list_of_states,
                                #                                                       vehicle_params.b))

    solution = Solution(scenario.scenario_id, [pps], computation_time=computation_time_in_sec)

    # write solution to a xml file
    csw = CommonRoadSolutionWriter(solution)

    if save_gif:
        # create directory if it does not exist
        os.makedirs(output_path_gif, exist_ok=True)
        hf.save_gif2(scenario, planning_problem_set.find_planning_problem_by_id(planning_problem_id), pps.trajectory,
                     output_path=output_path_gif)

    # create directory if not exists
    os.makedirs(output_path, exist_ok=True)

    if validate_solution:
        if valid_solution(scenario, planning_problem_set, solution)[0]:
            # print(f"{scenario.benchmark_id:>30}\tSolution <VALID> - saving solution")
            csw.write_to_file(output_path=output_path, overwrite=overwrite)
        else:
            # print(f"{scenario.benchmark_id:>30}\tSolution <INVALID> - solution is not going to be saved")
            return False
    else:
        print("Saving solution WITHOUT VALIDATION!")
        csw.write_to_file(output_path=output_path, overwrite=overwrite)

    return True


def append_element_2_list_in_dict(the_dict, key, new_element, immutable_dictionary: bool = True):
    if immutable_dictionary:
        foo_list = the_dict[key]
        foo_list.append(new_element)
        the_dict[key] = foo_list
    else:
        the_dict[key].append(new_element)


def process_scenario(scenario_id, scenario_loader: ScenarioLoader, configuration_dict, def_automaton: ManeuverAutomaton,
                     result_dict, semaphore: Semaphore = None, logger: logging.Logger = logging.getLogger()
                     ):
    verbose = hf.str2bool(configuration_dict["setting"]["verbose"])
    # noinspection PyBroadException
    try:
        logger.debug("Start processing [{:<30}]".format(scenario_id))

        # Parse configuration dict
        scenario_config = ScenarioConfig(scenario_id, configuration_dict)

        # AUTOMATON preparation
        if def_automaton.type_vehicle != scenario_config.vehicle_type:
            # if the defined vehicle type differs from the default one then load custom automaton instead
            try:
                automaton = ManeuverAutomaton.load_automaton(
                    hf.get_default_automaton_by_veh_id(scenario_config.vehicle_type_id, configuration_dict))
            except FileNotFoundError:
                try:
                    automaton = ManeuverAutomaton.generate_automaton(
                        hf.get_default_motion_primitive_file_by_veh_id(scenario_config.vehicle_type_id,
                                                                       configuration_dict))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"No default MotionAutomaton found for vehicle type id: {scenario_config.vehicle_type}")
        else:
            # use the default automaton file if there is nothing else specified
            automaton = copy.deepcopy(def_automaton)
            automaton.deserialize()

        # Loading Scenario and Planning Problem Set
        scenario, planning_problem_set = scenario_loader.load_scenario(scenario_id)

        # Retrieve Planning Problem with given index (for cooperative scenario:0, 1, 2, ..., otherwise: 0)
        # with the GSMP approach we do not want to solve cooperative scenarios so in all cases we will have
        # only one planning problem
        planning_problem, planning_problem_id = get_planning_problem_and_id(planning_problem_set,
                                                                            scenario_config.planning_problem_idx)

        # Initial result
        result_dict[scenario_id] = None

        p = Process(target=solve_scenario, args=(scenario, planning_problem, automaton, scenario_config, result_dict))
        p.start()
        p.join(timeout=scenario_config.timeout)

        # if TIMEOUT
        if p.is_alive():
            p.terminate()
            result_dict[scenario_id] = SearchResult(scenario_id, ResultType.TIMEOUT, scenario_config.timeout,
                                                    scenario_config.motion_planner_type)

        if isinstance(result_dict[scenario_id], SearchResult):
            search_result: SearchResult = result_dict[scenario_id]

            if search_result.result == ResultType.TIMEOUT:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Timeout time [s]:  {int(scenario_config.timeout)}")
                append_element_2_list_in_dict(result_dict, ResultType.TIMEOUT, scenario_id)

            elif search_result.result == ResultType.EXCEPTION:
                if verbose:
                    print(f"\n{scenario_id:<25}  {search_result.result}  {search_result.error_msg}")
                append_element_2_list_in_dict(result_dict, ResultType.EXCEPTION, scenario_id)

            elif search_result.result == ResultType.FAILURE:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Computation time [ms]:  {int(search_result.search_time_ms)}"
                        f"  <{scenario_config.motion_planner_type}>  DID NOT FIND a solution.")
                append_element_2_list_in_dict(result_dict, ResultType.FAILURE, scenario_id)

            else:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Computation time [ms]:  {int(search_result.search_time_ms)}  "
                        f"<{scenario_config.motion_planner_type}>  FOUND a solution.")

                is_valid_solution = save_solution(scenario=scenario, planning_problem_set=planning_problem_set,
                                                  planning_problem_id=planning_problem_id, config=scenario_config,
                                                  computation_time_in_sec=search_result.search_time_sec,
                                                  list_of_list_of_states=search_result.list_of_list_of_states,
                                                  output_path=configuration_dict['setting']['output_path'],
                                                  overwrite=hf.str2bool(configuration_dict['setting']['overwrite']),
                                                  validate_solution=hf.str2bool(
                                                      configuration_dict['setting']['validate_solution']),
                                                  save_gif=hf.str2bool(
                                                      configuration_dict['setting']['create_gif']),
                                                  output_path_gif=configuration_dict['setting']['output_path_gif'],
                                                  logger=logger)
                if is_valid_solution:
                    append_element_2_list_in_dict(result_dict, ResultType.SUCCESS, scenario_id)
                else:
                    # replacing the search result in the statistics
                    tmp_sr = result_dict[scenario_id]
                    result_dict[scenario_id] = SearchResult(
                        scenario_benchmark_id=scenario_id,
                        result=ResultType.INVALID_SOLUTION,
                        search_time_ms=tmp_sr.search_time_ms,
                        motion_planner_type=scenario_config.motion_planner_type,
                        list_of_list_of_states=tmp_sr.list_of_list_of_states)

                    append_element_2_list_in_dict(result_dict, ResultType.INVALID_SOLUTION, scenario_id)
    except Exception as err:
        print("Something went wrong while processing scenario {:<30}]".format(scenario_id))
        error_msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        print(error_msg)

    if semaphore is not None:
        semaphore.release()


def debug_scenario(scenario_id, scenario_loader: ScenarioLoader, configuration_dict, def_automaton: ManeuverAutomaton,
                   result_dict, logger: logging.Logger = logging.getLogger(),
                   verbose=False):
    # noinspection PyBroadException
    try:
        logger.debug("Start processing [{:<30}]".format(scenario_id))

        # Parse configuration dict
        scenario_config = ScenarioConfig(scenario_id, configuration_dict)
        
        SMP.batch_processing.timeout_config.timeout = scenario_config.timeout

        # AUTOMATON preparation
        if def_automaton.type_vehicle != scenario_config.vehicle_type:
            # if the defined vehicle type differs from the default one then load custom automaton instead
            try:
                automaton = ManeuverAutomaton.load_automaton(
                    hf.get_default_automaton_by_veh_id(scenario_config.vehicle_type_id, configuration_dict))
            except FileNotFoundError:
                try:
                    automaton = ManeuverAutomaton.generate_automaton(
                        hf.get_default_motion_primitive_file_by_veh_id(scenario_config.vehicle_type_id,
                                                                       configuration_dict))
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"No default MotionAutomaton found for vehicle type id: {scenario_config.vehicle_type}")
        else:
            # use the default automaton file if there is nothing else specified
            automaton = copy.deepcopy(def_automaton)
            automaton.deserialize()

        # Loading Scenario and Planning Problem Set
        scenario, planning_problem_set = scenario_loader.load_scenario(scenario_id)

        # Retrieve Planning Problem with given index (for cooperative scenario:0, 1, 2, ..., otherwise: 0)
        # with the GSMP approach we do not want to solve cooperative scenarios so in all cases we will have
        # only one planning problem
        planning_problem, planning_problem_id = get_planning_problem_and_id(planning_problem_set,
                                                                            scenario_config.planning_problem_idx)

        # Initial result
        result_dict[scenario_id] = None

        # Solving scenario
        solve_scenario(scenario, planning_problem, automaton, scenario_config, result_dict)

        if isinstance(result_dict[scenario_id], SearchResult):
            search_result: SearchResult = result_dict[scenario_id]

            if search_result.result == ResultType.TIMEOUT:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Timeout time [s]:  {int(scenario_config.timeout)}")
                append_element_2_list_in_dict(result_dict, ResultType.TIMEOUT, scenario_id, immutable_dictionary=False)

            elif search_result.result == ResultType.EXCEPTION:
                if verbose:
                    print(f"\n{scenario_id:<25}  {search_result.result}  {search_result.error_msg}")
                append_element_2_list_in_dict(result_dict, ResultType.EXCEPTION, scenario_id,
                                              immutable_dictionary=False)

            elif search_result.result == ResultType.FAILURE:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Computation time [ms]:  {int(search_result.search_time_ms)}"
                        f"  <{scenario_config.motion_planner_type}>  DID NOT FIND a solution.")
                append_element_2_list_in_dict(result_dict, ResultType.FAILURE, scenario_id, immutable_dictionary=False)

            else:
                if verbose:
                    print(
                        f"\n{scenario_id:<25}  {search_result.result}  Computation time [ms]:  {int(search_result.search_time_ms)}  "
                        f"<{scenario_config.motion_planner_type}>  FOUND a solution.")

                is_valid_solution = save_solution(scenario=scenario, planning_problem_set=planning_problem_set,
                                                  planning_problem_id=planning_problem_id, config=scenario_config,
                                                  computation_time_in_sec=search_result.search_time_sec,
                                                  list_of_list_of_states=search_result.list_of_list_of_states,
                                                  output_path=configuration_dict['setting']['output_path'],
                                                  overwrite=hf.str2bool(configuration_dict['setting']['overwrite']),
                                                  validate_solution=hf.str2bool(
                                                      configuration_dict['setting']['validate_solution']),
                                                  save_gif=hf.str2bool(
                                                      configuration_dict['setting']['create_gif']),
                                                  output_path_gif=configuration_dict['setting']['output_path_gif'],
                                                  logger=logger)
                if is_valid_solution:
                    append_element_2_list_in_dict(result_dict, ResultType.SUCCESS, scenario_id,
                                                  immutable_dictionary=False)
                else:
                    # replacing the search result in the statistics
                    tmp_sr = result_dict[scenario_id]
                    result_dict[scenario_id] = SearchResult(
                        scenario_benchmark_id=scenario_id,
                        result=ResultType.INVALID_SOLUTION,
                        search_time_ms=tmp_sr.search_time_ms,
                        motion_planner_type=scenario_config.motion_planner_type,
                        list_of_list_of_states=tmp_sr.list_of_list_of_states)

                    append_element_2_list_in_dict(result_dict, ResultType.INVALID_SOLUTION, scenario_id,
                                                  immutable_dictionary=False)
    except Exception as err:
        print("Something went wrong while processing scenario {:<30}]".format(scenario_id))
        error_msg = "".join(traceback.format_exception(type(err), err, err.__traceback__))
        print(error_msg)
