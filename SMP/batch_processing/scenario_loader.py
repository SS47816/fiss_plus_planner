import copy
import fnmatch
import logging
import os
import pickle
import random
import time
import warnings
from enum import Enum, unique
from typing import Dict, Tuple, List

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import CostFunction, VehicleType, VehicleModel
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario

import SMP.batch_processing.helper_functions as hf
from SMP.motion_planner.motion_planner import MotionPlannerType


@unique
class ScenarioType(Enum):
    XML = "xml"
    PICKLE = "pickle"


@unique
class ScenarioLoadingMode(Enum):
    DEFAULT = 'DEFAULT'
    RANDOM = 'RANDOM'
    SPECIFIC = 'SPECIFIC'
    LEVEL_BASED = 'LEVEL_BASED'


class NoScenarioFound(FileNotFoundError):
    def __init__(self, message):
        self.message = message


class LevelBasedScenarios:
    easy_scenario_ids = [
        "BEL_Nivelles-3_2_T-1",
        "DEU_Rheinbach-5_2_T-1",
        "ESP_Cambre-3_3_T-1",
        "FRA_Miramas-2_1_T-1",
        "HRV_Pula-25_1_T-1"
    ]
    medium_scenario_ids = [
        "FRA_MoissyCramayel-1_1_T-1",
        "USA_US101-8_4_T-1"
    ]
    hard_scenario_ids = [
        "DEU_Muc-12_1_T-1"
    ]


class ScenarioLoader:

    def __init__(self, scenarios_root_folder, configuration, logger=logging.getLogger(), skip_scenarios=True,
                 verbose=False):
        self.scenarios_root_folder = scenarios_root_folder

        self.specified_scenario_ids = set(configuration['scenario_loader']['scenarios_to_run'])
        self.skip_scenario_ids = configuration['scenario_loader']['scenarios_to_skip']
        self.existing_scenario_ids, self.existing_scenarios_dict, self.found_scenarios_type \
            = self.get_existing_scenario_ids(self.scenarios_root_folder)

        # convert it to a set for efficiency
        self.existing_scenario_ids = set(self.existing_scenario_ids)

        if skip_scenarios:
            # skipping scenarios
            if len(self.skip_scenario_ids) != 0:
                self.existing_scenario_ids.difference_update(self.skip_scenario_ids)
                message = f"{len(self.skip_scenario_ids)} scenarios skipped."
                logger.info(message)
                print(message)

                if verbose:
                    for idx in self.skip_scenario_ids:
                        message = "\t{:<20}".format(idx)
                        logger.info(message)
                        print(message)

        self.scenario_loading_mode = self.parse_scenario_loading_mode(configuration)

        if self.scenario_loading_mode == ScenarioLoadingMode.DEFAULT:
            message = f"ScenarioLoader Mode <{ScenarioLoadingMode.DEFAULT.value}>"
            logger.info(message)
            print(message)

            message = "Run on all scenarios, except the skipped scenarios"
            logger.info(message)
            print(message)
            self._scenario_ids = self.existing_scenario_ids
            message = f"Number of scenarios: {len(self._scenario_ids)}"
            logger.info(message)
            print(message)
        elif self.scenario_loading_mode == ScenarioLoadingMode.SPECIFIC:
            message = f"ScenarioLoader Mode <{ScenarioLoadingMode.SPECIFIC.value}>"
            logger.info(message)
            print(message)
            # if there are specified scenarios then run only on them
            if len(self.specified_scenario_ids) != 0:
                message = "Only run on specified scenarios:"
                logger.info(message)
                print(message)
                not_found_ids = self.specified_scenario_ids.difference(self.existing_scenario_ids)
                if len(not_found_ids) != 0:
                    for idx in not_found_ids:
                        logger.warning("\t{} is NOT FOUND or SKIPPED and hence won't be processed!".format(idx))
                self._scenario_ids = self.specified_scenario_ids.intersection(self.existing_scenario_ids)

            else:
                message = "No specified Scenario found, hence run on all scenarios, expect the skipped scenarios"
                logger.info(message)
                print(message)
                self._scenario_ids = self.existing_scenario_ids
        elif self.scenario_loading_mode == ScenarioLoadingMode.RANDOM:
            message = f"ScenarioLoader Mode <{ScenarioLoadingMode.RANDOM.value}>"
            logger.info(message)
            print(message)
            # do not choose more scenarios then exists
            # TODO maybe put a warning message
            count = min(len(self.existing_scenario_ids), int(configuration['scenario_loader']['random_count']))
            self._scenario_ids = random.choices(list(self.existing_scenario_ids), k=count)

        elif self.scenario_loading_mode == ScenarioLoadingMode.LEVEL_BASED:
            message = f"ScenarioLoader Mode <{ScenarioLoadingMode.LEVEL_BASED.value}>"
            logger.info(message)
            print(message)

            self._level_based_scenarios = []
            for level in configuration['scenario_loader']['levels']:
                level = level.lower()
                if level == 'easy':
                    self._level_based_scenarios.extend(self.remove_not_existing_ids_from_list(
                        LevelBasedScenarios.easy_scenario_ids, logger))
                    continue
                if level == 'medium':
                    self._level_based_scenarios.extend(self.remove_not_existing_ids_from_list(
                        LevelBasedScenarios.medium_scenario_ids, logger))
                    continue
                if level == 'hard':
                    self._level_based_scenarios.extend(self.remove_not_existing_ids_from_list(
                        LevelBasedScenarios.hard_scenario_ids, logger))

            self._scenario_ids = self._level_based_scenarios

        # only sort if it is not level_based
        if self.scenario_loading_mode != ScenarioLoadingMode.LEVEL_BASED:
            self._scenario_ids = sorted(list(self.scenario_ids))

        if self.scenario_loading_mode != ScenarioLoadingMode.DEFAULT:
            message = f"Processing {len(self._scenario_ids)} scenarios:"
            logger.info(message)
            print(message)
            for idx, id_scenario in enumerate(self._scenario_ids):
                message = f"{idx + 1:4} {id_scenario}"
                logger.info(message)
                print(message)

        self._num_of_scenarios_to_solve = len(self.scenario_ids)

    def remove_not_existing_ids_from_list(self, my_list, logger=None) -> List:
        """
        The key aspect to keep the order of the original list while removing the elements.
        """
        ret_list = copy.deepcopy(my_list)
        for id_idx, id_to_check in enumerate(sorted(ret_list, reverse=True)):
            if id_to_check not in self.existing_scenario_ids:
                print_msg = f"\t{id_to_check} is NOT FOUND or SKIPPED and hence won't be processed!"
                if isinstance(logger, logging.Logger):
                    logger.warning(print_msg)
                else:
                    print(print_msg)

                del (ret_list[id_idx])

        return ret_list

    @property
    def scenario_ids(self):
        return self._scenario_ids

    @property
    def num_of_scenarios_to_solve(self):
        return self._num_of_scenarios_to_solve

    @staticmethod
    def _get_existing_scenarios_of_type(scenarios_root_dir, scenario_type: ScenarioType = ScenarioType.PICKLE) -> \
            Tuple[Dict[str, str], ScenarioType]:

        # +1 because of the point
        extension_length = len(scenario_type.value) + 1

        scenarios: Dict[str, str] = dict()
        for path, directories, files in os.walk(scenarios_root_dir):
            for scenario in fnmatch.filter(files, "*." + scenario_type.value):
                # res = os.path.normpath(path).split(os.path.sep)
                # rel_path_to_scenario_from_root = os.path.join(*res[1:])
                rel_path_to_scenario_from_root = path
                scenario_name = scenario[:-extension_length]  # chop the extension
                scenarios[scenario_name] = rel_path_to_scenario_from_root

        if len(scenarios) == 0:
            raise NoScenarioFound(f"No Scenario of type <{scenario_type}> found in directory: {scenarios_root_dir}")

        return scenarios, scenario_type

    @staticmethod
    def _get_existing_scenarios(scenarios_root_dir) -> Tuple[Dict[str, str], ScenarioType]:
        # if directory not exists create it
        assert os.path.exists(scenarios_root_dir), f"Scenarios root path <{scenarios_root_dir}> does not exist!"

        try:
            try:
                scenarios, found_scenarios_type = ScenarioLoader._get_existing_scenarios_of_type(scenarios_root_dir,
                                                                                                 scenario_type=
                                                                                                 ScenarioType.PICKLE)
            except NoScenarioFound:
                scenarios, found_scenarios_type = ScenarioLoader._get_existing_scenarios_of_type(scenarios_root_dir,
                                                                                                 scenario_type=
                                                                                                 ScenarioType.XML)
        except NoScenarioFound:
            raise NoScenarioFound(
                f"Neither <{ScenarioType.PICKLE}> nor <{ScenarioType.XML}> scenarios found in the directory: "
                f"{scenarios_root_dir}")

        return scenarios, found_scenarios_type

    @staticmethod
    def get_existing_scenario_ids(root_dir) -> Tuple[List[str], Dict[str, str], ScenarioType]:
        existing_scenarios, scenarios_type = ScenarioLoader._get_existing_scenarios(root_dir)
        scenario_ids = list(existing_scenarios.keys())
        return scenario_ids, existing_scenarios, scenarios_type

    def _load_xml_scenario(self, scenario_id, print_loadtime: bool = False) -> Tuple[Scenario, PlanningProblemSet]:

        scenario_path = os.path.join(self.existing_scenarios_dict[scenario_id],
                                     scenario_id + '.' + ScenarioType.XML.value)
        # open and read in scenario and planning problem set
        time1 = time.perf_counter()
        loaded_scenario_tuple = CommonRoadFileReader(scenario_path).open()
        pickle_load_time = (time.perf_counter() - time1) * 1000
        if print_loadtime:
            print("Loading scenario [{:<20}] as <xml> took\t{:10.4f}\tms".format(scenario_id, pickle_load_time))

        return loaded_scenario_tuple

    def _load_pickle_scenario(self, scenario_id, print_loadtime: bool = False) -> Tuple[Scenario, PlanningProblemSet]:
        scenario_path = os.path.join(self.existing_scenarios_dict[scenario_id],
                                     scenario_id + '.' + ScenarioType.PICKLE.value)

        with open(scenario_path, "rb") as pickle_scenario:
            time1 = time.perf_counter()
            loaded_scenario_tuple = pickle.load(pickle_scenario)
            pickle_load_time = (time.perf_counter() - time1) * 1000
            if print_loadtime:
                print("Loading scenario [{:<20}] as <pickle> took\t{:10.4f}\tms".format(scenario_id, pickle_load_time))

        return loaded_scenario_tuple

    def load_scenario(self, scenario_id, print_loadtime=False) -> Tuple[Scenario, PlanningProblemSet]:
        try:
            if self.found_scenarios_type == ScenarioType.XML:
                return self._load_xml_scenario(scenario_id, print_loadtime=print_loadtime)
            else:
                return self._load_pickle_scenario(scenario_id, print_loadtime=print_loadtime)
        except (FileNotFoundError, KeyError):
            raise FileNotFoundError(
                f"Scenario [{scenario_id}] is not found in the scenarios root directory ({self.scenarios_root_folder}) "
                f"either as <{ScenarioType.XML.value}> nor as <{ScenarioType.PICKLE.value}> format.")

    @staticmethod
    def parse_scenario_loading_mode(configuration) -> ScenarioLoadingMode:
        try:
            input_mode = configuration['scenario_loader']['inputmode']
        except KeyError:
            warnings.warn("No inputmode is given, the default will be used (load all scenarios in the root folder and "
                          "it's subfolders)")
            return ScenarioLoadingMode.DEFAULT

        input_mode = input_mode.upper()
        if input_mode == ScenarioLoadingMode.DEFAULT.value:
            return ScenarioLoadingMode.DEFAULT
        elif input_mode == ScenarioLoadingMode.RANDOM.value:
            return ScenarioLoadingMode.RANDOM
        elif input_mode == ScenarioLoadingMode.SPECIFIC.value:
            return ScenarioLoadingMode.SPECIFIC
        elif input_mode == ScenarioLoadingMode.LEVEL_BASED.value:
            return ScenarioLoadingMode.LEVEL_BASED


class ScenarioConfig:
    DEFAULT = 'default'
    VEHICLE_MODEL = 'vehicle_model'
    VEHICLE_TYPE = 'vehicle_type'
    COST_FUNCTION = 'cost_function'
    PLANNING_PROBLEM_IDX = 'planning_problem_idx'
    PLANNER = 'planner'
    MAX_TREE_DEPTH = 'max_tree_depth'
    TIMEOUT = 'timeout'

    def __init__(self, scenario_benchmark_id, configuration):
        self._scenario_id = scenario_benchmark_id

        # default configuration
        try:
            self._vehicle_model = hf.parse_vehicle_model(
                configuration[ScenarioConfig.DEFAULT][ScenarioConfig.VEHICLE_MODEL])
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.VEHICLE_MODEL} attribute!")

        try:
            self._vehicle_type, self._vehicle_type_id = hf.parse_vehicle_type(
                configuration[ScenarioConfig.DEFAULT][ScenarioConfig.VEHICLE_TYPE])
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.VEHICLE_TYPE} attribute!")

        try:
            self._cost_function = hf.parse_cost_function(
                configuration[ScenarioConfig.DEFAULT][ScenarioConfig.COST_FUNCTION])
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.COST_FUNCTION} attribute!")

        try:
            self._planning_problem_idx = int(configuration[ScenarioConfig.DEFAULT][ScenarioConfig.PLANNING_PROBLEM_IDX])
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.PLANNING_PROBLEM_IDX} attribute!")

        try:
            self._motion_planner_type = hf.parse_motion_planner_type(
                configuration[ScenarioConfig.DEFAULT][ScenarioConfig.PLANNER])
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.PLANNER} attribute!")

        try:
            self._max_tree_depth = hf.parse_max_tree_depth(
                int(configuration[ScenarioConfig.DEFAULT][ScenarioConfig.MAX_TREE_DEPTH]))
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.MAX_TREE_DEPTH} attribute!")
        try:
            self._timeout = hf.parse_timeout(int(configuration[ScenarioConfig.DEFAULT][ScenarioConfig.TIMEOUT]))
        except KeyError:
            raise AttributeError(
                f"Config file does NOT have the necessary {ScenarioConfig.DEFAULT} "
                f"{ScenarioConfig.TIMEOUT} attribute!")

        # get configuration for each scenario
        if self._scenario_id in configuration.keys():
            # configuration for specific scenario
            try:
                self._vehicle_model = hf.parse_vehicle_model(
                    configuration[self._scenario_id][ScenarioConfig.VEHICLE_MODEL])
            except KeyError:
                pass
            try:
                self._vehicle_type, self._vehicle_type_id = hf.parse_vehicle_type(
                    configuration[self._scenario_id][ScenarioConfig.VEHICLE_TYPE])
            except KeyError:
                pass
            try:
                self._cost_function = hf.parse_cost_function(
                    configuration[self._scenario_id][ScenarioConfig.COST_FUNCTION])
            except KeyError:
                pass
            try:
                self._planning_problem_idx = int(configuration[self._scenario_id][ScenarioConfig.PLANNING_PROBLEM_IDX])
            except KeyError:
                pass
            try:
                self._motion_planner_type = hf.parse_motion_planner_type(
                    configuration[self._scenario_id][ScenarioConfig.PLANNER])
            except KeyError:
                pass
            try:
                self._max_tree_depth = hf.parse_max_tree_depth(
                    int(configuration[self._scenario_id][ScenarioConfig.MAX_TREE_DEPTH]))
            except KeyError:
                pass
            try:
                self._timeout = hf.parse_timeout(int(configuration[self._scenario_id][ScenarioConfig.TIMEOUT]))
            except KeyError:
                pass

    @property
    def vehicle_model(self) -> VehicleModel:
        return self._vehicle_model

    @property
    def vehicle_type(self) -> VehicleType:
        return self._vehicle_type

    @property
    def vehicle_type_id(self) -> int:
        return self._vehicle_type_id

    @property
    def cost_function(self) -> CostFunction:
        return self._cost_function

    @property
    def planning_problem_idx(self) -> int:
        return max(self._planning_problem_idx, 0)

    @property
    def motion_planner_type(self) -> MotionPlannerType:
        return self._motion_planner_type

    @property
    def max_tree_depth(self) -> int:
        return self._max_tree_depth

    @property
    def timeout(self) -> int:
        """
        timeout value in seconds
        """
        return self._timeout
