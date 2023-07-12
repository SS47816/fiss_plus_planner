import sys
import pathlib
from queue import PriorityQueue

from commonroad.scenario.scenario import Scenario
from commonroad.scenario.obstacle import Obstacle

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from common.scenario.frenet import FrenetState, FrenetTrajectory
from common.vehicle.vehicle import Vehicle
from planners.frenet_optimal_planner import Stats, FrenetOptimalPlannerSettings, FrenetOptimalPlanner

class FopPlusPlanner(FrenetOptimalPlanner):
    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle, scenario: Scenario=None):
        super().__init__(planner_settings, ego_vehicle, scenario)
        self.candidate_trajs = PriorityQueue()

    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list[Obstacle], time_step_now: int = 0) -> FrenetTrajectory:
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        
        fplist = self.calc_frenet_paths(frenet_state)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        
        self.candidate_trajs = PriorityQueue()
        for traj in fplist:
            self.candidate_trajs.put(traj)

        while not self.candidate_trajs.empty():
            self.stats.num_iter += 1
            candidate = self.candidate_trajs.get()
            passed_candidate = self.check_constraints([candidate])
            self.stats.num_trajs_validated += 1
            safe_candidate = self.check_collisions(passed_candidate, obstacles, time_step_now)
            self.stats.num_collison_checks += 1
            # safe_candidate = self.check_collisions(passed_candidate, time_step_now)
            if safe_candidate:
                self.best_traj = safe_candidate[0]
                return self.best_traj

        return None