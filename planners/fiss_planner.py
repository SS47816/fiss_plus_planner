import copy
from queue import PriorityQueue

import numpy as np
from commonroad.scenario.scenario import Scenario

from planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from planners.common.vehicle.vehicle import Vehicle
from planners.frenet_optimal_planner import FrenetOptimalPlanner, FrenetOptimalPlannerSettings, Stats


class FissPlannerSettings(FrenetOptimalPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5):
        super().__init__(num_width, num_speed, num_t)
        # heuristic cost weight
        self.w_heuristic = 10.0
        self.vis_all_candidates = False
        
class FissPlanner(FrenetOptimalPlanner):
    def __init__(self, planner_settings: FissPlannerSettings, ego_vehicle: Vehicle, scenario: Scenario=None):
        super().__init__(planner_settings, ego_vehicle, scenario)
        self.sampling_res = np.empty(3)
        self.sampling_min = np.empty(3)
        self.sampling_max = np.empty(3)
        self.candidate_trajs = PriorityQueue()
        self.trajs_3d = []
        self.sizes = None
        self.start_state = None
        self.prev_best_idx = None
        self.trajs_per_timestep = []
        
    def sample_end_frenet_states(self) -> list:
        # list of frenet end states sampled
        self.trajs_3d = []
        # Heuristic parameters
        max_sqr_dist = np.power(self.settings.num_width, 2) + np.power(self.settings.num_speed, 2) + np.power(self.settings.num_t, 2)
        
        # Define the lateral sampling positions (left being negative, right being positive)
        sampling_width = self.settings.max_road_width - self.vehicle.w + 0.3
        left_bound = -sampling_width/2
        right_bound = sampling_width/2

        # Sampling on the lateral direction
        self.sampling_min[0] = left_bound
        self.sampling_max[0] = right_bound
        d_samples, self.sampling_res[0] = np.linspace(left_bound, right_bound, self.settings.num_width, retstep=True)
        for i, d in enumerate(d_samples):
            
            trajs_2d = []
            
            # Estimate the lateral cost
            lat_norm = max(np.power(left_bound, 2), np.power(right_bound, 2))
            cost_est_lat = np.power(d, 2)/lat_norm
            
            # Longitudinal sampling
            self.sampling_min[1] = self.settings.lowest_speed
            self.sampling_max[1] = self.settings.highest_speed
            v_samples, self.sampling_res[1] = np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed, retstep=True)
            for j, v in enumerate(v_samples):
                trajs_1d = []
                
                # Estimate the speed cost
                cost_est_speed = np.power(self.settings.highest_speed - v, 2)/np.power(self.settings.highest_speed - self.settings.lowest_speed, 2)
                
                # Time sampling
                self.sampling_min[2] = self.settings.min_t
                self.sampling_max[2] = self.settings.max_t
                t_samples, self.sampling_res[2] = np.linspace(self.settings.min_t, self.settings.max_t, self.settings.num_t, retstep=True)
                for k, t in enumerate(t_samples):
                    
                    end_state = FrenetState(t=t, s=0.0, s_d=v, s_dd=0.0, s_ddd=0.0, d=d, d_d=0.0, d_dd=0.0, d_ddd=0.0)
                    
                    # Planning Horizon cost (encourage longer planning horizon)
                    cost_est_time = 1.0 - (end_state.t - self.settings.min_t)/(self.settings.max_t - self.settings.min_t)
                    
                    # Fixed cost terms
                    cost_est = cost_est_lat + cost_est_time + cost_est_speed
                    
                    # Estimated heuristic cost terms
                    if self.prev_best_idx is not None: # Add history heuristic
                        heu_sqr_dist = np.power(i - self.prev_best_idx[0], 2) + np.power(j - self.prev_best_idx[1], 2) + np.power(k - self.prev_best_idx[2], 2)
                        cost_heu = self.settings.w_heuristic * heu_sqr_dist/max_sqr_dist
                    else:
                        cost_heu = 0.0

                    # Create the trajectory (placeholders) objects and compute the estimated cost
                    traj = FrenetTrajectory()
                    traj.idx = np.array([i, j, k])
                    traj.end_state = end_state
                    traj.cost_heu = cost_heu
                    traj.cost_est = cost_est + cost_heu
                    trajs_1d.append(traj)
                    
                trajs_2d.append(trajs_1d)
                
            self.trajs_3d.append(trajs_2d)

        return self.trajs_3d

    def generate_trajectory(self, idx: np.ndarray) -> tuple:
        traj = self.trajs_3d[idx[0]][idx[1]][idx[2]]
        
        if traj.is_generated:
            return False, traj.cost_final
        else:
            self.stats.num_trajs_generated += 1
            traj.is_generated = True
            traj.idx = idx
            
            # Generate the time steps in the trajectory
            end_state = traj.end_state
            traj.t = [t for t in np.arange(0.0, end_state.t, self.settings.tick_t)]
            
            # Generate lateral quintic polynomial
            lat_qp = QuinticPolynomial(self.start_state.d, self.start_state.d_d, self.start_state.d_dd, end_state.d, end_state.d_d, end_state.d_dd, end_state.t)
            traj.d = [lat_qp.calc_point(t) for t in traj.t]
            traj.d_d = [lat_qp.calc_first_derivative(t) for t in traj.t]
            traj.d_dd = [lat_qp.calc_second_derivative(t) for t in traj.t]
            traj.d_ddd = [lat_qp.calc_third_derivative(t) for t in traj.t]
            
            # Generate longitudinal quartic polynomial
            lon_qp = QuarticPolynomial(self.start_state.s, self.start_state.s_d, self.start_state.s_dd, end_state.s_d, end_state.s_dd, end_state.t)
            traj.s = [lon_qp.calc_point(t) for t in traj.t]
            traj.s_d = [lon_qp.calc_first_derivative(t) for t in traj.t]
            traj.s_dd = [lon_qp.calc_second_derivative(t) for t in traj.t]
            traj.s_ddd = [lon_qp.calc_third_derivative(t) for t in traj.t]

            # Compute the final cost
            traj.cost_final = self.cost_function.cost_total(traj, self.settings.highest_speed)

            # Add this trajectory to the candidate queue
            self.trajs_per_timestep.append(traj)
            # self.candidate_trajs.put(traj)
            self.trajs_3d[idx[0]][idx[1]][idx[2]] = traj
            self.candidate_trajs.put((traj.cost_final, traj.idx))

            return True, traj.cost_final
    
    def find_initial_guess(self) -> np.ndarray:
        best_idx = None
        min_cost = float("inf")
        for trajs_2d in self.trajs_3d:
            for trajs_1d in trajs_2d:
                for traj in trajs_1d:
                    if not traj.is_generated and traj.cost_est <= min_cost:
                        min_cost = traj.cost_est
                        best_idx = traj.idx

        return best_idx

    def find_gradients(self, idx: np.ndarray) -> np.ndarray:
        _, cost_center = self.generate_trajectory(idx)
        
        gradients = np.empty(3)
        
        for dim in range(3):
            next_idx = copy.deepcopy(idx)
            if idx[dim] < self.sizes[dim] - 1: # the right neighbor exists
                next_idx[dim] += 1
                is_new, cost = self.generate_trajectory(next_idx)
                gradients[dim] = cost - cost_center
                if gradients[dim] >= 0 and idx[dim] == 0: # the right neighbor has higher cost and there's no left neighbor
                    gradients[dim] = 0.0
            else: # the right neighbor does not exist, calculating the gradient using the left neighbor
                next_idx[dim] -= 1
                is_new, cost = self.generate_trajectory(next_idx)
                gradients[dim] = cost_center - cost
                if gradients[dim] <= 0 and idx[dim] == self.sizes[dim]-1: # the left neighbor has higher cost and there is no right neighbor
                    gradients[dim] = 0.0
                    
        return gradients

    def explore_next_sample(self, curr_idx: np.ndarray) -> tuple:
        if self.trajs_3d[curr_idx[0]][curr_idx[1]][curr_idx[2]].is_generated:
            return True, curr_idx # converged
        else:
            next_idx = copy.deepcopy(curr_idx)
            gradients = self.find_gradients(curr_idx)
            for dim in range(3):
                if gradients[dim] > 0.0: # move in the max gradient direction, towards lower cost
                    next_idx[dim] += -1
                else:
                    next_idx[dim] += +1

            next_idx = np.clip(next_idx, 0, self.sizes - 1)

            return False, next_idx
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0) -> FrenetTrajectory:
        # Reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        self.start_state = frenet_state
        self.candidate_trajs = PriorityQueue()
        self.best_traj = None
        
        # Sample all the end states in 3 dimensions, [d, v, t] and form the 3d traj candidate array
        self.trajs_3d = self.sample_end_frenet_states()
        self.sizes = np.array([len(self.trajs_3d), len(self.trajs_3d[0]), len(self.trajs_3d[0][0])])
        
        best_idx = None
        best_traj_found = False
        
        while not best_traj_found:
            self.stats.num_iter += 1
            # ############################### Initial Guess #####################################
            if self.candidate_trajs.empty():
                best_idx = self.find_initial_guess()
                if best_idx is None:
                    # all samples have been searched and no feasible candidate found
                    # print("fiss: Searched through all trajectories, found no suitable candidate")
                    break
            else:
                # best_idx = self.candidate_trajs.queue[0].idx # peek the index of the most likely candidate
                best_idx = self.candidate_trajs.queue[0][1] # peek the index of the most likely candidate
                
            # print("fiss: initial guess index:", best_idx, "Q size:", self.candidate_trajs.qsize())
            
            # ################################ Search Process #####################################
            i = 0
            converged = False
            while not converged:
                # print("fiss: Current idx", best_idx(0), best_idx(1), best_idx(2))
                i += 1
                # Perform a search for the real best trajectory using gradients
                converged, best_idx = self.explore_next_sample(best_idx)

            # print("fiss: exploration converged in", i, "iterations")

            # ################################ Validation Process #####################################
            # print("fiss: Validating", self.candidate_trajs.size(), "Candidate Trajectories")
            if not self.candidate_trajs.empty():
                # candidate = self.candidate_trajs.get()
                _, idx = self.candidate_trajs.get()
                candidate = self.trajs_3d[idx[0]][idx[1]][idx[2]]
                self.stats.num_trajs_validated += 1
                
                # Convert to the global frame
                candidate = self.calc_global_paths([candidate])

                # Check for constraints
                passed_candidate = self.check_constraints(candidate)
                
                if passed_candidate:
                    # Check for collisions
                    safe_candidate = self.check_collisions(passed_candidate, obstacles, time_step_now)
                    self.stats.num_collison_checks += 1
                    if safe_candidate:
                        best_traj_found = True
                        self.best_traj = safe_candidate[0]
                        self.prev_best_idx = self.best_traj.idx
                        # print("fiss: Best Trajectory Found", self.best_traj.idx)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                break
        # print("fiss: Search Done in", self.stats.num_iter, "iterations, generated", self.stats.num_trajs_generated, "trajectories")

        # Convert the other unused candiates as well (for visualization only)
        if self.settings.vis_all_candidates:
            self.trajs_per_timestep = self.calc_global_paths(self.trajs_per_timestep)
        self.all_trajs.append(self.trajs_per_timestep)
        self.trajs_per_timestep = []
            
        
        return self.best_traj