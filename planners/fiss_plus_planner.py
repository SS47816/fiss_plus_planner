import copy
import time
from queue import PriorityQueue

import numpy as np
from commonroad.scenario.scenario import Scenario

from planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from planners.common.vehicle.vehicle import Vehicle
from planners.fiss_planner import FissPlanner, FissPlannerSettings
from planners.frenet_optimal_planner import Stats


class FissPlusPlannerSettings(FissPlannerSettings):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, refine_iters: int = 3):
        super().__init__(num_width, num_speed, num_t)
        self.refine_trajectory = True
        self.max_refine_iters = refine_iters
        self.has_time_limit = False
        self.time_limit = 0.5
        self.decaying_factor = 0.5

class FissPlusPlanner(FissPlanner):
    def __init__(self, planner_settings: FissPlusPlannerSettings, ego_vehicle: Vehicle, scenario: Scenario=None):
        super().__init__(planner_settings, ego_vehicle, scenario)
        self.frontier_idxs = PriorityQueue()
        self.refined_trajs = PriorityQueue()
        
    def explore_neighbors(self, idx: np.ndarray) -> tuple:
        _, cost_center = self.generate_trajectory(idx)
        min_cost = copy.deepcopy(cost_center)
        best_idx = copy.deepcopy(idx)
        is_local_minimum = True
        
        # explore all six neighbors on three dimensions
        for dim in range(3):
            if idx[dim] >= 1: # left neighbor exists
                prev_idx = copy.deepcopy(idx)
                prev_idx[dim] -= 1
                is_new, cost = self.generate_trajectory(prev_idx)
                if is_new and cost <= cost_center:
                    self.frontier_idxs.put((cost, prev_idx))
                if cost <= min_cost:
                    min_cost = cost
                    best_idx = prev_idx
                    is_local_minimum = False
            if idx[dim] < self.sizes[dim] - 1: # right neighbor exists
                next_idx = copy.deepcopy(idx)
                next_idx[dim] += 1
                is_new, cost = self.generate_trajectory(next_idx)
                if is_new and cost <= cost_center:
                    self.frontier_idxs.put((cost, next_idx))
                if cost <= min_cost:
                    min_cost = cost
                    best_idx = next_idx
                    is_local_minimum = False
        
        return is_local_minimum, best_idx
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0) -> FrenetTrajectory:
        t_start = time.time()
        
        # Reset values for each planning cycle
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        self.start_state = frenet_state
        self.frontier_idxs = PriorityQueue()
        self.candidate_trajs = PriorityQueue()
        self.refined_trajs = PriorityQueue()
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
                    # print("fiss+: Searched through all trajectories, found no suitable candidate")
                    break
            else:
                # best_idx = self.candidate_trajs.queue[0].idx # peek the index of the most likely candidate
                best_idx = self.candidate_trajs.queue[0][1] # peek the index of the most likely candidate
                
            # Ablation Study
            # _, cost = self.generate_trajectory(best_idx)
            # candidate = self.trajs_3d[best_idx[0]][best_idx[1]][best_idx[2]]
            # candidate = self.calc_global_paths([candidate])
            # self.best_traj = candidate[0]
            # best_traj_found = True
                
            # print("fiss+: initial guess index:", best_idx, "Q size:", self.candidate_trajs.qsize())
            
            # ################################ Search Process #####################################
            
            i = 0
            converged = False
            while not converged:
                # print("fiss+: Current idx", best_idx(0), best_idx(1), best_idx(2))
                i += 1
                # Perform a search for the real best trajectory using gradients
                is_minimum, best_idx = self.explore_neighbors(best_idx)
                if self.frontier_idxs.empty():
                    converged = True
                else:
                    _, best_idx = self.frontier_idxs.get()

            # print("fiss+: exploration converged in", i, "iterations")

            # ################################ Validation Process #####################################
            
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
                        # print("fiss+: Best Trajectory Found", self.best_traj.idx)
                        break
                    else:
                        continue
                else:
                    continue
            else:
                break
        
        # print("fiss+: Search Done in", self.stats.num_iter, "iterations, generated", self.stats.num_trajs_generated, "trajectories")
        
        if best_traj_found and self.settings.refine_trajectory:
            time_spent = time.time() - t_start
            time_left = self.settings.time_limit - time_spent
            
            if not self.settings.has_time_limit or time_left > 0.0:
                # print(f"fiss+: Coarse Search took {time_spent:.2f} s/{self.settings.time_limit:.2f} s, start refinement")
                refined_traj = self.refine_solution(self.best_traj, time_left, obstacles, time_step_now)
                if refined_traj is not None:
                    self.best_traj = refined_traj
            # else:
            #     print(f"fiss+: Coarse Search took {time_spent:.2f} s{self.settings.time_limit:.2f} s, no time for refinement")
                
        # Convert the other unused candiates as well (for visualization only)
        if self.settings.vis_all_candidates:
            self.trajs_per_timestep = self.calc_global_paths(self.trajs_per_timestep)
        self.all_trajs.append(self.trajs_per_timestep)
        self.trajs_per_timestep = []
        # self.prev_best_grad = 
        return self.best_traj
        
    def generate_trajectory_by_end_state(self, end_state: FrenetState) -> float:
        # Create the trajectory
        traj = FrenetTrajectory()
        
        # Generate the end state by given specs
        traj.end_state = end_state
        
        self.stats.num_trajs_generated += 1
        traj.is_generated = True
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
        self.refined_trajs.put(traj)
        if self.settings.vis_all_candidates:
            self.all_trajs.append(traj)
            
        return traj.cost_final
    
    def gradient_decent(self, J: float, x: np.ndarray, resolutions: np.ndarray, decaying_factor: float) -> tuple:
        # Compute the initial gradient at the best trajectory
        d_J = np.empty(3)
        d_x = np.empty(3)
        # print("initial x:", x)
        
        for dim in range(3):
            # print("dim:", dim)
            # print("resolutions:", resolutions[dim])
            
            # left neighbor
            x_l = copy.deepcopy(x)
            x_l[dim] -= resolutions[dim]
            x_l = np.clip(x_l, self.sampling_min, self.sampling_max)
            end_state_l = FrenetState(t=x_l[2], s=0.0, s_d=x_l[1], s_dd=0.0, s_ddd=0.0, d=x_l[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            J_l = self.generate_trajectory_by_end_state(end_state_l)
            # print("x_l:", x_l)
            # print("end_state_l:", end_state_l)
            
            # right neighbor
            x_r = copy.deepcopy(x)
            x_r[dim] += resolutions[dim]
            x_r = np.clip(x_r, self.sampling_min, self.sampling_max)
            end_state_r = FrenetState(t=x_r[2], s=0.0, s_d=x_r[1], s_dd=0.0, s_ddd=0.0, d=x_r[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            J_r = self.generate_trajectory_by_end_state(end_state_r)
            # print("x_r:", x_r)
            # print("end_state_r:", end_state_r)
            
            # gradient
            d_J[dim] = J_r - J_l
            d_x[dim] = x_r[dim] - x_l[dim]
            
            # if x[dim] + resolutions[dim] <= self.sampling_max[dim]: # the right neighbor exists
            #     x_r = copy.deepcopy(x)
            #     x_r[dim] = x[dim] + resolutions[dim]
            #     end_state_r = FrenetState(t=x_r[2], s=0.0, s_d=x_r[1], s_dd=0.0, s_ddd=0.0, d=x_r[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            #     J_r = self.generate_trajectory_by_end_state(end_state_r)
            #     d_J[dim] = J_r - J
            #     d_x[dim] = x_r[dim] - x[dim]
            # else: # the right neighbor does not exist, calculating the gradient using the left neighbor
            #     x_l = copy.deepcopy(x)
            #     x_l[dim] = x[dim] - resolutions[dim]
            #     end_state_l = FrenetState(t=x_l[2], s=0.0, s_d=x_l[1], s_dd=0.0, s_ddd=0.0, d=x_l[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
            #     J_l = self.generate_trajectory_by_end_state(end_state_l)
            #     d_J[dim] = J - J_l
            #     d_x[dim] = x[dim] - x_l[dim]
            
            # print("d_J[dim]:", d_J[dim], "d_x[dim]:", d_x[dim])
            # print("Refined Q size:", self.refined_trajs.qsize())
            
        # Compute the location of the next candidate
        # print("d_J:", d_J, "d_x:", d_x)
        grad = d_J/d_x
        # print("grad:", grad)
        resolutions *= decaying_factor
        # x_new = x - np.linalg.norm(resolutions) * grad/np.linalg.norm(grad)
        x_new = x - resolutions * grad/np.linalg.norm(grad)
        # print("x_new:", x_new)
        
        # Validate the location of the next candidate is within the sampling region
        x_new_clipped = np.clip(x_new, self.sampling_min, self.sampling_max)

        # Generate the next candiate
        end_state_new = FrenetState(t=x_new_clipped[2], s=0.0, s_d=x_new_clipped[1], s_dd=0.0, s_ddd=0.0, d=x_new_clipped[0], d_d=0.0, d_dd=0.0, d_ddd=0.0)
        J_new = self.generate_trajectory_by_end_state(end_state_new)
        # print(f"J_new: {J_new:.2f}, J_coarse: {self.best_traj.cost_final:.2f}, improved: {(self.best_traj.cost_final - J_new):.2f}")
        # if np.isnan(x_new).any():
        #     return False, 0.0, x_new, resolutions
        # if x_new_clipped != x_new:
        #     return False, 0.0, x_new, resolutions
        return True, J_new, x_new_clipped, resolutions
    
    def refine_solution(self, traj: FrenetTrajectory, time_limit: float, obstacles: list, time_step_now: int) -> FrenetTrajectory:
        # Initialization
        t_start = time.time()
        resolutions = self.sampling_res
        # print("resolutions:", resolutions)
        alpha = self.settings.decaying_factor
        
        J_new = traj.cost_final
        x = np.array([traj.end_state.d, traj.end_state.s_d, traj.end_state.t])
        
        for i in range(self.settings.max_refine_iters):
            valid, J_new, x, resolutions = self.gradient_decent(J_new, x, resolutions, alpha)
            # if resolutions <= self.threshold:
            if not valid:
                # print("fiss+: New location outside sampling region")
                break
            
            t_spent = time.time() - t_start
            if t_spent >= time_limit:
                # print("fiss+: Refinement time is up")
                break
        
        while not self.refined_trajs.empty():
            candidate = self.refined_trajs.get()
            if candidate.cost_final > traj.cost_final:
                break

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
                    # print("fiss+: Refined Trajectory Cost:", safe_candidate[0].cost_final, "Coarse Trajectory Cost:", traj.cost_final)
                    return safe_candidate[0]
                else:
                    continue
            else:
                continue
        
        # print("fiss+: Refined Trajectories Not Feasible")
        return None