import numpy as np
from planners.common.scenario.frenet import FrenetTrajectory

class CostFunction:
    def __init__(self, cost_type: str):
        if cost_type is "WX1":
            self.w_T = 10
            self.w_V = 1
            self.w_A = 0.1
            self.w_J = 0.1
            self.w_D = 0.1
            self.w_LC = 10
    
    def cost_time(self) -> float:
        pass
    
    def cost_terminal_time(self, terminal_time: float) -> float:
        pass
        # return self.w_T * terminal_time
    
    def cost_dist_obstacle(self, obstacles: list, time_step_now: int = 0) -> float:
        pass
        # dists = []
        # for obstacle in obstacles:
        #     obstacle
        # Xis = np.exp(-dists)
        # return self.w_D * sum(Xis)
    
    def cost_velocity_offset(self, vels: list, v_target: float) -> float:
        return self.w_V * sum(np.power(np.subtract(vels, v_target), 2))
    
    def cost_acceleration(self, accels: list) -> float:
        return self.w_A * sum(np.power(accels, 2))
            
    def cost_jerk(self, jerks: list) -> float:
        return self.w_J * sum(np.power(jerks, 2))
    
    def cost_lane_center_offset(self, offsets: list) -> float:
        return self.w_LC * sum(np.power(offsets, 2))
    
    def cost_total(self, traj: FrenetTrajectory, target_speed: float) -> float:
        cost_time = 10.0 - traj.t[-1] # self.cost_time()
        cost_obstacle = 0.0 # self.cost_dist_obstacle()
        cost_speed = self.cost_velocity_offset(traj.s_d, target_speed)
        cost_accel = self.cost_acceleration(traj.s_dd) + self.cost_acceleration(traj.d_dd)
        cost_jerk = self.cost_jerk(traj.s_ddd) + self.cost_jerk(traj.d_ddd)
        cost_offset = self.cost_lane_center_offset(traj.d)
        # return cost_speed + cost_accel + cost_jerk + cost_offset
        cost_total = (cost_time + cost_obstacle + cost_speed + cost_accel + cost_jerk + cost_offset)/len(traj.t)
        return cost_total