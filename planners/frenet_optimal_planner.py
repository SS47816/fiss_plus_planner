import copy
import math

import numpy as np
from commonroad.scenario.scenario import Scenario
from shapely import Polygon, affinity

from planners.common.cost.cost_function import CostFunction
from planners.common.geometry.cubic_spline import CubicSpline2D
from planners.common.geometry.polynomial import QuarticPolynomial, QuinticPolynomial
from planners.common.scenario.frenet import FrenetState, FrenetTrajectory
from planners.common.vehicle.vehicle import Vehicle


class Stats(object):
    def __init__(self):
        self.num_iter = 0
        self.num_trajs_generated = 0
        self.num_trajs_validated = 0
        self.num_collison_checks = 0
        # self.best_traj_costs = [] # float("inf")
        
    def __add__(self, other):
        self.num_iter += other.num_iter
        self.num_trajs_generated += other.num_trajs_generated
        self.num_trajs_validated += other.num_trajs_validated
        self.num_collison_checks += other.num_collison_checks
        # self.best_traj_costs.extend(other.best_traj_costs)
        return self
    
    def average(self, value: int):
        self.num_iter /= value
        self.num_trajs_generated /= value
        self.num_trajs_validated /= value
        self.num_collison_checks /= value
        return self
    
class FrenetOptimalPlannerSettings(object):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5):
        # time resolution between two planned waypoints
        self.tick_t = 0.1  # time tick [s]
        
        # sampling parameters
        self.max_road_width = 3.5           # maximum road width [m]
        self.num_width = num_width          # road width sampling number

        self.highest_speed = 13.4112        # highest sampling speed [m/s]
        self.lowest_speed = 0.0             # lowest sampling speed [m/s]
        self.num_speed = num_speed          # speed sampling number
        
        self.min_t = 8.0                    # min prediction time [m]
        self.max_t = 10.0                   # max prediction time [m]
        self.num_t = num_t                  # time sampling number

        self.check_obstacle = True          # True if check collison with obstacles
        self.check_boundary = True          # True if check collison with road boundaries

class FrenetOptimalPlanner(object):
    def __init__(self, planner_settings: FrenetOptimalPlannerSettings, ego_vehicle: Vehicle, scenario: Scenario=None):
        self.settings = planner_settings
        self.vehicle = ego_vehicle
        self.cost_function = CostFunction("WX1")
        self.cubic_spline = None
        self.best_traj = None
        self.all_trajs = []
        # Statistics
        self.stats = Stats()

    def calc_frenet_paths(self, frenet_state: FrenetState) -> list:
        frenet_paths = []

        sampling_width = self.settings.max_road_width - self.vehicle.w
        # lateral sampling
        traj_per_timestep = []
        for di in np.linspace(-sampling_width/2, sampling_width/2, self.settings.num_width):

            # time sampling
            for Ti in np.linspace(self.settings.min_t, self.settings.max_t, self.settings.num_t):
                fp = FrenetTrajectory()
                
                lat_qp = QuinticPolynomial(frenet_state.d, frenet_state.d_d, frenet_state.d_dd, di, 0.0, 0.0, Ti)
                fp.t = [t for t in np.arange(0.0, Ti, self.settings.tick_t)]
                fp.d = [lat_qp.calc_point(t) for t in fp.t]
                fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
                fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
                fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

                # longitudinal sampling
                for tv in np.linspace(self.settings.lowest_speed, self.settings.highest_speed, self.settings.num_speed):
                    tfp = copy.deepcopy(fp)
                    
                    lon_qp = QuarticPolynomial(frenet_state.s, frenet_state.s_d, frenet_state.s_dd, tv, 0.0, Ti)
                    tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                    tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                    tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                    tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]
                    
                    # Compute the final cost
                    tfp.cost_final = self.cost_function.cost_total(tfp, self.settings.highest_speed)
                    frenet_paths.append(tfp)
                    traj_per_timestep.append(tfp)
        self.all_trajs.append(traj_per_timestep)

        return frenet_paths

    def calc_global_paths(self, fplist: list) -> list:
        passed_fplist = []
        for fp in fplist:
            # calc global positions
            for i in range(len(fp.s)):
                ix, iy = self.cubic_spline.calc_position(fp.s[i])
                if ix is None:
                    break
                i_yaw = self.cubic_spline.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
                fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
                fp.x.append(fx)
                fp.y.append(fy)
            
            if len(fp.x) >= 2:
                # calc yaw and ds
                fp.x = np.array(fp.x)
                fp.y = np.array(fp.y)
                x_d = np.diff(fp.x)
                y_d = np.diff(fp.y)
                fp.yaw = np.arctan2(y_d, x_d)
                fp.ds = np.hypot(x_d, y_d)
                fp.yaw = np.append(fp.yaw, fp.yaw[-1])
                # calc curvature
                dt = self.settings.tick_t
                fp.c = np.divide(np.diff(fp.yaw), fp.ds)
                fp.c_d = np.divide(np.diff(fp.c), dt)
                fp.c_dd = np.divide(np.diff(fp.c_d), dt)
                
                passed_fplist.append(fp)

        return fplist
    
    def check_constraints(self, trajs: list) -> list:
        passed = []

        for i, traj in enumerate(trajs):
            # Max curvature check
            # if any([abs(c) > self.vehicle.max_curvature for c in traj.c]):
            #     continue
            # if any([abs(c_d) > self.vehicle.max_kappa_d for c_d in traj.c_d]):
            #     continue
            # if any([abs(c_dd) > self.vehicle.max_kappa_dd for c_dd in traj.c_dd]):
            #     continue
            # Max speed check
            if any([v > self.vehicle.max_speed for v in traj.s_d]):
                continue
            # Max accel check
            if any([abs(a) > self.vehicle.max_accel for a in traj.s_dd]):
                continue

            passed.append(i)
            
        return [trajs[i] for i in passed]
    
    def construct_polygon(self, polygon: Polygon, x: float, y: float, yaw: float) -> Polygon:
        polygon_translated = affinity.translate(polygon, xoff=x, yoff=y)
        polygon_rotated = affinity.rotate(polygon_translated, yaw, use_radians=True)
        
        return polygon_rotated
    
    def has_collision(self, traj: FrenetTrajectory, obstacles: list, time_step_now: int = 0, check_res: int = 1) -> tuple:
        num_polys = 0
        if len(obstacles) <= 0:
            return False, 0
        
        final_time_step = obstacles[0].prediction.final_time_step
        t_step_max = min(len(traj.x), final_time_step - time_step_now)
        for i in range(t_step_max):
            if i%check_res == 0:
                # construct a polygon for the ego vehicle at time step i
                try:
                    ego_polygon = self.construct_polygon(self.vehicle.polygon, traj.x[i], traj.y[i], traj.yaw[i])
                except:
                    print(f"Failed to create Polygon for t={i} x={traj.x[i]}, y={traj.y[i]}, yaw={traj.y[i]}")
                    return True, num_polys
                else:
                    # construct a polygon for the obstacle at time step i
                    t_step = i + time_step_now
                    for obstacle in obstacles:
                        state = obstacle.state_at_time(t_step)
                        if state is not None:
                            obstacle_polygon = self.construct_polygon(obstacle.obstacle_shape.shapely_object, state.position[0], state.position[1], state.orientation)
                            num_polys += 1
                            if ego_polygon.intersects(obstacle_polygon):
                                # plot_collision(ego_polygon, obstacle_polygon, t_step)
                                return True, num_polys

        return False, num_polys
    
    def check_collisions(self, trajs: list, obstacles: list, time_step_now: int = 0) -> list:
        passed = []

        for i, traj in enumerate(trajs):
            # Collision check
            collision, num_polys = self.has_collision(traj, obstacles, time_step_now, 2)
            if collision:
                continue

            passed.append(i)

        return [trajs[i] for i in passed]
    
    # def check_collisions(self, trajs: list[FrenetTrajectory], time_step_now: int = 0) -> list[FrenetTrajectory]:
    #     passed = []
    #     for traj in trajs:
            
    #         state_list = []
    #         initial_state = None
    #         for i in range(len(traj.x)):
    #             current_state = traj.state_at_time_step(i)
    #             current_frenet_state = traj.frenet_state_at_time_step(i)
                
    #             if i == 0:
    #                 initial_state = InitialState(time_step=time_step_now + i, 
    #                                              position=np.array([current_state.x, current_state.y]), orientation=current_state.yaw,
    #                                              velocity=current_frenet_state.s_d, acceleration=current_frenet_state.s_dd)
    #             state = PMState(time_step=time_step_now + i, position=np.array([current_state.x, current_state.y]),
    #                             velocity=current_frenet_state.s_d, velocity_y=current_frenet_state.d_d)
    #             state_list.append(state)

    #         ego_vehicle_traj = Trajectory(initial_time_step=time_step_now, state_list=state_list)
            
    #         # The ego vehicle can be visualized by converting it into a DynamicObstacle
    #         ego_vehicle_shape = Rectangle(length=self.vehicle.l, width=self.vehicle.w)
    #         ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_traj, shape=ego_vehicle_shape)
    #         ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ObstacleType.CAR,
    #                                       obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
    #                                       prediction=ego_vehicle_prediction)
    #         # create ego vehicle collision object
    #         ego_vehicle_co = create_collision_object(ego_vehicle)

    #         # check if ego vehicle collides
    #         if self.collision_checker.collide(ego_vehicle_co):
    #             continue
    #         else:
    #             passed.append(traj)

    #     return passed
    
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int = 0) -> FrenetTrajectory:
        # reset stats
        self.stats = Stats()
        self.settings.highest_speed = max_target_speed
        
        fplist = self.calc_frenet_paths(frenet_state)
        fplist = self.calc_global_paths(fplist)
        self.stats.num_trajs_generated = len(fplist)
        self.stats.num_trajs_validated = len(fplist)
        self.stats.num_collison_checks = len(fplist)
        fplist = self.check_constraints(fplist)
        # print(len(fplist), "trajectories passed constraint check")
        fplist = self.check_collisions(fplist, obstacles, time_step_now)
        # fplist = self.check_collisions(fplist, time_step_now)
        # print(len(fplist), "trajectories passed collision check")

        # find minimum cost path
        min_cost = float("inf")
        for fp in fplist:
            if min_cost >= fp.cost_final:
                min_cost = fp.cost_final
                self.best_traj = fp

        return self.best_traj

    def generate_frenet_frame(self, centerline_pts: np.ndarray):
        self.cubic_spline = CubicSpline2D(centerline_pts[:, 0], centerline_pts[:, 1])
        s = np.arange(0, self.cubic_spline.s[-1], 0.1)
        ref_xy = [self.cubic_spline.calc_position(i_s) for i_s in s]
        ref_yaw = [self.cubic_spline.calc_yaw(i_s) for i_s in s]
        ref_rk = [self.cubic_spline.calc_curvature(i_s) for i_s in s]
        return self.cubic_spline, np.column_stack((ref_xy, ref_yaw, ref_rk))
