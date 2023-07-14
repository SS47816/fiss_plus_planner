import argparse
import glob
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

from commonroad.scenario.obstacle import Obstacle, DynamicObstacle, ObstacleType
from commonroad.geometry.shape import Rectangle
from commonroad.scenario.state import InitialState, CustomState
from commonroad.scenario.trajectory import Trajectory
from commonroad.prediction.prediction import TrajectoryPrediction

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping, VehicleType

from common.vehicle.vehicle import Vehicle
from common.scenario.frenet import State, FrenetState
from fiss_planner.fiss_plus_planner import FissPlusPlannerSettings, FissPlusPlanner

def convert_waymo_obstacle_to_cr(waymo_trajs: np.ndarray, waymo_traj_masks: np.ndarray) -> list:
    # Create Obstacles
    obstacles = []
    N_obs, N_t, N_f = waymo_trajs.shape
    
    for i in range(N_obs):
        obstacle_traj = waymo_trajs[i]
        if obstacle_traj.shape[0] <= 0:
            continue
        dynamic_obstacle_shape = Rectangle(
            width=obstacle_traj[0, 4], length=obstacle_traj[0, 3])
        # initial state has a time step of 0
        dynamic_obstacle_initial_state = CustomState(position=np.array([obstacle_traj[0, 0], obstacle_traj[0, 1]]),
                                                        velocity=np.hypot(
                                                            obstacle_traj[0, 7], obstacle_traj[0, 8]),
                                                        orientation=obstacle_traj[0, 6],
                                                        time_step=0)

        # create the trajectory of the obstacle, starting at time step 1
        state_list = []
        for t in range(1, N_t):
            if waymo_traj_masks[i, t]:
                # create new state
                new_state = CustomState(position=np.array([obstacle_traj[t, 0], obstacle_traj[t, 1]]),
                                        velocity=np.hypot(
                                            obstacle_traj[t, 7], obstacle_traj[t, 8]),
                                        orientation=obstacle_traj[t, 6],
                                        time_step=t)
            else:
                # new_state = CustomState(time_step=t)
                break
            # add new state to state_list
            state_list.append(new_state)
        
        if state_list:
            dynamic_obstacle_trajectory = Trajectory(1, state_list)

            # create the prediction using the trajectory and the shape of the obstacle
            dynamic_obstacle_prediction = TrajectoryPrediction(
                dynamic_obstacle_trajectory, dynamic_obstacle_shape)

            # generate the dynamic obstacle according to the specification
            dynamic_obstacle_id = i
            dynamic_obstacle_type = ObstacleType.CAR
            dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id,
                                                dynamic_obstacle_type,
                                                dynamic_obstacle_shape,
                                                dynamic_obstacle_initial_state,
                                                dynamic_obstacle_prediction)

            obstacles.append(dynamic_obstacle)
            
    return obstacles

def convert_cr_traj_to_waymo(cr_traj: np.ndarray, vehicle: Vehicle) -> np.ndarray:
    waymo_traj = np.vstack([cr_traj.x,
                            cr_traj.y,
                            np.zeros_like(cr_traj.x),
                            np.full_like(cr_traj.x, vehicle.l),
                            np.full_like(cr_traj.x, vehicle.w),
                            np.full_like(cr_traj.x, vehicle.h),
                            cr_traj.yaw,
                            np.zeros_like(cr_traj.x),  # cr_traj.vx,
                            np.zeros_like(cr_traj.x),  # cr_traj.vy,
                            np.full_like(cr_traj.x, 1, dtype=int),
                            np.full_like(cr_traj.x, 1, dtype=int)])
    return np.transpose(waymo_traj)

def polygon_completion(polygon):
    polyline_x = []
    polyline_y = []

    for i in range(len(polygon)):
        if i+1 < len(polygon):
            next = i+1
        else:
            next = 0
        dist_x = polygon[next, 0] - polygon[i, 0]
        dist_y = polygon[next, 1] - polygon[i, 1]
        dist = np.linalg.norm([dist_x, dist_y])
        interp_num = np.ceil(dist) * 2
        interp_index = np.arange(2 + interp_num)
        point_x = np.interp(
            interp_index, [0, interp_index[-1]], [polygon[i, 0], polygon[next, 0]]).tolist()
        point_y = np.interp(
            interp_index, [0, interp_index[-1]], [polygon[i, 1], polygon[next, 1]]).tolist()
        polyline_x.extend(point_x[:-1])
        polyline_y.extend(point_y[:-1])

    return np.array([polyline_x, polyline_y]).T


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--load_path', default="./data/waymo/processed_scenarios_training",
                        type=str, help='path to dataset files')

    args = parser.parse_args()
    data_files = glob.glob(args.load_path+'/*')

    scene_id = 0
    for file in data_files:
        with open(file, 'rb') as f:
            info = pickle.load(f)
        scene_id += 1
        print(f"Scene {scene_id}")
        sdc_track_index = info['sdc_track_index']
        track_index_to_predict = np.array(info['predict_list'])
        obj_types = np.array(info['object_type'])
        obj_ids = np.array(info['object_id'])
        obj_trajs_full = info['all_agent']
        # [N_p, 7] [x, y, z, ori_x, ori_y, ori_z, type, theta] --> [N_l, 50, 7]
        all_polylines = info['all_polylines']
        lane_polylines = info['lane']
        crosswalks_p = info['crosswalk']
        speed_bump = info['speed_bump']
        driveway = info['drive_way']
        stop_sign = info['stop_sign']
        road_polylines = info['road_polylines']
        ego_traj = obj_trajs_full[sdc_track_index]  # [91, 11]
        ego_mask = ego_traj[:, 9] > 0                 # [91,]
        # [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid, object_type] --> # [center_x, center_y, length, width, heading, velocity_x, velocity_y, object_type]
        ego_traj_hitory = ego_traj[0:11]      # [T=11, F=11]
        ego_traj_hitory_mask = ego_mask[0:11]  # [11]
        ego_traj_future = ego_traj[11:]       # [T=80, F=11]
        ego_traj_future_mask = ego_mask[11:]  # [80]

        other_traj = np.concatenate(
            (obj_trajs_full[:sdc_track_index], obj_trajs_full[sdc_track_index+1:]), axis=0)  # [N_o, 91, 11]
        other_mask = other_traj[:, :, 9] > 0            # [N_0, 91]
        other_traj_hitory = other_traj[:, 0:11]      # [N_o, 11, 11]
        other_traj_hitory_mask = other_mask[:, 0:11]  # [N_o, 11]
        other_traj_future = other_traj[:, 11:]       # [N_o, 80, 11]
        other_traj_future_mask = other_mask[:, 11:]  # [N_o, 11]

        ref_path = info['ref_path']
        # [1200,5]
        # [x,y,theta,curvature,one of [speedlimit,trafficlight,crosswalks]]## 0 is red light, 1 is crosswalk, other is speed_limi

        ###########################################   Convertion from Waymo to CommonRoad  ####################################################
        obstacles_all = convert_waymo_obstacle_to_cr(other_traj_future, other_traj_future_mask)

        ###########################################   Planning   ####################################################
        # Initialize local planner
        d_t = 0.1
        vehicle_type = VehicleType.VW_VANAGON
        vehicle_params = VehicleParameterMapping[vehicle_type.name].value
        vehicle = Vehicle(vehicle_params)
        planner_settings = FissPlusPlannerSettings(9, 9, 3, 3)
        planner = FissPlusPlanner(planner_settings, vehicle)

        # Establish frenet frame
        ego_lane_pts = ref_path[:, :3]  # [x, y, yaw, optional: width]
        csp_ego, ref_ego_lane_pts = planner.generate_frenet_frame(ego_lane_pts)

        # Calculate the starting state
        current_frenet_state = FrenetState()
        initial_state = ego_traj_hitory[-1]
        start_state = State(t=0.0, x=initial_state[0], y=initial_state[1],
                            yaw=initial_state[6], v=np.hypot(initial_state[7], initial_state[8]), a=0.0)
        current_frenet_state.from_state(start_state, ref_ego_lane_pts)

        # Plan
        best_traj_ego = planner.plan(
            current_frenet_state, np.amax(ref_path[:, -1]), obstacles_all, 0)

        ###########################################   Convertion from CommonRoad to Waymo   ####################################################
        # Convert the result to waymo style [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid, object_type]
        ego_traj_planned = ego_traj_future
        if best_traj_ego:
            ego_traj_planned = convert_cr_traj_to_waymo(best_traj_ego, vehicle)
            print(f"Planning Successfully {ego_traj_planned.shape}")

        ##############################################   VIZ   ####################################################
        # viz ego
        color = 'r'
        rect = plt.Rectangle((ego_traj_hitory[-1, 0]-ego_traj_hitory[-1, 3]/2, ego_traj_hitory[-1, 1]-ego_traj_hitory[-1, 4]/2),
                             ego_traj_hitory[-1, 3], ego_traj_hitory[-1, 4], linewidth=2, color=color, alpha=0.6, zorder=6,
                             transform=mpl.transforms.Affine2D().rotate_around(*(ego_traj_hitory[-1, 0], ego_traj_hitory[-1, 1]), ego_traj_hitory[-1, 6]) + plt.gca().transData)
        plt.gca().add_patch(rect)
        plt.plot(ego_traj_hitory[ego_traj_hitory_mask][::3, 0], ego_traj_hitory[ego_traj_hitory_mask]
                 [::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=5)
        plt.plot(ego_traj_future[ego_traj_future_mask][::5, 0], ego_traj_future[ego_traj_future_mask]
                 [::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)
        plt.plot(ego_traj_planned[::5, 0], ego_traj_planned[::5, 1],
                 linewidth=2, color='cyan', marker='*', markersize=6, zorder=10)

        # viz other
        for i in range(other_traj_hitory.shape[0]):
            if other_traj_hitory[i, 0, 10] == 1 and other_traj_hitory[i, 10, 9] != 0:  # vehicle
                color = 'm'
                # (num_timestamps=11, num_attrs=29)
                single_traj = other_traj_hitory[i]
                single_traj_mask = other_traj_hitory_mask[i]
                single_traj_future = other_traj_future[i]
                single_traj_future_mask = (other_traj_future_mask[i] == 1)
                rect = plt.Rectangle((single_traj[-1, 0]-single_traj[-1, 3]/2, single_traj[-1, 1]-single_traj[-1, 4]/2),
                                     single_traj[-1, 3], single_traj[-1, 4], linewidth=2, color=color, alpha=0.6, zorder=5,
                                     transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1, 0], single_traj[-1, 1]), single_traj[-1, 6]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask]
                         [::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
                plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask]
                         [::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)

            if other_traj_hitory[i, 0, 10] == 2 and other_traj_hitory[i, 10, 9] != 0:  # vehicle
                color = 'b'
                # (num_timestamps=11, num_attrs=29)
                single_traj = other_traj_hitory[i]
                single_traj_mask = other_traj_hitory_mask[i]
                single_traj_future = other_traj_future[i]
                single_traj_future_mask = (other_traj_future_mask[i] == 1)
                rect = plt.Rectangle((single_traj[-1, 0]-single_traj[-1, 3]/2, single_traj[-1, 1]-single_traj[-1, 4]/2),
                                     single_traj[-1, 3], single_traj[-1, 4], linewidth=2, color=color, alpha=0.6, zorder=5,
                                     transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1, 0], single_traj[-1, 1]), single_traj[-1, 6]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask]
                         [::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
                plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask]
                         [::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)

            if other_traj_hitory[i, 0, 10] == 3 and other_traj_hitory[i, 10, 9] != 0:  # vehicle
                color = 'g'
                # (num_timestamps=11, num_attrs=29)
                single_traj = other_traj_hitory[i]
                single_traj_mask = other_traj_hitory_mask[i]
                single_traj_future = other_traj_future[i]
                single_traj_future_mask = (other_traj_future_mask[i] == 1)
                rect = plt.Rectangle((single_traj[-1, 0]-single_traj[-1, 3]/2, single_traj[-1, 1]-single_traj[-1, 4]/2),
                                     single_traj[-1, 3], single_traj[-1, 4], linewidth=2, color=color, alpha=0.6, zorder=5,
                                     transform=mpl.transforms.Affine2D().rotate_around(*(single_traj[-1, 0], single_traj[-1, 1]), single_traj[-1, 6]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                plt.plot(single_traj[single_traj_mask][::3, 0], single_traj[single_traj_mask]
                         [::3, 1], linewidth=2, color=color, marker='*', markersize=2, zorder=4)
                plt.plot(single_traj_future[single_traj_future_mask][::5, 0], single_traj_future[single_traj_future_mask]
                         [::5, 1], linewidth=2, color=color, marker='.', markersize=6, zorder=4)

        # viz_ref_line
        plt.plot(ref_path[:, 0], ref_path[:, 1], 'y', linewidth=2, zorder=4)

        # viz_map
        # lane_polylines = info['lane']
        # crosswalks_p  =  info['crosswalk']
        # speed_bump =     info['speed_bump']
        # driveway   =     info['drive_way']
        # stop_sign  =     info['stop_sign']
        # road_polylines = info['road_polylines']
        # viz center lanes
        for key, polyline in lane_polylines.items():
            map_type = polyline[0, 6]
            if map_type == 1 or map_type == 2 or map_type == 3:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'g', linestyle='solid', linewidth=1)

        # viz roadlines
        for key, polyline in road_polylines.items():
            map_type = polyline[0, 6]
            if map_type == 6:
                plt.plot(polyline[:, 0], polyline[:, 1], 'w',
                         linestyle='dashed', linewidth=1)
            elif map_type == 7:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'w', linestyle='solid', linewidth=1)
            elif map_type == 8:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'w', linestyle='solid', linewidth=1)
            elif map_type == 9:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:yellow', linestyle='dashed', linewidth=1)
            elif map_type == 10:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:yellow', linestyle='dashed', linewidth=1)
            elif map_type == 11:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:yellow', linestyle='solid', linewidth=1)
            elif map_type == 12:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:yellow', linestyle='solid', linewidth=1)
            elif map_type == 13:
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:yellow', linestyle='dotted', linewidth=1)
            elif map_type == 15:
                plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)
            elif map_type == 16:
                plt.plot(polyline[:, 0], polyline[:, 1], 'k', linewidth=1)

        # viz stop sign
        for key, polyline in stop_sign.items():
            map_type = polyline[0, 6]
            if map_type == 17:
                if len(polyline) < 2:
                    plt.gca().add_patch(plt.Circle(
                        polyline[0][:2], 2, color='r'))
                else:
                    for pol in polyline:
                        plt.gca().add_patch(
                            plt.Circle(pol[0][:2], 2, color='r'))

        for key, polyline in crosswalks_p.items():
            map_type = polyline[0, 6]
            if map_type == 18:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1], 'b', linewidth=1)

        for key, polyline in speed_bump.items():
            map_type = polyline[0, 6]
            if map_type == 19:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:orange', linewidth=1)

        for key, polyline in driveway.items():
            map_type = polyline[0, 6]
            if map_type == 20:
                polyline = polygon_completion(polyline).astype(np.float32)
                plt.plot(polyline[:, 0], polyline[:, 1],
                         'xkcd:orange', linewidth=1)

        plt.gca().set_facecolor('xkcd:grey')
        plt.gca().margins(0)
        plt.gca().set_aspect('equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.show()
        plt.close()
