import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import VehicleType
from commonroad.geometry.shape import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad_dc.feasibility.vehicle_dynamics import VehicleParameterMapping
from matplotlib.collections import LineCollection
from PIL import Image

from benchmark.planning import planning


def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def planning_example(cfg: dict):
    # Global benchmarking settings
    method = 'FISS+' # 'FOP', 'FOP\'', 'FISS', 'FISS+'
    num_samples = (5, 5, 5)
    save_gif = True
    
    vehicle_type = VehicleType.VW_VANAGON # FORD_ESCORT, BMW_320i, VW_VANAGON
    vehicle_params = VehicleParameterMapping[vehicle_type.name].value
    
    # Read all files under this directory
    root_dir = os.path.join(os.getcwd(), "data/")
    dataset = "demo/"
    result_dir = "results/"
    file = "DEU_Flensburg-1_1_T-1.xml"
    file_name = os.path.join(root_dir, dataset, file)
    scenario, planning_problem_set = CommonRoadFileReader(file_name).open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    initial_state = planning_problem.initial_state

    ##################################################### Planning #########################################################
    try:
        goal_reached, ego_vehicle_trajectory, processing_time, time_list, planner_stats, fplist = planning(scenario, planning_problem, vehicle_params, method, num_samples)

        if ego_vehicle_trajectory is None:
            print("No ego vehicle trajectory found")
            raise RuntimeError
        
        # The ego vehicle can be visualized by converting it into a DynamicObstacle
        ego_vehicle_shape = Rectangle(length=vehicle_params.l, width=vehicle_params.w)
        ego_vehicle_prediction = TrajectoryPrediction(trajectory=ego_vehicle_trajectory, shape=ego_vehicle_shape)
        ego_vehicle_type = ObstacleType.CAR
        ego_vehicle = DynamicObstacle(obstacle_id=100, obstacle_type=ego_vehicle_type,
                                        obstacle_shape=ego_vehicle_shape, initial_state=initial_state,
                                        prediction=ego_vehicle_prediction)
        
    except RuntimeError:
        print("   ", f"{file} not feasible, proceding to next file!")
        
    ##################################################### Visualization #########################################################
    if save_gif:
        images = []
        mpl.rcParams['font.size'] = 20
        for i in range(len(fplist)):
            plt.figure(figsize=(25, 10))
            
            rnd = MPRenderer()
            rnd.draw_params.time_begin = i
            scenario.draw(rnd)
            rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "g"
            ego_vehicle.draw(rnd)
            planning_problem_set.draw(rnd)
            rnd.render()
            costs = []
            xs = []
            ys = []
            for fp in fplist[i]:
                costs.append(fp.cost_final)
                xs.append(fp.x[1:])
                ys.append(fp.y[1:])
            lc = multiline(xs, ys, costs,ax=rnd.ax, cmap='RdYlGn_r', lw=2, zorder = 20)
            plt.colorbar(lc)

            x_coords = [state.position[0] for state in ego_vehicle_trajectory.state_list]
            y_coords = [state.position[1] for state in ego_vehicle_trajectory.state_list]
            x_coords_p = [state.position[0] for state in ego_vehicle_trajectory.state_list[0:i]]
            y_coords_p = [state.position[1] for state in ego_vehicle_trajectory.state_list[0:i]]
            x_coords_f = [state.position[0] for state in ego_vehicle_trajectory.state_list[i:]]
            y_coords_f = [state.position[1] for state in ego_vehicle_trajectory.state_list[i:]]
            dx_ego_f = np.diff(x_coords_f)
            dy_ego_f = np.diff(y_coords_f)
            rnd.ax.plot(x_coords_p, y_coords_p, color='#9400D3', alpha=1,  zorder = 25, lw = 1)
            rnd.ax.plot(x_coords_f, y_coords_f, color='#AFEEEE', alpha=1,  zorder = 25, lw = 1)
            rnd.ax.quiver(x_coords_f[:-1:5], y_coords_f[:-1:5], dx_ego_f[::5], dy_ego_f[::5], scale_units='xy', angles='xy', scale=1, width = 0.009, color='#AFEEEE', zorder = 26)

            x_min = min(x_coords)-8
            x_max = max(x_coords)+8
            y_min = min(y_coords)-8
            y_max = max(y_coords)+8
            l = max(x_max-x_min, y_max-y_min)
            
            if l == x_max - x_min:
                plt.xlim(x_min, x_max)
                plt.ylim(y_min - (l-(y_max-y_min))/2, y_max + (l-(y_max-y_min))/2)
            else:
                plt.xlim(x_min - (l-(x_max-x_min))/2, x_max + (l-(x_max-x_min))/2)
                plt.ylim(y_min, y_max)
                
            for obs in scenario.dynamic_obstacles:
                t = 0
                obs_traj_x = []
                obs_traj_y = []
                while obs.state_at_time(t) is not None:  
                    obs_traj_x.append(obs.state_at_time(t).position[0])
                    obs_traj_y.append(obs.state_at_time(t).position[1])
                    t += 1
                dx = np.diff(obs_traj_x)
                dy = np.diff(obs_traj_y)
                obs_traj_x=obs_traj_x[:-1]
                obs_traj_y=obs_traj_y[:-1]
                rnd.ax.quiver(obs_traj_x[:i:5], obs_traj_y[:i:5], dx[:i:5], dy[:i:5], scale_units='xy', angles='xy', scale=1, width = 0.006, color='#BA55D3', zorder = 25)
                rnd.ax.quiver(obs_traj_x[i::5], obs_traj_y[i::5], dx[i::5], dy[i::5], scale_units='xy', angles='xy', scale=1, width = 0.006, color='#1d7eea', zorder = 25)
                rnd.ax.plot(obs_traj_x[0:i], obs_traj_y[0:i], color='#BA55D3', alpha=0.8,  zorder = 25, lw = 0.6)
                rnd.ax.plot(obs_traj_x[i:], obs_traj_y[i:], color='#1d7eea', alpha=0.8,  zorder = 25, lw = 0.6)
            time_list.append(0)
            plt.title("{method}: {n} Samples  {time}s".format(method = method, n=len(fplist[i]),time = round(time_list[i], 3)))
            # plt.subplots_adjust(top=0.85, bottom=0.16)
            scenario_id = os.path.splitext(file)[0]
            plt.suptitle(f'Scenario ID: {scenario_id}', fontsize = 20, x = 0.59, y = 0.06)
            # Write the results into a jpg file
            result_path = os.path.join(root_dir, result_dir, 'gif_cache', method, scenario_id)
            if not os.path.exists(result_path):
                os.makedirs(result_path)
                print("Target directory: {} Created".format(result_path))
            file_path = os.path.join(result_path, "{time_step}.jpg".format(time_step=i))
            
            # if not os.path.exists(file_path):
            plt.savefig(file_path, dpi=200, bbox_inches='tight')
            print("Fig saved to:", file_path)
            plt.close()
            # plt.show()
            images.append(Image.open(file_path))

        gif_dirpath = os.path.join(root_dir, result_dir, 'gif/', method)
        if not os.path.exists(gif_dirpath):
            os.makedirs(gif_dirpath)
            print("Target directory: {} Created".format(gif_dirpath))
        gif_filepath = os.path.join(gif_dirpath, f"{scenario_id}.gif")
        images[0].save(gif_filepath, save_all=True, append_images=images[1:], optimize=True, duration=100, loop=0)
        print("Gif saved to:", gif_filepath)

    return


if __name__ == '__main__':
    repo_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg_file', type=str, default=os.path.join(repo_dir, 'cfgs/demo_config.yaml'), help='specify the config file for the demo')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
    
    planning_example(cfg)