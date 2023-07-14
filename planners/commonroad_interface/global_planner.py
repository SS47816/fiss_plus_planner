import numpy as np

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
from commonroad.scenario.traffic_sign_interpreter import TrafficSigInterpreter

class GlobalPlan(object):
    def __init__(self):
        self.lanelets = None
        self.lanelet_centerlines = None
        self.concat_centerline = None
        self.speed_limits = None
        self.required_speeds = None
        # self.widths = None

class GlobalPlanner(object):
    
    #  NETWORKX: uses built-in functions from the networkx package, tends to change lane earlier
    #  PRIORITY_QUEUE: uses A-star search to find routes, lane change maneuver depends on the heuristic cost
    def plan_global_route(self, scenario: Scenario, planning_problem, method: str = 'NETWORKX_REVERSED', plan_all_routes: bool = False, view_route: bool = False):
        ''' Plan the global route for a given scenario and problem
        
        Parameters
        ----------
        
        `scenerio` (`commonroad.scenario.Scenerio`): the CommonRoad scenerio
        `planning_problem` (`commonroad.planning_problem.PlanningProblemSet`): the CommonRoad planning problem
        `method` (`string`): the method used by the planner. Options: `'NETWORKX_REVERSED'`, `'NETWORKX'`, `'PRIORITY_QUEUE'`
        `plan_all_routes` (`bool`): if all alternative routes are returned as well, default: `False`

        Returns
        -------
        (`commonroad_interface.global_planner.GlobalPlan`): planned global route information
        '''
        if method == 'NETWORKX_REVERSED':
            route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
        if method == 'NETWORKX':
            route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX)
        if method == 'PRIORITY_QUEUE':
            route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.PRIORITY_QUEUE)

        # plan routes, and save the routes in a route candidate holder
        candidate_holder = route_planner.plan_routes()
        
        # option 1: retrieve all routes
        if plan_all_routes:
            list_routes, num_route_candidates = candidate_holder.retrieve_all_routes()
            # print(f"Number of route candidates: {num_route_candidates}")
            # here we retrieve the first route in the list, this is equivalent to: route = list_routes[0]
            route = candidate_holder.retrieve_first_route()
        # option 2: retrieve the best route by orientation metric
        else:
            route = candidate_holder.retrieve_best_route_by_orientation()
        
        # Assemble the GlobalPlan
        global_plan = GlobalPlan()
        
        # generate the lanelet network
        llnet = scenario.lanelet_network
        laneletlist = route.list_ids_lanelets
        
        # traffic_sign_interpreter = TrafficSigInterpreter(SupportedTrafficSignCountry.GERMANY, scenario.lanelet_network)
        # self.speed_limits = traffic_sign_interpreter.speed_limit(route.list_ids_lanelets)
        # self.required_speeds = traffic_sign_interpreter.required_speed(route.list_ids_lanelets)
        
        if len(llnet.find_lanelet_by_id(laneletlist[len(laneletlist)-1]).successor) != 0:
            lastlanelet = llnet.find_lanelet_by_id(laneletlist[len(laneletlist)-1]).successor[0]
        else:
            lastlanelet = None
        global_plan.lanelets = [llnet.find_lanelet_by_id(laneletlist[i]) for i in range(len(laneletlist))]
        if lastlanelet is not None:
            lastlanelet = llnet.find_lanelet_by_id(lastlanelet)
            global_plan.lanelets.append(lastlanelet)
        global_plan.lanelet_centerlines = np.array([lanelet.center_vertices for lanelet in global_plan.lanelets],dtype=object)
        
        # Concatenate the centerlines into one np.ndarray, and remove duplicates
        concat_centerline = np.concatenate(global_plan.lanelet_centerlines)
        concat_centerline = np.array(concat_centerline,dtype=float)
        _, unqiue_indices = np.unique(concat_centerline, return_index=True, axis=0)
        concat_centerline = concat_centerline[np.sort(unqiue_indices)]
        
        # Calculate the orientation of each centerline point
        diff_y = np.diff(concat_centerline[:, 1].flatten())
        diff_x = np.diff(concat_centerline[:, 0].flatten())
        yaws = np.arctan2(diff_y, diff_x)
        yaws = np.append(yaws, [yaws[-1]], axis=0)
        
        # Calcuate the width of each centerline point, and remove duplicates
        widths = np.concatenate(
            [np.array([np.linalg.norm(lanelet.left_vertices[i] - lanelet.right_vertices[i]) for i in range(len(lanelet.left_vertices))])
            for lanelet in global_plan.lanelets]
            )[np.sort(unqiue_indices)]

        global_plan.concat_centerline = np.hstack((concat_centerline, yaws[:, np.newaxis], widths[:, np.newaxis]))
        
        # Visualization
        if view_route:
            print('Global Planning Results:')
            print('Passing through:', global_plan.lanelet_centerlines.shape[0], 'lanelets')
            print('Contains:', global_plan.concat_centerline.shape, 'lane points')
            print(global_plan.concat_centerline)
            visualize_route(route, draw_route_lanelets=True, draw_reference_path=True, size_x=6)
        
        return global_plan
