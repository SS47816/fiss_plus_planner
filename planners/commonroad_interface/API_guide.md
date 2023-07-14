# User  API

## Collision Checking
* **Obstacle**
* class obstacle(**id = None, type = None, timestamp = None, polygon = None, pos = None, yaw = None, prediction = None**)
Class representing a dynamic polygon obstacle

* cr2irl(**obs, t**)
    Parameters
        obs(DynamicObstacle) - the dynamic obstacle of CommonRoad
        t(int) - time stamp of the dynamic obstacle

    Return Type
        obstacle

* **SATCollisionChecker**
* class SATCollisionChecker()
Class representing a seperate axis theorem collision checker

* check_collision(**ego: obstacle, obstacle: obstacle**)
    Parameters
        ego(obstacle) - the ego vehicle represented by a *obstacle*
        obstacle(obstacle) - the obstacle represented by a *obstacle*

    Return Type
        Bool



## Global Planner
* **GlobalPlanner**
* class GlobalPlanner(**plan_all_routes=False**)
Class representing a global planner

* plan_global_route(**scenario, planning_problem, method = 'NETWORKX_REVERSED'**)
```python
    Parameters
        scenerio(scenerio) - the CommonRoad scenerio
        planning_problem(planning_problem) - the CommonRoad planning problem
        method(string) - the method used by the planner. Optional: 'NETWORKX', 'PRIORITY_QUEUE'

    Return Type
        Union[List(Array(Array)), List(List), List(Int)]

    Returns
        The vertices of the center lines of the planned lanelets in sequence, in two formats, and the width of each lanelet
```