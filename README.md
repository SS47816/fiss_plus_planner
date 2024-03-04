# FISS+ Planner

## [IROS 2023] FISS+: Efficient and Focused Trajectory Generation and Refinement using Fast Iterative Search and Sampling Strategy

![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-informational?style=flat&logo=ubuntu&logoColor=white&color=2bbc8a)
![Python](https://img.shields.io/badge/Code-Python-informational?style=flat&logo=Python&logoColor=white&color=2bbc8a)
[![CodeFactor](https://www.codefactor.io/repository/github/ss47816/lgsvl_utils/badge)](https://www.codefactor.io/repository/github/ss47816/fiss_plus_planner)
![GitHub Repo stars](https://img.shields.io/github/stars/ss47816/fiss_plus_planner?color=FFE333)
![GitHub Repo forks](https://img.shields.io/github/forks/ss47816/fiss_plus_planner?color=FFE333)

<p float="left">
  <img src="media/FOP+/DEU_Flensburg-1_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS/DEU_Flensburg-1_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS+/DEU_Flensburg-1_1_T-1.gif" alt="drawing" width="200"/>
</p>

<p float="left">
  <img src="media/FOP+/DEU_Lohmar-54_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS/DEU_Lohmar-54_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS+/DEU_Lohmar-54_1_T-1.gif" alt="drawing" width="200"/>
</p>

<p float="left">
  <img src="media/FOP+/DEU_Lohmar-65_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS/DEU_Lohmar-65_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS+/DEU_Lohmar-65_1_T-1.gif" alt="drawing" width="200"/>
</p>

## Video

<a href="https://youtu.be/ZLGDUYyel30?si=d9aoYWL5ZRjnv4mJ" target="_blank"><img src="media/video_cover.png" alt="video" width="640" height="360" border="10" /></a>

_[Our paper](https://doi.org/10.1109/IROS55552.2023.10341498) has been accepted by IROS 2023._

```bibtex
@INPROCEEDINGS{Sun_fiss+_2023,
  author={Sun, Shuo and Chen, Jie and Sun, Jiawei and Yuan, Chengran and Li, Yuanchen and Zhang, Tangyike and Ang, Marcelo H.},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  title={FISS+: Efficient and Focused Trajectory Generation and Refinement Using Fast Iterative Search and Sampling Strategy},
  year={2023},
  pages={10527-10534},
  doi={10.1109/IROS55552.2023.10341498},
  organization={IEEE}
}
```

## Dependencies

- python>=3.8
- commonroad==2022.3
- cvxpy>=1.0.0

## Install

We recommend using the anaconda environment

```bash
# Clone this repository to a location <PARENT_DIR> (change this part)
cd <PARENT_DIR>
git clone https://github.com/SS47816/fiss_plus_planner
cd fiss_plus_planner
# Create a conda envrionment from the file
conda env create -f environment.yml
conda activate cr
# Export path
export PYTHONPATH=$PYTHONPATH:"<PARENT_DIR>/fiss_plus_planner"
```

## Usage

1. You may modify the configuration file located at `./cfgs/demo_config.yaml`
2. Run the demo script:

   ```bash
   python3 scripts/demo_cr.py
   ```

   The result of each scenario will be save as a separate `.gif` file under the specified ouput directory (`./data/output/gif/`)

## Kown Issues and Fixes

### 1. AttributeError: 'numpy.int64' object has no attribute 'intersects'

As pointed out by @ggosjw, you might encounter an error similar to this:

```bash
  Traceback (most recent call last):
    File "<YOUR_DIR>/fiss_plus_planner/scripts/demo_cr.py", line 28, in
    planning(cfg, output_dir, input_dir, file)
    File "<YOUR_DIR>/fiss_plus_planner/planners/benchmark/planning.py", line 314, in planning
    _, ego_vehicle_trajectory, _, time_list, _, fplist = frenet_optimal_planning(
    File "<YOUR_DIR>/fiss_plus_planner/planners/benchmark/planning.py", line 38, in frenet_optimal_planning
    global_plan = global_planner.plan_global_route(scenario, planning_problem)
    File "<YOUR_DIR>/fiss_plus_planner/planners/commonroad_interface/global_planner.py", line 38, in plan_global_route
    route_planner = RoutePlanner(scenario, planning_problem, backend=RoutePlanner.Backend.NETWORKX_REVERSED)
    File "<YOUR_DIR>/anaconda3/envs/cr/lib/python3.10/site-packages/commonroad_route_planner/route_planner.py", line 138, in init
    self.id_lanelets_start = self._retrieve_ids_lanelets_start()
    File "<YOUR_DIR>/anaconda3/envs/cr/lib/python3.10/site-packages/commonroad_route_planner/route_planner.py", line 192, in _retrieve_ids_lanelets_start
    list_ids_lanelets_start = self.lanelet_network.find_lanelet_by_position([post_start])[0]
    File "<YOUR_DIR>/anaconda3/envs/cr/lib/python3.10/site-packages/commonroad/scenario/lanelet.py", line 1549, in find_lanelet_by_position
    return [[self._get_lanelet_id_by_shapely_polygon(lanelet_shapely_polygon) for lanelet_shapely_polygon in
    File "<YOUR_DIR>/anaconda3/envs/cr/lib/python3.10/site-packages/commonroad/scenario/lanelet.py", line 1549, in
    return [[self._get_lanelet_id_by_shapely_polygon(lanelet_shapely_polygon) for lanelet_shapely_polygon in
    File "<YOUR_DIR>/anaconda3/envs/cr/lib/python3.10/site-packages/commonroad/scenario/lanelet.py", line 1551, in
    lanelet_shapely_polygon.intersects(point) or lanelet_shapely_polygon.buffer(1e-15).intersects(point)]
    AttributeError: 'numpy.int64' object has no attribute 'intersects'
```

which might be caused by a newer version of the `commonroad` pkg.

### Fix

Find function `find_lanelet_by_position()` in `commonroad/scenario/lanelet.py`, and change it to:

```python
  def find_lanelet_by_position(self, point_list: List[np.ndarray]) -> List[List[int]]:
    """
    Finds the lanelet id of a given position

    :param point_list: The list of positions to check
    :return: A list of lanelet ids. If the position could not be matched to a lanelet, an empty list is returned
    """
    assert isinstance(point_list,
                      ValidTypes.LISTS), '<Lanelet/contains_points>: provided list of points is not a list! type ' \
                                        '= {}'.format(type(point_list))

    return [[self._get_lanelet_id_by_shapely_polygon(lanelet_shapely_polygon) for lanelet_shapely_polygon in
            self._buffered_polygons.values() if
            lanelet_shapely_polygon.intersects(point) or lanelet_shapely_polygon.buffer(1e-15).intersects(point)]
            for point in [ShapelyPoint(point) for point in point_list]]
```

## Contribution

You are welcome contributing to the package by opening a pull-request

We are following: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s2.2-imports)

## License

Licensed under [Apache License 2.0](https://github.com/SS47816/fiss_plus_planner/blob/main/LICENSE)
