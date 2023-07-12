# fiss_plus_planner

## Dependencies
* python>=3.8
* commonroad==2022.3
* cvxpy>=1.0.0

## Install
We recommend using anaconda environment
```bash
git clone https://github.com/SS47816/irl_planner
cd irl_planner

conda env create -f environment.yml
conda activate cr
```

### CommonRoad Issue Fix

In `~/anaconda3/envs/<ENV_NAME>/python3.10/site-packages/commonraod/scenario/lanelet.py`, 
at line 1550, part of the return statement of function `find_lanelet_by_position()`:

change the `self._strtee.query(point)` to `self._buffered_polygons.values()`, as following:

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

## Usage

```bash

```

## Contribution
You are welcome contributing to the package by opening a pull-request

We are following: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s2.2-imports)

## License
Licensed under [Apache License 2.0](https://github.com/SS47816/fiss_plus_planner/blob/main/LICENSE)