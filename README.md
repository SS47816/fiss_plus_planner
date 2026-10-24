# FISS+ Planner

> [!IMPORTANT]
> **This project now lives in [SS47816/frenet_planner](https://github.com/SS47816/frenet_planner).**

FISS+ (IROS 2023) is maintained in **frenet_planner**, together with FOP, FOP+ and the other FISS variants: one
C++17 core with Python bindings (`pip install frenet-planner`), ROS 2 / ROS 1 nodes, Lanelet2 / CommonRoad /
CARLA support and a reproducible benchmark. Select it with `planner.type: fiss_plus`.

## Paper version

The code used in the paper is kept at the tag
[`paper-iros2023`](https://github.com/SS47816/fiss_plus_planner/tree/paper-iros2023):

```bash
git clone --branch paper-iros2023 https://github.com/SS47816/fiss_plus_planner.git
```

## Citation

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
