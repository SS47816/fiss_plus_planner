# FISS+ Planner

## [IROS 2023] FISS+: Efficient and Focused Trajectory Generation and Refinement using Fast Iterative Search and Sampling Strategy

<p float="left">
  <img src="media/FOP'/DEU_Flensburg-78_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS/DEU_Flensburg-78_1_T-1.gif" alt="drawing" width="200"/>
  <img src="media/FISS+/DEU_Flensburg-78_1_T-1.gif" alt="drawing" width="200"/>
</p>

## Dependencies

* python>=3.8
* commonroad==2022.3
* cvxpy>=1.0.0

## Install

We recommend using the anaconda environment
```bash
git clone https://github.com/SS47816/fiss_plus_planner
cd fiss_plus_planner

conda env create -f environment.yml
conda activate cr
```

## Usage

```bash
python3 scripts/demo_cr.py
```

## Contribution

You are welcome contributing to the package by opening a pull-request

We are following: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s2.2-imports)

## License

Licensed under [Apache License 2.0](https://github.com/SS47816/fiss_plus_planner/blob/main/LICENSE)