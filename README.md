# FISS+ Planner

## [IROS 2023] FISS+: Efficient and Focused Trajectory Generation and Refinement using Fast Iterative Search and Sampling Strategy



## Dependencies

* python>=3.8
* commonroad==2022.3
* cvxpy>=1.0.0

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

## Contribution

You are welcome contributing to the package by opening a pull-request

We are following: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s2.2-imports)

## License

Licensed under [Apache License 2.0](https://github.com/SS47816/fiss_plus_planner/blob/main/LICENSE)