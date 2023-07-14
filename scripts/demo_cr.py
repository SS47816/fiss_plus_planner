import argparse
import os

import yaml

from planners.benchmark.planning import planning

if __name__ == '__main__':
    repo_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg_file', type=str, default=os.path.join(repo_dir, 'cfgs/demo_config.yaml'), help='specify the config file for the demo')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
        file.close()
        
    output_dir = os.path.join(os.getcwd(), cfg['OUTPUT_DIR'])
    input_dir = os.path.join(os.getcwd(), cfg['INPUT_DIR'])
    
    if cfg['FILES']:
        # Only run the specified scenario files under the input directory
        for i, file in enumerate(cfg['FILES']):
            planning(cfg, output_dir, input_dir, file)
    else:
        # Read all scenario files under the input directory
        for i, file in enumerate(os.listdir(input_dir)):
            planning(cfg, output_dir, input_dir, file)