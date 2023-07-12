import warnings

import SMP.batch_processing.helper_functions as hf
from SMP.batch_processing.process_scenario import debug_scenario
import SMP.batch_processing.timeout_config

def run_sequential_processing():
    
    SMP.batch_processing.timeout_config.use_sequential_processing = True
    
    warnings.filterwarnings("ignore")

    configuration, logger, scenario_loader, def_automaton, result_dict = hf.init_processing(
        "Sequential Batch Processor", for_multi_processing=False)

    for idx, scenario_id in enumerate(scenario_loader.scenario_ids):
        debug_scenario(scenario_id=scenario_id, scenario_loader=scenario_loader, configuration_dict=configuration,
                       def_automaton=def_automaton, result_dict=result_dict, logger=logger)
        result_dict["started_processing"] += 1
        hf.print_current_status(result_dict)

    hf.log_detailed_scenario_results(logger, result_dict)
    hf.log_statistics(logger, result_dict, verbose=True)


if __name__ == '__main__':
    run_sequential_processing()
