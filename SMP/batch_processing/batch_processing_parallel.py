import logging
import warnings
from multiprocessing import Process, Semaphore

import SMP.batch_processing.helper_functions as hf
from SMP.batch_processing.process_scenario import process_scenario


def run_parallel_processing():
    warnings.filterwarnings("ignore")

    configuration, logger, scenario_loader, def_automaton, result_dict = hf.init_processing("Batch Processor")

    num_worker = hf.parse_processes_number(int(configuration['setting']['num_worker_processes']))
    semaphore = Semaphore(num_worker)

    message = f"Number of parallel processes: {num_worker}"
    logger.info(message)
    print(message)

    logger.setLevel(logging.INFO)

    list_processes = []
    for idx, scenario_id in enumerate(result_dict["scenarios_to_process"]):
        """
        Once the maximum number of workers are deployed, the main process
        will be blocked. The loop will continue only after at least one 
        worker has completed its process.
        """
        semaphore.acquire()
        p = Process(target=process_scenario,
                    args=(scenario_id, scenario_loader, configuration, def_automaton, result_dict, semaphore))
        list_processes.append(p)
        p.start()
        result_dict["started_processing"] += 1
        hf.print_current_status(result_dict)

    # wait for all processes to finish
    for p in list_processes:
        p.join()

    hf.log_detailed_scenario_results(logger, result_dict)
    hf.log_statistics(logger, result_dict, verbose=True)


if __name__ == '__main__':
    run_parallel_processing()
