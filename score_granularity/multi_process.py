import os
import gc
import sys
import signal
from typing import Iterable, Callable, Optional, List, Any
import traceback

import multiprocessing as mp

def worker_func(process_func: Callable[[List[Any], Any], List[Any]], 
                inputs: List[Any], 
                worker_id):
    try:
        return process_func(inputs, worker_id)
    except Exception as e:
        print(f"Error in child process: {e}", file=sys.stderr)
        print("Traceback info:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        os._exit(1)

def terminate_children(*args, **kwargs):
    for p in mp.active_children():
        p.terminate()
    os._exit(1)

def multi_process(inputs: Iterable[Any], 
                  process_func: Callable[[List[Any], Any], List[Any]], 
                  num_workers: int, 
                  worker_ids: Optional[List[Any]] = None):
    """
    A function to process inputs using multiple worker processes.

    Parameters:
    - inputs (iterable): An iterable of inputs to be processed.
    - process_func (function): A processing function that takes an input and a worker_id as arguments, and return a list. 
    - num_workers (int): The number of worker processes to spawn.
    - worker_ids (list, optional): A list of IDs to be assigned to each worker process. If not provided, worker IDs will default to their index in the worker pool.

    Returns:
    - List[Any]: A list of results from processing the inputs.

    Description:
    This function distributes the inputs across multiple worker processes. Each worker process applies
    the given processing function to its assigned inputs. If `worker_ids` is provided, each worker is
    assigned an ID from this list; otherwise, the worker ID is its index in the worker pool.

    Example:
    ```python
    def square(input_data, worker_id):
        return [x * x for x in input_data]

    inputs = range(10)
    num_workers = 4
    worker_ids = ['A', 'B', 'C', 'D']
    
    out = multi_process(inputs, square, num_workers, worker_ids=worker_ids)
    ```
    """

    try:
        inputs = list(inputs)
        if worker_ids is None:
            worker_ids = list(range(num_workers))
        if num_workers == 1:
            return worker_func(process_func, inputs, worker_ids[0])

        assert isinstance(num_workers, int) and num_workers >= 2, "Number of workers must be postive integer"
        signal.signal(signal.SIGTERM, terminate_children)
        
        with mp.Pool(num_workers - 1) as pool:
            _step = len(inputs) // num_workers
            remainder = len(inputs) % num_workers
            step = lambda i: _step + (i < remainder)
            results = []
            start = step(0)
            for i in range(1, num_workers):
                results.append(pool.apply_async(worker_func, (process_func, inputs[start: start + step(i)], worker_ids[i])))
                start += step(i)
            
            inputs = inputs[:step(0)]
            gc.collect()
            out = worker_func(process_func, inputs, worker_ids[0])
            for result in results:
                out += result.get()

        return out
    except Exception as e:
        print(f"Error in father process: {e}", file=sys.stderr)
        print("Traceback info:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        terminate_children()

