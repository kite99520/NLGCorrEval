from multiprocessing import Queue, pool, Process
from abc import ABC, abstractmethod

class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(pool.Pool):

    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess

        return proc

class Processer(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def merge(self, rhs):
        pass

    @abstractmethod
    def count(self, inputs, worker_id):
        pass

    @abstractmethod
    def out(self):
        pass

def worker_func(Counter, worker_id_queue, in_queue, out_queue):
    # Counter, worker_id, in_queue, out_queue = args
    worker_id = worker_id_queue.get()
    counter = Counter()
    while True:
        inputs = in_queue.get()
        if inputs is None:
            break
        counter.count(inputs, worker_id)
    out_queue.put(counter)

def pcount(inputs, Counter : Processer, workers, *, worker_id_list = None):
    """
    abcd
    """
    in_queue, out_queue = Queue(workers + 1), Queue()
    worker_id_queue = Queue()
    if worker_id_list is None:
        worker_id_list = list(range(workers))
    for i in worker_id_list:
        worker_id_queue.put(i)
    pool = NoDaemonProcessPool(workers, worker_func, (Counter, worker_id_queue, in_queue, out_queue))

    step = (len(inputs) + workers - 1) // workers
    assert workers * step >= len(inputs)
    for i in range(workers):
        in_queue.put(inputs[i * step: (i+1) * step])
    for i in range(workers):
        in_queue.put(None)
    
    counter = out_queue.get()
    for _ in range(workers - 1):
        counter.merge(out_queue.get())
    pool.terminate()
    return counter.out()
