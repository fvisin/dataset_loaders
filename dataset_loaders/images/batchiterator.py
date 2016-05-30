try:
    import Queue
except ImportError:
    import queue as Queue
import threading
import numpy as np
import time


class BatchIterator(object):
    """
     Cyclic Iterators over batches.
    """

    def __init__(self, minibatch_size, data_list, testing=False):

        self.n_samples = len(data_list)
        self.minibatch_size = minibatch_size
        self.n_batches = self.n_samples / self.minibatch_size
        self.testing = testing

        if not self.testing:
            self.createindices = lambda: np.random.permutation(self.n_samples)
        else:  # testing == true
            assert self.n_samples % self.minibatch_size == 0, """for testing n must be
            multiple of batch size"""
            self.createindices = lambda: range(self.n_samples)

        self.data_list = data_list

        self.perm = self.createindices()
        assert self.n_samples > self.minibatch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _get_permuted_batches(self, n_batches):
        # Return a list of permuted batch indeces
        batches = []
        for i in range(n_batches):
            # Extend random permuation if shorter than minibatch_size
            if len(self.perm) <= self.minibatch_size:
                new_perm = self.createindices()
                self.perm = np.hstack([self.perm, new_perm])

            batches.append(self.perm[:self.minibatch_size])
            self.perm = self.perm[self.minibatch_size:]
        return batches

    def get_n_batches(self):
        return self.n_batches

    def fetch_from_dataset(self, batch_to_load):
        raise NotImplementedError

    def next(self):
        # Extract a single batch of data
        batch = self._get_permuted_batches(1)[0]
        data_to_load = [self.data_list[i] for i in batch]

        # Load data
        batch_of_data = self.fetch_from_dataset(data_to_load)

        return batch_of_data

    def get_minibatch_size(self):
        return self.minibatch_size


def threaded_generator(generator, num_cached=50, wait_time=0.05):
    queue = Queue.Queue()
    sentinel = object()  # guaranteed unique reference
    _stop = threading.Event()

    # Define producer (putting items into queue)
    def producer():
        # for item in generator:
        while not _stop.is_set():
            try:
                if queue.qsize() < num_cached:
                    try:
                        item = next(generator)
                    except ValueError:
                        continue
                    queue.put(item)
                else:
                    time.sleep(wait_time)
            except Exception:
                _stop.set()
                # raise
        queue.put(sentinel)

    # Start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # Run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()
