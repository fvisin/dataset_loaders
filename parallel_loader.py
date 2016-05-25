from itertools import izip_longest
try:
    import Queue
except ImportError:
    import queue as Queue
from threading import Condition, Thread
from time import sleep

import numpy as np
from numpy.random import shuffle


def fetch_data(names_queue, out_queue, _reset_lock, one_hot_fetch):
    """
    Fill the out_queue.

    Whenever there are names in the names queue, it will read them,
    fetch the corresponding data and fill the out_queue.
    """
    while True:
        try:
            with _reset_lock:
                _reset_lock.wait()
                # Grabs names from queue
                batch_to_load = names_queue.get(False)

                # Load the data
                minibatch_data = one_hot_fetch(batch_to_load)
                if isinstance(minibatch_data, np.ndarray):
                    gt = minibatch_data[1].flatten()
                    gt_out = np.zeros((gt.shape[0], 63), dtype='int32')
                    gt_out[range(gt.shape[0]), gt] = 1
                    minibatch_data[1] = gt_out

                # Place it in out_queue
                out_queue.put(minibatch_data)

                # Signal to the names queue that the job is done
                names_queue.task_done()

        except IOError as e:
            print("Image in image_group corrupted!")
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except Queue.Empty:
            # print 'Fetch: wait'
            # _reset_lock.wait()
            pass
        except:
            raise
            break
        finally:
            # print 'Fetch: release'
            # _reset_lock.release()
            pass


class ThreadedDataset(object):
    """
    Threaded dataset.

    This is an abstract class and should not be used as is. Each
    specific dataset class should implement its `get_names` and
    `fetch_from_dataset` functions to load the list of filenames to be
    loaded and define how to load the data from the dataset,
    respectively.
    """
    def __init__(self,
                 is_one_hot,
                 nclasses,
                 minibatch_size=1,
                 queues_size=5,
                 use_threads=False,
                 convert_to_one_hot=True,
                 shuffle_at_each_epoch=True,
                 infinite_iterator=True,
                 data_dim_ordering='tf',
                 get_dim_ordering='tf'):

        self.is_one_hot = is_one_hot
        self.nclasses = nclasses
        self.minibatch_size = minibatch_size
        self.queues_size = queues_size
        self.use_threads = use_threads
        self.convert_to_one_hot = convert_to_one_hot
        self.shuffle_at_each_epoch = shuffle_at_each_epoch
        self.infinite_iterator = infinite_iterator
        self.data_dim_ordering = data_dim_ordering
        self.get_dim_ordering = get_dim_ordering

        self.names = self.get_names()

        # initialize
        if self.use_threads:
            self._reset_lock = Condition()
            self.names_queue = Queue.Queue(maxsize=self.queues_size)
            self.out_queue = Queue.Queue(maxsize=self.queues_size)

            # Start the data fetcher thread
            data_fetcher = Thread(
                target=fetch_data,
                args=(self.names_queue, self.out_queue, self._reset_lock,
                      self.one_hot_fetch))
            data_fetcher.setDaemon(True)
            data_fetcher.start()
            self.data_fetcher = data_fetcher
        self.reset(self.shuffle_at_each_epoch)

        # Give time to the data fetcher to die, in case of errors
        sleep(1)

    def get_names(self):
        raise NotImplementedError

    def fetch_from_dataset(self, batch_to_load):
        raise NotImplementedError

    def one_hot_fetch(self, batch_to_load):
        x, y = self.fetch_from_dataset(batch_to_load)
        if self.data_dim_ordering != self.get_dim_ordering:
            if self.get_dim_ordering == 'th':
                # b,s,0,1,c --> b,s,c,0,1
                x = x.transpose([0, 1, 4, 2, 3])
            elif self.get_dim_ordering == 'tf':
                # b,s,c,0,1 --> b,s,0,1,c
                x = x.transpose([0, 1, 3, 4, 2])

        if not self.is_one_hot and self.convert_to_one_hot:
            nc = self.nclasses
            sh = y.shape
            y = y.flatten()
            y_hot = np.zeros((y.shape[0], nc), dtype='int32')
            y = y.astype('int32')
            y_hot[range(y.shape[0]), y] = 1
            y_hot = y_hot.reshape(sh + (nc,))
            if self.get_dim_ordering == 'th':
                # b,s,0,1,c --> b,s,c,0,1
                y_hot = y_hot.transpose([0, 1, 4, 2, 3])
        else:
            y_hot = y
        return x, y_hot

    def reset(self, shuffle_at_each_epoch):
        # Shuffle data
        if shuffle_at_each_epoch:
            shuffle(self.names)

        # Group data into minibatches
        name_batches = [el for el in izip_longest(
            fillvalue=None, *[iter(self.names)] * self.minibatch_size)]
        self.epoch_length = len(name_batches)
        self.name_batches = iter(name_batches)

        # Reset the queues
        if self.use_threads:
            with self._reset_lock:
                # Empty the queues
                queues = self.names_queue, self.out_queue
                for q in queues:
                    with q.mutex:
                        q.queue.clear()
                        q.all_tasks_done.notify_all()
                        q.unfinished_tasks = 0

                # Refill the names queue and release the data fetcher lock
                self._init_names_queue()
                self._reset_lock.notifyAll()
            # Wait for the out queue to be filled
            while self.out_queue.empty():
                sleep(0.1)

    def _init_names_queue(self):
        for _ in range(self.queues_size):
            name_batch = self.name_batches.next()
            self.names_queue.put(name_batch)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def _step(self):
        if self.use_threads:
            # Kill main process if fetcher died
            if not self.data_fetcher.isAlive():
                import sys
                sys.exit(0)
            try:
                # Get one minibatch from the out queue
                data_batch = self.out_queue.get(False)
                # Refill the names queue, if any left
                try:
                    name_batch = self.name_batches.next()
                    self.names_queue.put(name_batch)
                except StopIteration:
                    pass
                self.out_queue.task_done()
            except Queue.Empty:
                # No more minibatches in the out queue
                self.reset(self.shuffle_at_each_epoch)
                if not self.infinite_iterator:
                    raise StopIteration
                else:
                    data_batch = self._step()
        else:
            try:
                name_batch = self.name_batches.next()
                data_batch = self.one_hot_fetch(name_batch)
            except StopIteration:
                self.reset(self.shuffle_at_each_epoch)
                if not self.infinite_iterator:
                    raise
                else:
                    data_batch = self._step()

        if data_batch is None:
            pass
        return data_batch

    def finish(self):
        self.data_fetcher.join()
