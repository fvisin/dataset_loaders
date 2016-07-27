from itertools import izip_longest
try:
    import Queue
except ImportError:
    import queue as Queue
import os
import shutil
from threading import Thread
from time import sleep

import numpy as np
from numpy.random import RandomState


def threaded_fetch(names_queue, out_queue, sentinel, fetch_from_dataset):
    """
    Fill the out_queue.

    Whenever there are names in the names queue, it will read them,
    fetch the corresponding data and fill the out_queue.
    """
    while True:
        try:
            # Grabs names from queue
            batch_to_load = names_queue.get()

            if batch_to_load is sentinel:
                names_queue.task_done()
                break

            # Load the data
            minibatch_data = fetch_from_dataset(batch_to_load)

            # Place it in out_queue
            out_queue.put(minibatch_data)

            # Signal to the names queue that the job is done
            names_queue.task_done()
        except IOError as e:
            print("Image in image_group corrupted!")
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except:
            # If any uncaught exception, die
            raise
            break


class ThreadedDataset(object):
    """
    Threaded dataset.

    This is an abstract class and should not be used as is. Each
    specific dataset class should implement its `get_names` and
    `load_sequence` functions to load the list of filenames to be
    loaded and define how to load the data from the dataset,
    respectively.
    """
    def __init__(self,
                 seq_per_video=0,  # if 0 all frames
                 seq_length=0,  # if 0, return 4D
                 crop_size=None,
                 batch_size=1,
                 queues_size=50,
                 get_one_hot=False,
                 get_01c=False,
                 use_threads=False,
                 nthreads=1,
                 shuffle_at_each_epoch=True,
                 infinite_iterator=True,
                 overlap=None,
                 remove_mean=False,
                 divide_by_std=False,
                 rng=RandomState(0xbeef),
                 wait_time=0.05,
                 **kwargs):

        if len(kwargs):
            print('Ignored arguments: {}'.format(kwargs.keys()))

        if nthreads > 1 and not shuffle_at_each_epoch:
            raise NotImplementedError('Multiple threads are not order '
                                      'preserving')

        # for attr in ['name', 'nclasses', '_is_one_hot', '_is_01c',
        #             'debug_shape', 'path', 'sharedpath']:
        #    assert attr in dir(self), (
        #        '{} did not set the mandatory attribute {}'.format(
        #            self.__class__.name, attr))
        ds = getattr(self.__class__, 'data_shape', (None, None, 3))
        self.data_shape = ds[2], ds[0], ds[1]

        self.has_GT = getattr(self, 'has_GT', True)

        # Copy the data to the local path if not existing
        if not os.path.exists(self.path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(self.path))
            shutil.copytree(self.sharedpath, self.path)
            print('Done.')

        self.overlap = overlap if overlap is not None else seq_length - 1
        self.seq_per_video = seq_per_video
        self.return_sequence = seq_length != 0
        self.seq_length = seq_length if seq_length else 1
        if crop_size and tuple(crop_size) == (0, 0):
            crop_size = None
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.queues_size = queues_size
        self.get_one_hot = get_one_hot
        self.get_01c = get_01c
        self.use_threads = use_threads
        self.nthreads = nthreads
        self.shuffle_at_each_epoch = shuffle_at_each_epoch
        self.infinite_iterator = infinite_iterator
        self.rng = rng
        self.sentinel = object()  # guaranteed unique reference
        self.wait_time = wait_time
        self.data_fetchers = []
        self.divide_by_std=divide_by_std
        self.remove_mean=remove_mean

        # self.names_list = self.get_names()
        # if len(self.names_list) == 0:
        #     raise RuntimeError('The name list cannot be empty')

        # initialize
        if self.use_threads:
            self.names_queue = Queue.Queue(maxsize=self.queues_size)
            self.out_queue = Queue.Queue(maxsize=self.queues_size)

        self.reset(self.shuffle_at_each_epoch)

        if len(self.names_list) == 0:
            raise RuntimeError('The name list cannot be empty')

        # Give time to the data fetcher to die, in case of errors
        sleep(1)

    def reset(self, shuffle_at_each_epoch):
        # Reload the names list, so a different subset is selected when
        # not all the possible sequences are picked
        self.names_list = self.get_names()
        # Shuffle data
        if shuffle_at_each_epoch:
            self.rng.shuffle(self.names_list)

        # Group names into minibatches
        name_batches = [el for el in izip_longest(
            fillvalue=None, *[iter(self.names_list)] * self.batch_size)]
        self.nsamples = len(self.names_list)
        self.nbatches = len(name_batches)
        self.epoch_length = len(name_batches)
        self.name_batches = iter(name_batches)

        # Reset the queues
        if self.use_threads:
            # Empty the queues and kill the threads
            queues = self.names_queue, self.out_queue
            for q in queues:
                with q.mutex:
                    q.queue.clear()
                    q.all_tasks_done.notify_all()
                    q.unfinished_tasks = 0
            for _ in self.data_fetchers:
                self.names_queue.put(self.sentinel)
            while any([df.isAlive() for df in self.data_fetchers]):
                sleep(self.wait_time)

            # Refill the names queue
            self._init_names_queue()

            # Start the data fetcher threads
            self.data_fetchers = []
            for _ in range(self.nthreads):
                data_fetcher = Thread(
                    target=threaded_fetch,
                    args=(self.names_queue, self.out_queue, self.sentinel,
                          self.fetch_from_dataset))
                data_fetcher.setDaemon(True)
                data_fetcher.start()
                self.data_fetchers.append(data_fetcher)

            # # Wait for the out queue to be filled
            # while self.out_queue.empty():
            #     sleep(0.1)

    def _init_names_queue(self):
        for _ in range(self.queues_size):
            try:
                name_batch = self.name_batches.next()
                self.names_queue.put(name_batch)
            except StopIteration:
                # Queue is bigger than the tot number of batches
                break

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        return self._step()

    def _step(self):
        done = False
        while not done:
            if self.use_threads:
                # Kill main process if fetcher died
                if all([not df.isAlive() for df in self.data_fetchers]):
                    import sys
                    sys.exit(0)
                try:
                    # Get one minibatch from the out queue
                    data_batch = self.out_queue.get(False)
                    self.out_queue.task_done()
                    done = True
                    # Refill the names queue, if any left
                    try:
                        name_batch = self.name_batches.next()
                        self.names_queue.put(name_batch)
                    except StopIteration:
                        pass
                except Queue.Empty:
                    if self.names_queue.unfinished_tasks:
                        sleep(self.wait_time)
                    else:
                        # No more minibatches in the out queue
                        self.reset(self.shuffle_at_each_epoch)
                        if not self.infinite_iterator:
                            raise StopIteration
                        # else, it will cycle again in the while loop
            else:
                try:
                    name_batch = self.name_batches.next()
                    data_batch = self.fetch_from_dataset(name_batch)
                    done = True
                except StopIteration:
                    self.reset(self.shuffle_at_each_epoch)
                    if not self.infinite_iterator:
                        raise
                    # else, loop to the next image
                except IOError as e:
                    print "{0} I/O error({1}): {2}".format(name_batch, e.errno,
                                                           e.strerror)

        if data_batch is None:
            raise RuntimeError("WTF")
        return data_batch

    def get_names(self):
        raise NotImplementedError

    def load_sequence(self, first_frame):
        """ Loads ONE 4D batch or 5D sequence.

        Should return a *list* of 2+ elements. The first two have to
        be input and labels. If no label is available, return None
        """
        raise NotImplementedError

    def fetch_from_dataset(self, batch_to_load):
        """
        Return *batches* of 5D sequences/clips or 4D images.

        `batch_to_load` contains the indices of the first frame/image of
        each element of the batch.
        `load_sequence` should return a numpy array of 2 or more
        elements, the first of which 4-dimensional (frame, 0, 1, c)
        or (frame, c, 0, 1) containing the data and the second 3D or 4D
        containing the label.
        """
        X = []
        Y = []
        Other = []

        # Create batches
        for el in batch_to_load:

            if el is None:
                continue

            # Load sequence, format is (s, 0, 1, c)
            ret = self.load_sequence(el)
            seq_x, seq_y = ret[0:2]
            # assert seq_x.max() <= 1
            if seq_x.ndim == 3:
                seq_x = seq_x[np.newaxis, ...]
                seq_y = seq_y[np.newaxis, ...]
            assert seq_x.ndim == 4

            # Crop
            crop = list(self.crop_size) if self.crop_size else None
            if crop:
                height, width = seq_x.shape[-3:-1]

                if crop[0] < height:
                    top = self.rng.randint(height - crop[0])
                else:
                    print('Dataset loader: Crop height greater or equal to '
                          'image size')
                    top = 0
                    crop[0] = height
                if crop[1] < width:
                    left = self.rng.randint(width - crop[1])
                else:
                    print('Dataset loader: Crop width greater or equal to '
                          'image size')
                    left = 0
                    crop[1] = width

                seq_x = seq_x[..., top:top+crop[0], left:left+crop[1], :]
                if self.has_GT:
                    seq_y = seq_y[..., top:top+crop[0], left:left+crop[1]]

            if self.has_GT:
                # Map all void classes to nclasses and shift the other values
                # accordingly, so that the valid values are between 0 and
                # nclasses-1 and the void_classes are all equal to nclasses.
                void_l = self._void_labels
                void_l.sort(reverse=True)
                mapping = {}
                delta = 0
                # Prepare the mapping
                for i in range(self.nclasses + len(self._void_labels)):
                    if i in self._void_labels:
                        mapping[i] = self.nclasses
                        delta += 1
                    else:
                        mapping[i] = i - delta
                # Apply the mapping
                seq_y[seq_y == self.nclasses] = -1
                for i in range(self.nclasses + len(self._void_labels)):
                    if i == self.nclasses:
                        continue
                    seq_y[seq_y == i] = mapping[i]
                seq_y[seq_y == -1] = mapping[self.nclasses]

            # Transform targets seq_y to one hot code if get_one_hot
            # is True
            if self.has_GT and self.get_one_hot:
                nc = (self.nclasses if self._void_labels == [] else
                      self.nclasses + 1)
                sh = seq_y.shape
                seq_y = seq_y.flatten()
                seq_y_hot = np.zeros((seq_y.shape[0], nc),
                                     dtype='int32')
                seq_y = seq_y.astype('int32')
                seq_y_hot[range(seq_y.shape[0]), seq_y] = 1
                seq_y_hot = seq_y_hot.reshape(sh + (nc,))
                seq_y = seq_y_hot

            # Dimshuffle if get_01c is False
            if not self.get_01c:
                # s,0,1,c --> s,c,0,1
                seq_x = seq_x.transpose([0, 3, 1, 2])
                if self.has_GT and self.get_one_hot:
                    seq_y = seq_y.transpose([0, 3, 1, 2])

            # Return 4D images
            if not self.return_sequence:
                seq_x = seq_x[0, ...]
                if self.has_GT:
                    seq_y = seq_y[0, ...]

            # Append stuff to minibatch list
            X.append(seq_x)
            Y.append(seq_y)
            Other.append(ret[2:])

        if len(Other[0]) == 0:
            return np.array(X), np.array(Y)
        else:
            return np.array(X), np.array(Y), np.array(Other)

    def finish(self):
        for data_fetcher in self.data_fetchers:
            data_fetcher.join()

    def get_mean(self):
        return getattr(self, 'mean', [])

    def get_std(self):
        return getattr(self, 'std', [])

    @classmethod
    def get_void_label(self):
        vl = getattr(self, '_void_labels', [])
        vl = [] if vl == [] else [self.nclasses]
        return vl

    def get_cmap(self):
        return getattr(self, 'cmap', [])

    def get_labels(self):
        return getattr(self, 'labels', [])

    def get_n_classes(self):
        return self.nclasses

    def get_n_batches(self):
        return self.nbatches

    def get_n_samples(self):
        return len(self.names_list)

    def get_batch_size(self):
        return self.batch_size
