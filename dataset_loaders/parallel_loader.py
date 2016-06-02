from itertools import izip_longest
try:
    import Queue
except ImportError:
    import queue as Queue
import os
import shutil
from threading import Condition, Thread
from time import sleep

import numpy as np
from numpy.random import RandomState


def threaded_fetch(names_queue, out_queue, _reset_lock, fetch_from_dataset):
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
                minibatch_data = fetch_from_dataset(batch_to_load)

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
                 shuffle_at_each_epoch=True,
                 infinite_iterator=True,
                 rng=RandomState(0xbeef),
                 **kwargs):

        if len(kwargs):
            print('Ignored arguments: {}'.format(kwargs.keys()))

        # for attr in ['name', 'nclasses', '_is_one_hot', '_is_01c',
        #             'debug_shape', 'path', 'sharedpath']:
        #    assert attr in dir(self), (
        #        '{} did not set the mandatory attribute {}'.format(
        #            self.__class__.name, attr))
        ds = getattr(self.__class__, 'data_shape', (None, None, 3))

        self.data_shape = ds[2], ds[0], ds[1]

        # Copy the data to the local path if not existing
        if not os.path.exists(self.path):
            print('The local path {} does not exist. Copying '
                  'dataset...'.format(self.path))
            shutil.copytree(self.sharedpath, self.path)
            print('Done.')

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
        self.shuffle_at_each_epoch = shuffle_at_each_epoch
        self.infinite_iterator = infinite_iterator
        self.rng = rng

        self.names_list = self.get_names()
        if len(self.names_list) == 0:
            raise RuntimeError('The name list cannot be empty')

        # initialize
        if self.use_threads:
            self._reset_lock = Condition()
            self.names_queue = Queue.Queue(maxsize=self.queues_size)
            self.out_queue = Queue.Queue(maxsize=self.queues_size)

            # Start the data fetcher thread
            data_fetcher = Thread(
                target=threaded_fetch,
                args=(self.names_queue, self.out_queue, self._reset_lock,
                      self.fetch_from_dataset))
            data_fetcher.setDaemon(True)
            data_fetcher.start()
            self.data_fetcher = data_fetcher
        self.reset(self.shuffle_at_each_epoch)

        # Give time to the data fetcher to die, in case of errors
        sleep(1)

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
            if seq_x.ndim == 3:
                seq_x = seq_x[np.newaxis, ...]
                seq_y = seq_y[np.newaxis, ...]
            assert seq_x.ndim == 4

            # Crop
            crop = self.crop_size
            if crop:
                height, width = seq_x.shape[-3:-1]

                if crop[0] < height:
                    top = self.rng.randint(height - crop[0])
                else:
                    print('Crop size exceeds image size')
                    top = 0
                    crop[0] = height
                if crop[1] < width:
                    left = self.rng.randint(width - crop[1])
                else:
                    print('Crop size exceeds image size')
                    left = 0
                    crop[1] = width

                seq_x = seq_x[..., top:top+crop[0], left:left+crop[1], :]
                seq_y = seq_y[..., top:top+crop[0], left:left+crop[1]]
                # if not _is_one_hot:
                #     seq_x = seq_x[..., top:top+crop[0], left:left+crop[1], :]
                #     seq_y = seq_y[..., top:top+crop[0], left:left+crop[1]]

            # Transform targets seq_y to one hot code if get_one_hot is
            # true
            if self.get_one_hot:
                nc = self.nclasses
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
                if self.get_one_hot:
                    seq_y = seq_y.transpose([0, 3, 1, 2])

            # if self._is_one_hot != self.get_one_hot:
            #     if self._is_one_hot and self.get_01c:
            #         seq_y = seq_y.argmax(-1)
            #     elif self._is_one_hot and not self.get_01c:
            #         seq_y = seq_y.argmax(-3)
            #     else:
            #         nc = self.nclasses
            #         sh = seq_y.shape
            #         seq_y = seq_y.flatten()
            #         seq_y_hot = np.zeros((seq_y.shape[0], nc), dtype='int32')
            #         seq_y = seq_y.astype('int32')
            #         seq_y_hot[range(seq_y.shape[0]), Y] = 1
            #         seq_y_hot = seq_y_hot.reshape(sh + (nc,))
            #         if not self.get_01c:
            #             # b,s,0,1,c --> b,s,c,0,1
            #             seq_y_hot = seq_y_hot.transpose([0, 1, 4, 2, 3])
            #         seq_y = seq_y_hot

            # Return 4D images
            if not self.return_sequence:
                seq_x, seq_y = seq_x[0, ...], seq_y[0, ...]

            # Append stuff to minibatch list
            X.append(seq_x)
            Y.append(seq_y)
            Other.append(ret[2:])

        if len(Other[0]) == 0:
            return np.array(X), np.array(Y)
        else:
            return np.array(X), np.array(Y), np.array(Other)

    def reset(self, shuffle_at_each_epoch):
        # Shuffle data
        self.end_of_epoch = False
        if shuffle_at_each_epoch:
            self.rng.shuffle(self.names_list)
            print("shuffle %s" % self.names_list[0])

        # Group names into minibatches
        name_batches = [el for el in izip_longest(
            fillvalue=None, *[iter(self.names_list)] * self.batch_size)]
        self.nsamples = len(self.names_list)
        self.nbatches = len(name_batches)
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
            try:
                name_batch = self.name_batches.next()
                self.names_queue.put(name_batch)
            except StopIteration:
                break

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
                    self.end_of_epoch = True
                except StopIteration:
                    pass
                self.out_queue.task_done()
            except Queue.Empty:
                if self.end_of_epoch:
                    # No more minibatches in the out queue
                    print "reseting in threads"
                    import ipdb; ipdb.set_trace()
                    self.reset(self.shuffle_at_each_epoch)
                    if not self.infinite_iterator:
                        raise StopIteration
                    else:
                        data_batch = self._step()
                else:
                    data_batch = self._step()
        else:
            try:
                name_batch = self.name_batches.next()
                data_batch = self.fetch_from_dataset(name_batch)
            except StopIteration:
                self.reset(self.shuffle_at_each_epoch)
                if not self.infinite_iterator:
                    raise
                else:
                    data_batch = self._step()

        if data_batch is None:
            raise RuntimeError("ẀTF")
        return data_batch

    def finish(self):
        self.data_fetcher.join()

    def get_mean(self):
        return getattr(self, 'mean', [])

    def get_std(self):
        return getattr(self, 'std', [])

    def get_void_labels(self):
        return getattr(self, 'void_labels', [])

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
