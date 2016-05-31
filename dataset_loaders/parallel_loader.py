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
    `load_sequence` functions to load the list of filenames to be
    loaded and define how to load the data from the dataset,
    respectively.
    """
    def __init__(self,
                 seq_per_video=0,  # if 0 all frames
                 seq_length=0,  # if 0, return 4D
                 crop_size=None,
                 batch_size=1,
                 queues_size=10,
                 get_one_hot=False,
                 get_01c=False,
                 use_threads=False,
                 shuffle_at_each_epoch=True,
                 infinite_iterator=True,
                 rng=RandomState(0xbeef),
                 **kwargs):

        if len(kwargs):
            print('Ignored arguments: {}'.format(kwargs.keys()))

        #for attr in ['name', 'nclasses', '_is_one_hot', '_is_01c',
        #             'debug_shape', 'path', 'sharedpath']:
        #    assert attr in dir(self), (
        #        '{} did not set the mandatory attribute {}'.format(
        #            self.__class__.name, attr))
        ds = getattr(self.__class__, 'data_shape', (None, None, 3))
        if get_01c == self._is_01c:
            self.data_shape = ds
        elif self._is_01c:
            self.data_shape = ds[2], ds[0], ds[1]
        else:
            self.data_shape = ds[1], ds[2], ds[0]

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
            # (s, 0, 1, c) or (s, c, 0, 1)
            ret = self.load_sequence(el)
            seq_x, seq_y = ret[0:2]
            assert seq_x.ndim == 4

            X.append(seq_x)
            Y.append(seq_y)
            Other.append(ret[2:])
        # (b, s, 0, 1, c) or (b, s, c, 0, 1)
        X, Y, Other = np.array(X), np.array(Y), np.array(Other)

        # Crop
        crop = self.crop_size
        if crop:
            if self._is_01c:
                height, width = X.shape[-3:-1]
            else:
                height, width = X.shape[-2:]
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
            if self._is_01c and self._is_one_hot:
                X = X[..., top:top+crop[0], left:left+crop[1], :]
                Y = Y[..., top:top+crop[0], left:left+crop[1], :]
            elif self._is_01c and not self._is_one_hot:
                X = X[..., top:top+crop[0], left:left+crop[1], :]
                Y = Y[..., top:top+crop[0], left:left+crop[1]]
            elif not self._is_01c and self._is_one_hot:
                X = X[..., :, top:top+crop[0], left:left+crop[1]]
                Y = Y[..., :, top:top+crop[0], left:left+crop[1]]
            else:
                X = X[..., :, top:top+crop[0], left:left+crop[1]]
                Y = Y[..., top:top+crop[0], left:left+crop[1]]

        # get_01c
        if self._is_01c != self.get_01c:
            if self._is_01c and self._is_one_hot:
                # b,s,0,1,c --> b,s,c,0,1
                X = X.transpose([0, 1, 4, 2, 3])
                Y = Y.transpose([0, 1, 4, 2, 3])
            elif self._is_01c and not self._is_one_hot:
                # b,s,0,1,c --> b,s,c,0,1
                X = X.transpose([0, 1, 4, 2, 3])
            elif not self._is_01c and self._is_one_hot:
                # b,s,c,0,1 --> b,s,0,1,c
                X = X.transpose([0, 1, 3, 4, 2])
            else:
                # b,s,c,0,1 --> b,s,0,1,c
                X = X.transpose([0, 1, 3, 4, 2])
                Y = Y.transpose([0, 1, 3, 4, 2])

        # get_one_hot
        if self._is_one_hot != self.get_one_hot:
            if self._is_one_hot and self.get_01c:
                Y = Y.argmax(-1)
            elif self._is_one_hot and not self.get_01c:
                Y = Y.argmax(-3)
            else:
                nc = self.nclasses
                sh = Y.shape
                Y = Y.flatten()
                Y_hot = np.zeros((Y.shape[0], nc), dtype='int32')
                Y = Y.astype('int32')
                Y_hot[range(Y.shape[0]), Y] = 1
                Y_hot = Y_hot.reshape(sh + (nc,))
                if not self.get_01c:
                    # b,s,0,1,c --> b,s,c,0,1
                    Y_hot = Y_hot.transpose([0, 1, 4, 2, 3])
                Y = Y_hot

        if not self.return_sequence:
            X, Y = X[0, ...], Y[0, ...]

        if len(Other) == 0:
            return X, Y
        else:
            return X, Y, Other

    def reset(self, shuffle_at_each_epoch):
        # Shuffle data
        if shuffle_at_each_epoch:
            self.rng.shuffle(self.names_list)

        # Group data into minibatches
        name_batches = [el for el in izip_longest(
            fillvalue=None, *[iter(self.names_list)] * self.batch_size)]
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
                data_batch = self.fetch_from_dataset(name_batch)
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
