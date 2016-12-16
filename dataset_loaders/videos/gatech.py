import os
import time

import numpy as np

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class GatechDataset(ThreadedDataset):
    name = 'gatech'
    non_void_nclasses = 8
    _void_labels = [0]
    debug_shape = (360, 640, 3)

    mean = [0.484375, 0.4987793, 0.46508789]
    std = [0.07699376, 0.06672145, 0.09592211]

    # wtf, sky, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    _cmap = {
        0: (255, 128, 0),      # wtf
        1: (255, 0, 0),        # sky (red)
        2: (0, 130, 180),      # ground (blue)
        3: (0, 255, 0),        # solid (buildings, etc) (green)
        4: (255, 255, 0),      # porous (yellow)
        5: (120, 0, 255),      # cars
        6: (255, 0, 255),      # humans (purple)
        7: (160, 160, 160),    # vert mix
        8: (64, 64, 64)}       # main mix
    _mask_labels = {0: 'wtf', 1: 'sky', 2: 'ground', 3: 'solid', 4: 'porous',
                    5: 'cars', 6: 'humans', 7: 'vert mix', 8: 'gen mix'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            all_prefix_list = np.unique(np.array([el[:el.index('_')]
                                                  for el in self.filenames]))
            nvideos = len(all_prefix_list)
            nvideos_set = int(nvideos*self.split)
            self._prefix_list = all_prefix_list[nvideos_set:] \
                if "val" in self.which_set else all_prefix_list[:nvideos_set]

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set
            self._filenames = os.listdir(self.image_path)
            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self,
                 which_set='train',
                 threshold_masks=False,
                 split=.75,
                 *args, **kwargs):

        self.which_set = which_set
        self.threshold_masks = threshold_masks

        self.path = os.path.join(dataset_loaders.__path__[0], 'datasets',
                                 'GATECH')
        self.sharedpath = '/data/lisatmp4/dejoieti/data/GATECH/'

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split
            if 'fcn8' in self.which_set:
                self.image_path = os.path.join(self.path, 'Images',
                                               'After_fcn8')
            else:
                self.image_path = os.path.join(self.path, 'Images',
                                               'Original')
            self.mask_path = os.path.join(self.path, 'Images', 'Ground_Truth')
        elif 'test' in self.which_set:
            self.image_path = os.path.join(self.path, 'Images_test',
                                           'Original')
            self.mask_path = os.path.join(self.path, 'Images_test',
                                          'Ground_Truth')
            self.split = split
            if 'fcn8' in self.which_set:
                raise RuntimeError('FCN8 outputs not available for test set')
        else:
            raise RuntimeError('Unknown set')

        super(GatechDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        per_video_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_video_names[prefix] = [el for el in filenames if
                                       el.startswith(prefix + '_')]
        return per_video_names


    def load_sequence(self, sequence):
        """
        Load ONE clip/sequence

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns images in [0, 1]
        """
        from skimage import io
        X = []
        Y = []
        F = []

        for prefix, frame in sequence:
            img = io.imread(os.path.join(self.image_path, frame))
            mask = io.imread(os.path.join(self.mask_path, frame))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = GatechDataset(
        which_set='train',
        batch_size=20,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        split=.75,
        get_one_hot=True,
        get_01c=True,
        return_list=True,
        use_threads=True)
    validiter = GatechDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        return_list=True,
        use_threads=True)
    testiter = GatechDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=10,
        seq_length=10,
        split=1.,
        get_one_hot=False,
        get_01c=False,
        return_list=True,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    nclasses = testiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    valid_batch_size = validiter.batch_size
    test_batch_size = testiter.batch_size

    print("Train %d, valid %d, test %d" % (train_nsamples, valid_nsamples,
                                           test_nsamples))

    start = time.time()
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()
            valid_group = validiter.next()
            test_group = testiter.next()
            if train_group is None or valid_group is None or \
               test_group is None:
                raise ValueError('.next() returned None!')

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1] == 224
            assert train_group[0].shape[2] == 224
            assert train_group[0].shape[3] == 3
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1] == 224
            assert train_group[1].shape[2] == 224
            assert train_group[1].shape[3] == nclasses

            # valid_group checks
            assert valid_group[0].ndim == 4
            assert valid_group[0].shape[0] <= valid_batch_size
            assert valid_group[0].shape[3] == 3
            assert valid_group[1].ndim == 3
            assert valid_group[1].shape[0] <= valid_batch_size
            assert valid_group[1].max() < nclasses

            # test_group checks
            assert test_group[0].ndim == 5
            assert test_group[0].shape[0] <= test_batch_size
            assert test_group[0].shape[1] == 10
            assert test_group[0].shape[2] == 3
            assert test_group[1].ndim == 4
            assert test_group[1].shape[0] <= test_batch_size
            assert test_group[1].shape[1] == 10
            assert test_group[1].max() < nclasses

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))


def test2():
    trainiter = GatechDataset(
        which_set='train',
        batch_size=500,
        seq_per_video=0,
        seq_length=7,
        overlap=6,
        crop_size=(224, 224),
        split=.75,
        get_one_hot=True,
        get_01c=True,
        use_threads=True,
        return_list=True,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size

    print("Train %d" % (train_nsamples))

    start = time.time()
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()

            print train_group[2]
            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1] == 224
            assert train_group[0].shape[2] == 224
            assert train_group[0].shape[3] == 3
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1] == 224
            assert train_group[1].shape[2] == 224
            assert train_group[1].shape[3] == nclasses

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))
if __name__ == '__main__':
    test()
    test2()
