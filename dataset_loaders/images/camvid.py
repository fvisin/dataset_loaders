import numpy as np
import os
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class CamvidDataset(ThreadedDataset):
    name = 'camvid'
    non_void_nclasses = 11
    path = os.path.join(
        dataset_loaders.__path__[0], 'datasets', 'camvid', 'segnet')
    sharedpath = '/data/lisatmp4/camvid/segnet/'
    _void_labels = [11]

    # optional arguments
    data_shape = (360, 480, 3)
    mean = [0.39068785, 0.40521392, 0.41434407]
    std = [0.29652068, 0.30514979, 0.30080369]

    _cmap = {
        0: (128, 128, 128),    # sky
        1: (128, 0, 0),        # building
        2: (192, 192, 128),    # column_pole
        3: (128, 64, 128),     # road
        4: (0, 0, 192),        # sidewalk
        5: (128, 128, 0),      # Tree
        6: (192, 128, 128),    # SignSymbol
        7: (64, 64, 128),      # Fence
        8: (64, 0, 128),       # Car
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Bicyclist
        11: (0, 0, 0)}         # Void
    _mask_labels = {0: 'sky', 1: 'building', 2: 'column_pole', 3: 'road',
                    4: 'sidewalk', 5: 'tree', 6: 'sign', 7: 'fence', 8: 'car',
                    9: 'pedestrian', 10: 'byciclist', 11: 'void'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            self._prefix_list = np.unique(np.array([el[:6]
                                                    for el in self.filenames]))

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set and year
            filenames = []
            with open(os.path.join(self.path, self.which_set + '.txt')) as f:
                for fi in f.readlines():
                    raw_name = fi.strip()
                    raw_name = raw_name.split("/")[4]
                    raw_name = raw_name.strip()
                    filenames.append(raw_name)
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set='train', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set == "train":
            self.image_path = os.path.join(self.path, "train")
            self.mask_path = os.path.join(self.path, "trainannot")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.path, "val")
            self.mask_path = os.path.join(self.path, "valannot")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "test")
            self.mask_path = os.path.join(self.path, "testannot")
        elif self.which_set == 'trainval':
            self.image_path = os.path.join(self.path, "trainval")
            self.mask_path = os.path.join(self.path, "trainvalannot")

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(CamvidDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_subset_names = {}
        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            per_subset_names[prefix] = [el for el in filenames if
                                        el.startswith(prefix)]
        return per_subset_names

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
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


def test1():
    d = CamvidDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=4,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)})
    start = time.time()
    tot = 0
    n_minibatches_to_run = 1000

    for mb in range(n_minibatches_to_run):
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()

        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        part = stop - start - 1
        start = stop
        tot += part
        print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def test2():
    trainiter = CamvidDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=10,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 5

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()
            if train_group is None:
                raise RuntimeError('One batch was missing')

            # train_group checks
            assert train_group[0].ndim == 5
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (10, 224, 224, 3)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 5
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (10, 224, 224, nclasses)

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


def test3():
    trainiter = CamvidDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=True,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 5

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()

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
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


if __name__ == '__main__':
    test1()
    test2()
    test3()
