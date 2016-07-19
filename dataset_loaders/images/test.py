import numpy as np
import os
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class TestDataset(ThreadedDataset):
    name = 'test'
    nclasses = 10
    debug_shape = (360, 480, 3)

    # optional arguments
    data_shape = (360, 480, 3)
    mean = 0.
    std = 1.
    _void_labels = [3, 5]

    _filenames = None

    @property
    def filenames(self):
        return ['test']

    def __init__(self, *args, **kwargs):

        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'camvid', 'segnet')
        self.sharedpath = '/data/lisa/exp/visin/datasets/camvid/segnet'

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(TestDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        return np.array(['test'])

    def load_sequence(self, first_frame):
        img = np.random.random((10, 10, 1))
        mask = np.array(range(10) * 10).reshape((10, 10))

        return np.array([img]), np.array([mask])


def test3():
    trainiter = TestDataset(
        batch_size=1,
        crop_size=(224, 224),
        get_one_hot=False)

    train_nsamples = trainiter.get_n_samples()
    nclasses = trainiter.get_n_classes()
    nbatches = trainiter.get_n_batches()
    train_batch_size = trainiter.get_batch_size()
    print("Train %d" % (train_nsamples))

    start = time.time()
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
            assert train_group[1].ndim == 3
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1] == 224
            assert train_group[1].shape[2] == 224
            assert train_group[1].shape[3] == nclasses

            # time.sleep approximates running some model
            time.sleep(0.1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))
        print('ended epoch --> should reset!')
        time.sleep(2)


def test1():
    d = CamvidDataset(
        which_set='train',
        batch_size=5,
        seq_per_video=4,
        seq_length=0,
        crop_size=(224, 224))
    start = time.time()
    n_minibatches_to_run = 1000
    itr = 1
    while True:
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        tot = stop - start
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        print("Tot time: %s" % (tot))
        itr += 1
        # test
        if itr >= n_minibatches_to_run:
            break


def test2():
    d = CamvidDataset(
        which_set='train',
        with_filenames=True,
        batch_size=5,
        seq_per_video=0,
        seq_length=10,
        overlap=10,
        get_one_hot=True,
        crop_size=(224, 224))
    for i, _ in enumerate(range(d.epoch_length)):
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()
        sh = image_group[0].shape
        print(image_group[2])
        if sh[1] != 2:
            raise RuntimeError()


if __name__ == '__main__':
    test2()
