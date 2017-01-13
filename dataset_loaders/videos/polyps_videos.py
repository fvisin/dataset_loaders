import os
import time

import numpy as np

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class PolypVideoDataset(ThreadedDataset):
    name = 'polyp_videos'
    non_void_nclasses = 2
    _void_labels = []

    _cmap = {
        0: (255, 255, 255),     # polyp
        1: (0, 0, 0)}           # background
    _mask_labels = {0: 'polyp', 1: 'background'}

    _filenames = None

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set
            filenames = os.listdir(self.image_path)
            filenames.sort(key=natural_keys)

            # Create a list of prefix out of the number of requested videos
            all_prefix_list = np.unique(np.array([el[:el.index('_')]
                                                  for el in filenames]))
            nvideos = len(all_prefix_list)
            nvideos_set = int(nvideos*self.split)
            this_set_prefix_list = (all_prefix_list[nvideos_set:]
                                    if "val" in self.which_set else
                                    all_prefix_list[:nvideos_set])

            # Create filenames, only include the current set (via prefix_list)
            self._filenames = {}
            for prefix in this_set_prefix_list:
                self._filenames[prefix] = [f for f in filenames if
                                           f[:f.index('_')] == prefix and
                                           f.index(prefix + '_') == 0]
        return self._filenames

    def __init__(self,
                 which_set='train',
                 threshold_masks=True,
                 split=.75, *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.threshold_masks = threshold_masks

        if self.which_set == "train" or self.which_set == "val":
            self.image_path = os.path.join(self.path,
                                           'polyp_video_frames',
                                           'Images', 'Original')
            self.mask_path = os.path.join(self.path,
                                          'polyp_video_frames',
                                          'Images', 'Ground_Truth')
            self.split = split
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path,
                                           'polyp_video_frames_test',
                                           'noBlackBand',
                                           'Original')
            self.mask_path = os.path.join(self.path,
                                          'polyp_video_frames_test',
                                          'noBlackBand',
                                          'Ground_Truth')
            self.split = 1.

        super(PolypVideoDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        return self.filenames

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

            if self.threshold_masks:
                masklow = mask < 127
                maskhigh = mask >= 127
                mask[masklow] = 0
                mask[maskhigh] = 1

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
    use_threads = False
    trainiter = PolypVideoDataset(
        which_set='train',
        batch_size=20,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        split=.75,
        return_one_hot=True,
        return_01c=True,
        return_list=True,
        use_threads=use_threads)
    validiter = PolypVideoDataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=0,
        split=.75,
        return_one_hot=False,
        return_01c=True,
        return_list=True,
        use_threads=use_threads)
    validiter2 = PolypVideoDataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=10,
        split=.75,
        return_one_hot=False,
        return_01c=True,
        return_list=True,
        use_threads=use_threads)
    testiter = PolypVideoDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=10,
        seq_length=10,
        split=1.,
        return_one_hot=False,
        return_01c=False,
        return_list=True,
        use_threads=use_threads)
    testiter2 = PolypVideoDataset(
        which_set='test',
        batch_size=1,
        seq_per_subset=10,
        seq_length=0,
        split=1.,
        return_one_hot=True,
        return_01c=False,
        return_list=True,
        use_threads=use_threads)

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
    tot = 0
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()
            valid_group = validiter.next()
            valid2_group = validiter2.next()
            test_group = testiter.next()
            test2_group = testiter2.next()
            if train_group is None or valid_group is None or \
               test_group is None or valid2_group is None or \
               test2_group is None:
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

            # valid2_group checks
            assert valid2_group[0].ndim == 5
            assert valid2_group[0].shape[0] <= valid_batch_size
            assert valid2_group[0].shape[1] == 10
            assert valid2_group[0].shape[4] == 3
            assert valid2_group[1].ndim == 4
            assert valid2_group[1].shape[0] <= valid_batch_size
            assert valid2_group[1].shape[1] == 10
            assert valid2_group[1].max() < nclasses

            # test_group checks
            assert test_group[0].ndim == 5
            assert test_group[0].shape[0] <= test_batch_size
            assert test_group[0].shape[1] == 10
            assert test_group[0].shape[2] == 3
            assert test_group[1].ndim == 4
            assert test_group[1].shape[0] <= test_batch_size
            assert test_group[1].shape[1] == 10
            assert test_group[1].max() < nclasses

            # test2_group checks
            assert len(test2_group) == 2
            assert test2_group[0].ndim == 4
            assert test2_group[0].shape[0] <= test_batch_size
            assert test2_group[0].shape[1] == 3
            assert test2_group[1].ndim == 4
            assert test2_group[1].shape[0] <= test_batch_size
            assert test2_group[1].shape[1] == nclasses

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s - Threaded time: %s (%s)" % (str(mb), part,
                                                             tot))


def test2():
    trainiter = PolypVideoDataset(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        split=.75,
        return_one_hot=True,
        return_01c=True,
        use_threads=True,
        return_list=True,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size

    print("Train %d" % (train_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 2

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
            print("Minibatch %s - Threaded time: %s (%s)" % (str(mb), part,
                                                             tot))


def run_tests():
    test()
    test2()


if __name__ == '__main__':
    run_tests()
