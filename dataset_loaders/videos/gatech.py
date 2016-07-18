import os
import time

import numpy as np

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class GatechDataset(ThreadedDataset):
    name = 'gatech'
    nclasses = 8
    _void_labels = [0]
    debug_shape = (360, 640, 3)
    mean = [0.484375, 0.4987793, 0.46508789]
    std = [0.07699376, 0.06672145, 0.09592211]
    # wtf, sky, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    cmap = np.array([
        (255, 128, 0),      # wtf
        (255, 0, 0),        # sky (red)
        (0, 130, 180),      # ground (blue)
        (0, 255, 0),        # solid (buildings, etc) (green)
        (255, 255, 0),      # porous (yellow)
        (120, 0, 255),      # cars
        (255, 0, 255),      # humans (purple)
        (160, 160, 160),    # vert mix
        (64, 64, 64)])      # main mix
    cmap = cmap / 255.
    labels = ('wtf', 'sky', 'ground', 'solid', 'porous',
              'cars', 'humans', 'vert mix', 'gen mix')

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
            self._prefix_list = all_prefix_list[-nvideos_set:] \
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
                 with_filenames=False,
                 split=.75,
                 *args, **kwargs):

        self.which_set = which_set
        self.threshold_masks = threshold_masks
        self.with_filenames = with_filenames

        self.path = os.path.join(dataset_loaders.__path__[0], 'datasets',
                                 'GATECH')
        self.sharedpath = '/data/lisatmp4/dejoieti/data/GATECH/'

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split if self.which_set == "train" else (1 - split)
            if 'fcn8' in self.which_set:
                self.image_path = os.path.join(self.path, 'Images',
                                               'After_fcn8')
            else:
                self.image_path = os.path.join(self.path, 'Images',
                                               'Original')
            self.mask_path = os.path.join(self.path, 'Images', 'Ground_Truth')
        elif 'test' in self.which_set:
            self.image_path = \
                os.path.join(self.path, 'Images_test', 'Original')
            self.mask_path = os.path.join(self.path, 'Images_test',
                                          'Ground_Truth')
            self.split = split
            if 'fcn8' in self.which_set:
                raise RuntimeError('FCN8 outputs not available for test set')
        else:
            raise RuntimeError('Unknown set')

        super(GatechDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length

        self.video_length = {}

        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # Discard filenames of videos we don't care
        filenames = [f for f in filenames if f[:f.index('_')]
                     in prefix_list]

        # cycle through the different videos
        for prefix in prefix_list:
            seq_per_video = self.seq_per_video
            new_prefix = prefix + '_'
            frames = [el for el in filenames if new_prefix in el and
                      el.index(prefix+'_') == 0]
            video_length = len(frames)
            self.video_length[prefix] = video_length

            # Fill sequences with (prefix, frame_idx)
            max_num_frames = video_length - seq_length + 1
            if (not self.seq_length or not self.seq_per_video or
                    self.seq_length >= video_length):
                # Use all possible frames
                for el in [(prefix, f) for f in
                           frames[:max_num_frames:self.overlap]]:
                    sequences.append(el)
            else:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                if max_num_frames < seq_per_video:
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, seq_length,
                                          prefix, video_length))
                    seq_per_video = max_num_frames

                # pick `seq_per_video` random indexes between 0 and
                # (video length - sequence length)
                first_frame_indexes = np.random.permutation(range(
                    max_num_frames))[0:seq_per_video]

                for i in first_frame_indexes:
                    sequences.append((prefix, frames[i]))

        return np.array(sequences)

    def load_sequence(self, first_frame):
        """
        Load ONE clip/sequence
        Auxiliary function which loads a sequence of frames with
        the corresponding ground truth and potentially filenames.
        Returns images in [0, 1]
        """
        from skimage import io
        X = []
        Y = []
        F = []

        prefix, first_frame_name = first_frame

        if (self.seq_length is None or
                self.seq_length > self.video_length[prefix]):
            seq_length = self.video_length[prefix]
        else:
            seq_length = self.seq_length

        start_idx = self.filenames.index(first_frame_name)
        for frame in self.filenames[start_idx:start_idx + seq_length]:
            img = io.imread(os.path.join(self.image_path, frame))
            mask = io.imread(os.path.join(self.mask_path, frame))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        if self.with_filenames:
            return np.array(X), np.array(Y), np.array(F)
        else:
            return np.array(X), np.array(Y)


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
        use_threads=True)
    validiter = GatechDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        use_threads=True)
    validiter2 = GatechDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=10,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        use_threads=True)
    testiter = GatechDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=10,
        seq_length=10,
        split=1.,
        get_one_hot=False,
        get_01c=False,
        use_threads=True)
    testiter2 = GatechDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=10,
        seq_length=0,
        split=1.,
        get_one_hot=True,
        get_01c=False,
        with_filenames=True,
        use_threads=True)

    train_nsamples = trainiter.get_n_samples()
    valid_nsamples = validiter.get_n_samples()
    test_nsamples = testiter.get_n_samples()
    nclasses = testiter.get_n_classes()
    nbatches = trainiter.get_n_batches()
    train_batch_size = trainiter.get_batch_size()
    valid_batch_size = validiter.get_batch_size()
    test_batch_size = testiter.get_batch_size()

    print("Train %d, valid %d, test %d" % (train_nsamples, valid_nsamples,
                                           test_nsamples))

    start = time.time()
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
            assert len(test2_group) == 3
            assert test2_group[0].ndim == 4
            assert test2_group[0].shape[0] <= test_batch_size
            assert test2_group[0].shape[1] == 3
            assert test2_group[1].ndim == 4
            assert test2_group[1].shape[0] <= test_batch_size
            assert test2_group[1].shape[1] == nclasses

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
        overlap=7,
        crop_size=(224, 224),
        split=.75,
        get_one_hot=True,
        get_01c=True,
        use_threads=True,
        nthreads=5)

    train_nsamples = trainiter.get_n_samples()
    nclasses = trainiter.get_n_classes()
    nbatches = trainiter.get_n_batches()
    train_batch_size = trainiter.get_batch_size()

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
    test2()
