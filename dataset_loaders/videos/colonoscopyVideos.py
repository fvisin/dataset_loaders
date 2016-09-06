import os
import time

import numpy as np

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class PolypVideoDataset(ThreadedDataset):
    name = 'colonoscopyVideos'
    nclasses = 2
    _void_labels = []
    debug_shape = (288, 384, 3)

    _cmap = {
        0: (255, 255, 255),     # polyp
        1: (0, 0, 0)}           # background
    _mask_labels = {0: 'polyp', 1: 'background'}

    _filenames = None

    @property
    def filenames(self):
        if self._filenames is None:
            # Get file names for this set
            self._filenames = os.listdir(self.image_path)
            self._filenames.sort(key=natural_keys)
        return self._filenames

    def __init__(self,
                 which_set='train',
                 threshold_masks=True,
                 with_filenames=False,
                 split=.75, *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.threshold_masks = threshold_masks
        self.with_filenames = with_filenames

        # Prepare data paths
        self.path = os.path.join(dataset_loaders.__path__[0], 'datasets',
                                 'POLYP_VIDEOS')
        self.sharedpath = ('/data/lisatmp4/dejoieti/data/data_colo/')
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
        sequences = []
        seq_length = self.seq_length

        all_prefix_list = np.unique(np.array([el[:el.index('_')]
                                              for el in self.filenames]))

        nvideos = len(all_prefix_list)
        nvideos_set = int(nvideos*self.split)
        prefix_list = all_prefix_list[nvideos_set:] \
            if self.which_set == "val" else all_prefix_list[:nvideos_set]

        # update filenames list
        self.filenames = [f for f in self.filenames if f[:f.index('_')]
                          in prefix_list]

        self.video_length = {}
        # cycle through the different videos
        for prefix in prefix_list:
            seq_per_video = self.seq_per_video
            new_prefix = prefix + '_'
            frames = [el for el in self.filenames if new_prefix in el and
                      el.index(prefix+'_') == 0]
            video_length = len(frames)
            self.video_length[prefix] = video_length

            # Fill sequences with (prefix, frame_idx)
            max_num_frames = video_length - seq_length + 1
            if (not self.seq_length or not self.seq_per_video or
                    self.seq_length >= video_length):
                # Use all possible frames
                for el in [(prefix, f) for f in frames[
                        :max_num_frames:self.seq_length - self.overlap]]:
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

                if self.overlap != self.seq_length - 1:
                    raise('Overlap other than seq_length - 1 is not '
                          'implemented')

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

            if self.threshold_masks:
                masklow = mask < 127
                maskhigh = mask >= 127
                mask[masklow] = 0
                mask[maskhigh] = 1

            X.append(img)
            Y.append(mask)
            F.append(frame)

        if self.with_filenames:
            return np.array(X), np.array(Y), np.array(F)
        else:
            return np.array(X), np.array(Y)


def test():
    trainiter = PolypVideoDataset(
        which_set='train',
        batch_size=20,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        split=.75,
        get_one_hot=True,
        get_01c=True,
        use_threads=True)
    validiter = PolypVideoDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        use_threads=True)
    validiter2 = PolypVideoDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=10,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        use_threads=True)
    testiter = PolypVideoDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=10,
        seq_length=10,
        split=1.,
        get_one_hot=False,
        get_01c=False,
        use_threads=True)
    testiter2 = PolypVideoDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=10,
        seq_length=0,
        split=1.,
        get_one_hot=True,
        get_01c=False,
        with_filenames=True,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
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
    trainiter = PolypVideoDataset(
        which_set='train',
        batch_size=10,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        split=.75,
        get_one_hot=True,
        get_01c=True,
        use_threads=True,
        nthreads=5)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.get_n_classes()
    nbatches = trainiter.get_n_batches()
    train_batch_size = trainiter.get_batch_size()

    print("Train %d" % (train_nsamples))

    start = time.time()
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
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))


if __name__ == '__main__':
    test2()
