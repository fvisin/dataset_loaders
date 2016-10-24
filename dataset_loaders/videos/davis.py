import os
import time

import numpy as np

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class DavisDataset(ThreadedDataset):
    name = 'davis'
    nclasses = 2
    _void_labels = []
    debug_shape = (360, 640, 3)

    _cmap = {
        0: (255, 255, 255),        # background
        1: (0, 0, 0)}              # foreground
    _mask_labels = {0: 'background', 1: 'foreground'}

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            all_prefix_list = np.unique(np.array(os.listdir(self.image_path)))
            nvideos = len(all_prefix_list)
            nvideos_set = int(nvideos*self.split)
            self._prefix_list = all_prefix_list[nvideos_set:] \
                if "val" in self.which_set else all_prefix_list[:nvideos_set]

        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = []
            # Get file names for this set
            for root, dirs, files in os.walk(self.image_path):
                for name in files:
                        self._filenames.append(os.path.join(
                          root[-root[::-1].index('/'):], name[:-3]))

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
                                 'Davis', 'davis')
        self.sharedpath = '/data/lisatmp4/romerosa/datasets/davis/'

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split
            self.image_path = os.path.join(self.path,
                                           'JPEGImages', '480p',
                                           'training')
            self.mask_path = os.path.join(self.path,
                                          'Annotations', '480p',
                                          'training')
        elif 'test' in self.which_set:
            self.image_path = os.path.join(self.path,
                                           'JPEGImages', '480p', 'test')
            self.mask_path = os.path.join(self.path,
                                          'Annotations', '480p', 'test')
            self.split = 1.
        else:
            raise RuntimeError('Unknown set')

        super(DavisDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length

        self.video_length = {}

        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # Discard filenames of videos we don't care
        filenames = [f for f in filenames if f[:f.index('/')]
                     in prefix_list]

        # cycle through the different videos
        for prefix in prefix_list:
            seq_per_video = self.seq_per_video
            new_prefix = prefix + '/'
            frames = [el for el in filenames if new_prefix in el and
                      el.index(prefix+'/') == 0]
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
            img = io.imread(os.path.join(self.image_path, frame + 'jpg'))
            mask = io.imread(os.path.join(self.mask_path, frame + 'png'))

            img = img.astype(floatX) / 255.
            mask = (mask / 255).astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(frame)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        if self.with_filenames:
            ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = DavisDataset(
        which_set='train',
        batch_size=20,
        seq_per_video=0,
        seq_length=0,
        overlap=0,
        crop_size=(224, 224),
        split=0.75,
        get_one_hot=True,
        get_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    validiter = DavisDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        overlap=0,
        split=.75,
        get_one_hot=False,
        get_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    testiter = DavisDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        overlap=0,
        split=1.,
        get_one_hot=False,
        get_01c=False,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    nbatches = trainiter.get_n_batches()

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

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))


if __name__ == '__main__':
    test()
