from __future__ import division
import os
import time

import numpy as np
import cv2
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class Davis2017Dataset(ThreadedDataset):
    name = 'davis2017'
    # Non void classes?
    non_void_nclasses = None
    _void_labels = []

    # NOTE: we only load the 480p
    # 1080p images are either (1920, 1080) or (1600, 900)
    data_shape = (480, 854, 3)
    # _cmap = {
    #     0: (255, 255, 255),        # background
    #     1: (0, 0, 0)}              # foreground
    # _mask_labels = {0: 'background', 1: 'foreground'}

    _unique_rgbs = None
    _filenames = None
    _image_sets_path = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            # Create a list of prefix out of the number of requested videos
            this_set = 'val' if self.which_set == 'valid' else self.which_set
            with open(os.path.join(self._image_sets_path,
                                   this_set+'.txt')) as f:
                content = f.readlines()
                f.close()
            self._prefix_list = [prefix.strip() for prefix in content]

        return self._prefix_list

    @property
    def unique_rgbs(self):
        if self._unique_rgbs is None:
            self._unique_rgbs = {}
            # Get first file name for this set and the corresponding
            # list of unique rgb values
            for root, dirs, files in os.walk(self.mask_path):
                if files == []:
                    continue
                name = sorted(files)[0]
                # Unique list of rgbs
                mask = cv2.imread(os.path.join(root, name))
                mask_rgbs = Image.fromarray(mask).getcolors()
                mask_rgbs = [el[1] for el in mask_rgbs]
                self._unique_rgbs[root[-root[::-1].index('/'):]] = mask_rgbs

        return self._unique_rgbs

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = []
            # Get file names for this set
            for vid_dir in self.prefix_list:
                for root, dirs, files in os.walk(os.path.join(
                        self.image_path, vid_dir)):
                    for name in files:
                        self._filenames.append(os.path.join(
                            root[-root[::-1].index('/'):], name[:-3]))
            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self,
                 which_set='train',
                 multiobject=True,
                 threshold_masks=False,
                 split=.75,
                 *args, **kwargs):

        self.which_set = which_set
        self.threshold_masks = threshold_masks
        if multiobject:
            self._image_sets_path = os.path.join(self.path, 'ImageSets/2017')
        else:
            self._image_sets_path = os.path.join(self.path, 'ImageSets/2016')

        self.image_path = os.path.join(self.path,
                                       'JPEGImages', '480p')
        self.mask_path = os.path.join(self.path,
                                      'Annotations', '480p')

        # Prepare data paths
        if 'train' in self.which_set or 'val' in self.which_set:
            self.split = split
        elif 'test-dev' in self.which_set:
            self.split = 1.
        else:
            raise RuntimeError('Unknown set')

        if 'test-dev' in self.which_set and not self.which_set:
            raise RuntimeError('Single object instance is not '
                               'available for test-dev in Davis2017')

        super(Davis2017Dataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
        per_video_names = {}

        # Populate self.filenames and self.prefix_list
        filenames = self.filenames
        prefix_list = self.prefix_list

        # cycle through the different videos
        for prefix in prefix_list:
            exp_prefix = prefix + '/'
            per_video_names[prefix] = [el.lstrip(exp_prefix) for el in
                                       filenames if el.startswith(exp_prefix)]
        return per_video_names

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

        rgbs = self.unique_rgbs

        for prefix, frame_name in sequence:
            frame = prefix + '/' + frame_name

            img = io.imread(os.path.join(self.image_path, frame + 'jpg'))
            img = img.astype(floatX) / 255.

            if self.which_set == 'train' or self.which_set == 'valid':
                mask = cv2.imread(os.path.join(self.mask_path, frame + 'png'))
                # Convert mask from RGB to ids format
                _id = 0
                temp_mask = np.zeros(mask.shape[:-1], np.int32)
                for rgb in sorted(rgbs[prefix]):
                    temp_mask[np.where(np.all(mask == rgb,
                                              -1).astype(int) == 1)] = _id
                    _id += 1

                Y.append(temp_mask)

            X.append(img)
            F.append(frame_name + 'jpg')

        if self.which_set == 'test-dev':
            frame = [el for el in self.filenames if prefix in el][0]
            mask = cv2.imread(os.path.join(self.mask_path, frame + 'png'))
            # Convert mask from RGB to ids format
            _id = 0
            temp_mask = np.zeros(mask.shape[:-1], np.int32)
            for rgb in sorted(rgbs[prefix]):
                temp_mask[np.where(np.all(mask == rgb,
                                          -1).astype(int) == 1)] = _id
                _id += 1

            Y.append(temp_mask)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    trainiter = Davis2017Dataset(
        which_set='train',
        batch_size=20,
        seq_per_subset=0,
        seq_length=1,
        overlap=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        split=0.75,
        return_one_hot=False,
        return_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    validiter = Davis2017Dataset(
        which_set='valid',
        batch_size=1,
        seq_per_subset=0,
        seq_length=1,
        overlap=0,
        split=.75,
        return_one_hot=False,
        return_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)
    testiter = Davis2017Dataset(
        which_set='test-dev',
        batch_size=1,
        seq_per_subset=0,
        seq_length=1,
        overlap=0,
        split=1.,
        return_one_hot=False,
        return_01c=False,
        shuffle_at_each_epoch=False,
        use_threads=True)

    train_nsamples = trainiter.nsamples
    valid_nsamples = validiter.nsamples
    test_nsamples = testiter.nsamples
    nbatches = trainiter.nbatches

    print("Train %d, valid %d, test %d" % (train_nsamples, valid_nsamples,
                                           test_nsamples))

    start = time.time()
    tot = 0
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
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s - Threaded time: %s (%s)" % (str(mb), part,
                                                             tot))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
