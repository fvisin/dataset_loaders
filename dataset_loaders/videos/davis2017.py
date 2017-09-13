from __future__ import division
import os
import time

import numpy as np
from PIL import Image

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys

floatX = 'float32'


class Davis2017Dataset(ThreadedDataset):
    name = 'davis2017'

    non_void_nclasses = None
    _void_labels = []

    # NOTE: we only load the 480p
    # 1080p images are either (1920, 1080) or (1600, 900)
    data_shape = (480, 854, 3)
    # _cmap = {
    #     0: (255, 255, 255),        # background
    #     1: (0, 0, 0)}              # foreground
    # _mask_labels = {0: 'background', 1: 'foreground'}

    _filenames = None
    _image_sets_path = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            which_set = ('test-dev' if self.which_set == 'test' else
                         self.which_set)
            # Create a list of prefix out of the number of requested videos
            with open(os.path.join(self._image_sets_path,
                                   which_set + '.txt')) as f:
                content = f.readlines()
                f.close()
            self._prefix_list = [prefix.strip() for prefix in content]
        return self._prefix_list

    def save_rgbs(self):
        unique_rgbs = {}
        # Get first file name for this set and the corresponding
        # list of unique rgb values
        for root, dirs, files in os.walk(self.mask_path):
            if files == []:
                continue
            name = sorted(files)[0]
            # Unique list of rgbs
            mask = Image.open(os.path.join(root, name)).convert('RGB')
            # getcolors() returns a list of unique RGB values
            # contained in an image. The elements in the list are
            # in the form [count, RGB] where count is the number of
            # pixels with the corresponding RGB value. We don't care
            # about count here, and order the list by
            mask_rgbs = {id_rgb: rgb for id_rgb, rgb in
                         enumerate(sorted([el[1] for el in mask.getcolors()]))}
            # unique_rgbs[os.path.split(root)[-1]] = sorted(mask_rgbs)
            unique_rgbs[os.path.split(root)[-1]] = mask_rgbs
        # Save the per-frame RGB values in a numpy file
        np.save(self.rgb_values_shared_path, unique_rgbs)

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = {}
            # Get file names for this set
            for vid_dir in self.prefix_list:
                self._filenames[vid_dir] = []
                for root, dirs, files in os.walk(os.path.join(
                        self.image_path, vid_dir)):
                    for name in files:
                        self._filenames[vid_dir].append(os.path.join(
                            os.path.split(root)[-1], name[:-4]))
                self._filenames[vid_dir].sort(key=natural_keys)
        return self._filenames

    def __init__(self,
                 which_set='train',
                 foreground_background=True,
                 dataset_version='2017',
                 *args, **kwargs):

        """ The Davis 2017 dataset
        The following loader allows to load both the 2016 and 2017
        version of the Davis dataset.

        The dataset is available at http://davischallenge.org/code.html
        where it is possible to download the TrainVal and Test-Dev sets
        in zip format containig the JPEG images and their corresponding
        annotations. In the 2017 version there is an additional folder
        'ImageSets' containing the '.txt' files where the training,
        validation and test set images are listed by name for both the
        2016 and 2017 versions of the dataset.

        Parameters
        ----------
        which_set: string
            A string in ['train', 'valid', 'test'], corresponding to
            the set to be loaded.
        foreground_background: boolean
            For the 2017 version of the dataset, determines whether to return
            the instance segmentation masks or the foreground-background ones.
        dataset_version: string
            A string in ['2016', '2017']. Allows to select which version of
            Davis to load, i.e., whether to load the images and masks from the
            path specified in the config.ini under 'davis' or 'davis2017'
            respectively.
        """

        if which_set not in {"train", "valid", "test"}:
            raise ValueError("Unknown set {}".format(which_set))
        self.which_set = 'val' if which_set == 'valid' else which_set
        self.foreground_background = foreground_background
        if dataset_version not in ['2016', '2017']:
            raise RuntimeError('Unknown dataset version')
        self._image_sets_path = os.path.join(self.path, 'ImageSets',
                                             dataset_version)
        self.image_path = os.path.join(self.path,
                                       'JPEGImages', '480p')
        self.mask_path = os.path.join(self.path,
                                      'Annotations', '480p')
        self.rgb_values_shared_path = os.path.join(
            self.shared_path, 'rgb_values_' + dataset_version + '.npy')
        self.rgb_values_local_path = os.path.join(
            self.path, 'rgb_values_' + dataset_version + '.npy')
        if not os.path.exists(self.rgb_values_shared_path):
            self.save_rgbs()

        super(Davis2017Dataset, self).__init__(*args, **kwargs)
        self.unique_rgbs = np.load(self.rgb_values_local_path).item()

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

        rgbs = self.unique_rgbs
        for prefix, frame_name in sequence:
            img = io.imread(os.path.join(self.image_path, frame_name + '.jpg'))
            img = img.astype(floatX) / 255.

            if self.which_set in ['train', 'val']:
                mask = np.array(Image.open(os.path.join(
                    self.mask_path, frame_name + '.png')).convert('RGB'))
            elif self.which_set == 'test':
                # By construction the test-dev set of the 2017 version
                # of the dataset provides the labels for the first frame
                # only
                first_frame = self.filenames[prefix][0]
                mask = np.array(Image.open(os.path.join(
                    self.mask_path, first_frame + '.png')).convert('RGB'))
            else:
                raise RuntimeError()

            # Convert mask from RGB to ids format
            for id_rgb, rgb in rgbs[prefix].iteritems():
                mask[np.all(mask == rgb, axis=-1), 0] = id_rgb
            mask = mask[..., 0]  # Keep only the id channel
            if self.foreground_background:
                mask[mask > 1] = 1

            Y.append(mask)
            X.append(img)
            F.append(frame_name + '.jpg')

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret


def test():
    use_threads = True
    trainiter = Davis2017Dataset(
        which_set='train',
        dataset_version='2017',
        batch_size=2,
        seq_per_subset=0,
        seq_length=7,
        overlap=0,
        data_augm_kwargs={'crop_size': None},
        foreground_background=False,
        return_one_hot=False,
        return_01c=True,
        use_threads=use_threads,
        nthreads=3,
        shuffle_at_each_epoch=True)
    validiter = Davis2017Dataset(
        which_set='valid',
        dataset_version='2017',
        batch_size=2,
        seq_per_subset=0,
        seq_length=7,
        overlap=0,
        foreground_background=True,
        return_one_hot=False,
        return_01c=True,
        use_threads=use_threads,
        shuffle_at_each_epoch=False)
    testiter = Davis2017Dataset(
        which_set='test',
        dataset_version='2017',
        batch_size=1,
        seq_per_subset=0,
        seq_length=1,
        overlap=0,
        foreground_background=True,
        return_one_hot=False,
        return_01c=True,
        shuffle_at_each_epoch=False,
        use_threads=use_threads)

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
