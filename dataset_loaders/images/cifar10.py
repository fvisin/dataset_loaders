from math import ceil, floor
import numpy as np
import os

from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import unpickle


floatX = 'float32'


class Cifar10Dataset(ThreadedDataset):
    # Adapted from
    # https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py

    name = 'cifar10'

    # The number of *non void* classes.
    non_void_nclasses = 10

    # A list of the ids of the void labels
    _void_labels = []

    data_shape = (32, 32, 3)

    # The dataset-wide statistics (either channel-wise or pixel-wise).
    # `extra/running_stats` contains utilities to compute them.
    # mean = []
    # std = []

    # A *dictionary* of the form `class_id: (R, G, B)`. `class_id` is
    # the class id in the original data.
    _cmap = {
        0: (128, 128, 128),
        1: (128, 0, 0),
        2: (192, 192, 128),
        3: (128, 64, 128),
        4: (0, 0, 192),
        5: (128, 128, 0),
        6: (192, 128, 128),
        7: (64, 64, 128),
        8: (64, 0, 128),
        9: (64, 64, 0)}

    # A *dictionary* of form `class_id: label`. `class_id` is the class
    # id in the original data.
    _mask_labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat',
                    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship',
                    9: 'truck'}

    def __init__(self, which_set='train', split=0.80, *args, **kwargs):
        self.which_set = 'val' if which_set == 'valid' else which_set
        self.split = split

        if self.which_set == 'train':
            g_start_idx = 0
            g_end_idx = int(50000 * self.split)
        elif self.which_set == 'val':
            assert self.split < 1, 'The validation set is empty with split=1'
            g_start_idx = int(50000 * self.split)
            g_end_idx = 50000
        elif self.which_set == 'test':
            g_start_idx = 0
            g_end_idx = 10000
        else:
            raise RuntimeError('Unknown set: {}'.format(which_set))
        self.indices = range(g_start_idx, g_end_idx)
        nindices = g_end_idx - g_start_idx

        # Call the ThreadedDataset constructor. This will automatically
        # copy the dataset in self.path (the local path) if needed.
        super(Cifar10Dataset, self).__init__(*args, **kwargs)

        if self.which_set in ['train', 'val']:
            # Pre-allocate the arrays for the images and class-numbers for
            # efficiency.
            self.images = np.zeros(shape=[nindices, 32, 32, 3], dtype=float)
            self.labels = np.zeros(shape=[nindices], dtype=int)

            # Find which files to read
            f_start_idx = int(floor(g_start_idx / 10000.))
            f_end_idx = int(ceil(g_end_idx / 10000.))
            nfiles = f_end_idx - f_start_idx

            # Find the start and end indices for each file
            s_indices = [g_start_idx % 10000] + [0] * (nfiles - 1)
            e_indices = [10000] * (nfiles - 1) + [(g_end_idx-1) % 10000 + 1]

            start = 0
            for f_idx, f_s, f_e in zip(range(f_start_idx, f_end_idx),
                                       s_indices,
                                       e_indices):
                # Load the whole file
                f_images, f_labels = self._load_data(filename="data_batch_" +
                                                     str(f_idx + 1))
                # Select the indices we care about and write them to memory
                end = start + f_e - f_s
                self.images[start:end, :] = f_images[f_s:f_e]
                self.labels[start:end] = f_labels[f_s:f_e]
                start = end
            assert end == nindices
        elif self.which_set == 'test':
            self.images, self.labels = self._load_data(filename="test_batch")

    def get_names(self):
        """Return a dict of indeces for the default subset."""
        return {'default': self.indices}

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        X = []
        Y = []
        F = []

        for prefix, idx in sequence:
            X.append(self.images[idx])
            Y.append(self.labels[idx])
            F.append(idx)

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)
        return ret

    # Auxiliary functions
    def _load_data(self, filename):
        """
        Load a pickled data-file from the CIFAR-10 data-set
        and return the converted images (see above) and the class-number
        for each image.
        """

        # Load the pickled data-file.
        data = unpickle(os.path.join(self.path, filename))

        # Get the raw images.
        raw_images = data[b'data']

        # Get the class-numbers for each image. Convert to numpy-array.
        labels = np.array(data[b'labels'])

        # Convert the images.
        images = self._convert_images(raw_images)

        return images, labels

    def _convert_images(self, raw):
        """
        Convert images from the CIFAR-10 format and
        return a 4-dim array with shape: [image_number, height, width, channel]
        where the pixels are floats between 0.0 and 1.0.
        """
        # Convert the raw images from the data-files to floating-points.
        raw_float = np.array(raw, dtype=float) / 255.0

        # Reshape the array to 4-dimensions.
        images = raw_float.reshape([-1, 3, 32, 32])

        # Reorder the indices of the array to have channels in the last axis
        images = images.transpose([0, 2, 3, 1])

        return images
