import numpy as np
import os
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
floatX = 'float32'


class Polyps300Dataset(ThreadedDataset):
    name = 'Polyps300'
    non_void_nclasses = 4
    debug_shape = (384, 288, 3)

    # optional arguments
    data_shape = (384, 288, 3)

    _void_labels = [0]
    _cmap = {
        0: (0, 0, 0),           # Void
        1: (128, 128, 0),       # Background
        2: (128, 0, 0),         # Polyp
        3: (192, 192, 128),     # Specularity
        4: (128, 64, 128)}      # Lumen
    _mask_labels = {0: 'Void', 1: 'Background', 2: 'Polyp', 3: 'Specularity',
                    4: 'Lumen'}

    def __init__(self, which_set='train', select='sequences', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.select = select
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'polyps')
        self.sharedpath = '/data/lisa/exp/vazquezd/datasets/polyps/'

        self.image_path_300 = os.path.join(self.path, 'CVC-300', "bbdd")
        self.mask_path_300 = os.path.join(self.path, 'CVC-300', "labelled")
        self.image_path_612 = os.path.join(self.path, 'CVC-612', "bbdd")
        self.mask_path_612 = os.path.join(self.path, 'CVC-612', "labelled")

        super(Polyps300Dataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""

        # Select elements where the column 'c' is in ids
        def select_elements(data, c, ids):
            select = data[np.logical_or.reduce([data[:, c] == x
                                               for x in ids])].astype(int)
            # print "Select: " + str(select)
            return select

        # Get file names from the selected files
        def select_filenames(select):
            filenames = []
            for i in range(select.shape[0]):
                filenames.append(str(select[i, 0]))
            # print "Filenames: " + str(filenames)
            return filenames

        # Get file names in this frame ids
        def by_frame(data, ids):
            return select_filenames(select_elements(data, 0, ids))

        # Get file names in this sequence ids
        def by_sequence(data, ids):
            return select_filenames(select_elements(data, 3, ids))

        # Get file names in this sequence ids
        def by_patience(data, ids):
            return select_filenames(select_elements(data, 1, ids))

        def get_file_names_by_frame(data, id_first, id_last):
            return by_frame(data, range(id_first, id_last))

        def get_file_names_by_sequence(data, id_first, id_last):
            return by_sequence(data, range(id_first, id_last))

        def get_file_names_by_patience(data, id_first, id_last):
            return by_patience(data, range(id_first, id_last))

        # Get the metadata info from the dataset
        def read_csv(file_name):
            from numpy import genfromtxt
            csv_data = genfromtxt(file_name, delimiter=';')
            return csv_data
            # [Frame ID, Patiend ID, Polyp ID, Polyp ID2]

        # Get the metadata info from the dataset
        self.CVC_300_data = read_csv(os.path.join(self.sharedpath,
                                                  'CVC-300', 'data.csv'))
        self.CVC_612_data = read_csv(os.path.join(self.sharedpath,
                                                  'CVC-612', 'data.csv'))

        if self.select == 'frames':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         1, 401)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         401, 501)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         501, 613)
                self.is_300 = False
            else:
                print 'EROR: Incorret set: ' + self.filenames
                exit()
        elif self.select == 'sequences':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            1, 21)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            21, 26)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            26, 32)
                self.is_300 = False
            else:
                print 'EROR: Incorret set: ' + self.filenames
                exit()
        elif self.select == 'patience':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            1, 16)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            16, 21)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            21, 26)
                self.is_300 = False
            else:
                print 'ERROR: Incorrect set: ' + self.filenames
                exit()
        else:
            print 'ERROR: Incorrect select: ' + self.select
            exit()

        return {'default': self.filenames}

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from skimage import io
        image_batch = []
        mask_batch = []
        filename_batch = []

        for prefix, frame in sequence:

            image_path = (self.image_path_300 if self.is_300 else
                          self.image_path_612)
            mask_path = (self.mask_path_300 if self.is_300 else
                         self.mask_path_612)

            # Load image
            img = io.imread(os.path.join(image_path, frame + ".bmp"))
            img = img.astype(floatX) / 255.
            # print 'Image shape: ' + str(img.shape)

            # Load mask
            mask = np.array(io.imread(os.path.join(mask_path, frame + ".tif")))
            mask = mask.astype('int32')

            # Add to minibatch
            image_batch.append(img)
            mask_batch.append(mask)
            filename_batch.append(frame)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['subset'] = prefix
        ret['filenames'] = np.array(filename_batch)
        return ret


def test1():
    trainiter = Polyps300Dataset(
        which_set='train',
        batch_size=5,
        seq_per_video=0,
        seq_length=10,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
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


if __name__ == '__main__':
    test1()
