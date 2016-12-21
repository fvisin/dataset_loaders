import os
import time

import numpy as np
from PIL import Image

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


class VOCdataset(ThreadedDataset):
    name = 'pascal_voc'
    non_void_nclasses = 20
    debug_shape = (375, 500, 3)

    mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
        'float32')
    std = 1.
    GTclasses = range(20) + [255]
    _void_labels = [255]

    _filenames = None
    _prefix_list = None

    @property
    def prefix_list(self):
        if self._prefix_list is None:
            prefix_list = np.array([el.split('_')[0] for el in self.filenames])
            self._prefix_list = np.unique(prefix_list)
        return self._prefix_list

    @property
    def filenames(self):
        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get paths for this year
            with open(os.path.join(
                    self.txt_path, self.which_set + ".txt")) as f:

                # Get file names for this set and year
                for fi in f.readlines():
                    raw_name = fi.strip()
                    filenames.append(raw_name)
            self._filenames = filenames
        return self._filenames

    def __init__(self,
                 which_set="train",
                 teacher_temperature=1,
                 year="VOC2012",
                 *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "trainval", "val", "test"):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)
        if self.which_set == "test" and year != "VOC2012":
            raise ValueError("No test set for other than 2012 year")
        if self.which_set == 'test':
            self.has_GT = False
        self.teacher_temperature = teacher_temperature
        self.year = year

        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'PASCAL-VOC',
            'VOCdevkit')
        self.sharedpath = '/data/lisa/data/PASCAL-VOC/VOCdevkit'

        self.txt_path = os.path.join(self.path, self.year,
                                     "ImageSets", "Segmentation")
        self.image_path = os.path.join(self.path, self.year, "JPEGImages")
        self.mask_path = os.path.join(self.path, self.year,
                                      "SegmentationClass")
        # self.pred_path = os.path.join(self.path, self.which_set,
        #                               "teacher_pred_temp1")

        super(VOCdataset, self).__init__(*args, **kwargs)

    def get_names(self):
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
        from skimage import io
        image_batch = []
        mask_batch = []
        # pred_batch = []
        filename_batch = []

        # Load image
        for _, img_name in sequence:
            img = io.imread(os.path.join(self.image_path,
                                         img_name + ".jpg"))
            img = img / 255.

            # Load mask
            if self.which_set != "test":
                mask = np.array(Image.open(
                    os.path.join(self.mask_path, img_name + ".png")))

            # Load teacher predictions and soft predictions
            # pred = np.load(os.path.join(self.pred_path, img_name + ".npy"))

            # Add to minibatch
            image_batch.append(img)
            if self.which_set != "test":
                mask_batch.append(mask)
            # pred_batch.append(pred)
            filename_batch.append(img_name)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['filenames'] = np.array(filename_batch)
        # ret['teacher'] = np.array(pred_batch)
        return ret


def test():
    trainiter = VOCdataset(
        which_set='train',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(71, 71),
        get_one_hot=True,
        get_01c=True,
        return_list=True,
        use_threads=False,
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
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (71, 71, 3)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (71, 71, nclasses)

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


if __name__ == '__main__':
    test()
