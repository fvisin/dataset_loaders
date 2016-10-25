import os
import numpy as np
from PIL import Image

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


class VOCdataset(ThreadedDataset):
    name = 'pascal_voc'
    nclasses = 21
    debug_shape = (375, 500, 3)

    data_shape = (None, None, 3)
    mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
        'float32')
    std = 1.
    GT_classes = range(20) + [255]
    _void_labels = [255]

    _filenames = None

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
        self.pred_path = os.path.join(self.path, self.which_set,
                                      "teacher_pred_temp1")

        super(VOCdataset, self).__init__(*args, **kwargs)

    def get_names(self):

        # Limit to the number of videos we want
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        image_names = self.filenames
        video_length = len(image_names)
        max_num_sequences = video_length - seq_length + 1
        if (not self.seq_length or not self.seq_per_video or
                self.seq_length >= video_length):
            # Use all possible frames
            sequences = image_names[:max_num_sequences:
                                    self.seq_length - self.overlap]
        else:
            if max_num_sequences < seq_per_video:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                print("/!\ Warning : you asked {} sequences of {} "
                      "frames each but the dataset only has {} "
                      "frames".format(seq_per_video, seq_length,
                                      video_length))
                seq_per_video = max_num_sequences

            if self.overlap != self.seq_length - 1:
                raise('Overlap other than seq_length - 1 is not '
                      'implemented')
            # pick `seq_per_video` random indexes between 0 and
            # (video length - sequence length)
            first_frame_indexes = self.rng.permutation(range(
                max_num_sequences))[0:seq_per_video]

            for i in first_frame_indexes:
                sequences.append(image_names[i])

        # Return images
        return np.array(sequences)

    def load_sequence(self, img_name):
        from skimage import io
        image_batch = []
        mask_batch = []
        pred_batch = []
        filename_batch = []

        if self.seq_length != 1:
            raise NotImplementedError()

        # Load image
        img = io.imread(os.path.join(self.image_path, img_name +
                                     ".jpg"))
        img = img / 255.

        # Load mask
        if self.which_set != "test":
            mask = np.array(Image.open(
                os.path.join(self.mask_path, img_name + ".png")))

        # Load teacher predictions and soft predictions
        pred = np.load(os.path.join(self.pred_path, img_name + ".npy"))

        # Add to minibatch
        image_batch.append(img)
        if self.which_set != "test":
            mask_batch.append(mask)
        pred_batch.append(pred)
        filename_batch.append(img_name)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['filenames'] = np.array(filename_batch)
        ret['teacher'] = np.array(pred_batch)
        return ret

if __name__ == '__main__':
    dd = VOCdataset(which_set='test',
                    shuffle_at_each_epoch=True,
                    get_one_hot=True,
                    get_01c=False,)

    print('Tot {}'.format(dd.epoch_length))
    for i, _ in enumerate(range(dd.epoch_length)):
        dd.next()
        if i % 20 == 0:
            print str(i)
