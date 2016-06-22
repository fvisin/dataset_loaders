import os
import numpy as np
from PIL import Image

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset

class_ids = {'255': 21}


class VOCdataset(ThreadedDataset):
    name = 'pascal_voc'
    nclasses = 21
    debug_shape = (375, 500, 3)

    data_shape = (None, None, 3)
    mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
        'float32')
    std = 1.
    void_labels = [21]

    def __init__(self,
                 which_set="train",
                 with_filenames=False,
                 with_predictions=False,
                 with_teacher=False,
                 teacher_temperature=1,
                 year="VOC2012",
                 *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set == "test" and year != "VOC2012":
            raise ValueError("No test set for other than 2012 year")
        if self.which_set == 'test':
            self.has_GT = False
        self.with_filenames = with_filenames
        self.with_predictions = with_predictions
        self.with_teacher = with_teacher
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
        # Get filenames for the current set
        def read_filenames(seg_file):

            image_filenames = []

            # Get paths for this year
            filenames_file = os.path.join(self.txt_path, seg_file)

            # Get file names for this set and year
            with open(filenames_file) as f:
                for fi in f.readlines():
                    raw_name = fi.strip()
                    image_filenames.append(raw_name)

            return image_filenames

        # Load filenames
        if self.which_set in ("train", "trainval", "val", "test"):
            image_names = read_filenames(self.which_set + ".txt")
        else:
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        # Limit to the number of videos we want
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        video_length = len(image_names)
        max_num_sequences = video_length - seq_length + 1
        if (not self.seq_length or not self.seq_per_video or
                self.seq_length >= video_length):
            # Use all possible frames
            sequences = image_names[:max_num_sequences]
        else:
            if max_num_sequences < seq_per_video:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                print("/!\ Warning : you asked {} sequences of {} "
                      "frames each but the dataset only has {} "
                      "frames".format(seq_per_video, seq_length,
                                      video_length))
                seq_per_video = max_num_sequences

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
            mask[mask == 255] = 21

        # Load teacher predictions and soft predictions
        if self.with_predictions:
            pred = np.load(os.path.join(self.pred_path, img_name +
                                        ".npy"))

        # Add to minibatch
        image_batch.append(img)
        if self.which_set != "test":
            mask_batch.append(mask)
        if self.with_predictions:
            pred_batch.append(pred)
        if self.with_filenames:
            filename_batch.append(img_name)

        ret = [np.array(image_batch), np.array(mask_batch)]

        other = []
        if self.with_filenames:
            other += [np.array(filename_batch)]
        if self.with_teacher:
            other += [pred_batch]

        # return image_batch, mask_batch, batch_to_load, pred_batch
        return tuple(ret + other)

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
