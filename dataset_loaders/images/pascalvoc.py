import time
import os
import numpy as np
from skimage import io
from PIL import Image

from batchiterator import BatchIterator, threaded_generator
from dataset_helpers import convert_01c_to_c01
import dataset_loaders


class VOCdataset(BatchIterator):
    name = 'pascal_voc'
    nclasses = 21
    debug_shape = (100, 100, 3)

    data_shape = (None, None, 3)
    mean = 0.
    std = 1.

    def __init__(self,
                 which_set="train",
                 with_filenames=False,
                 threshold_masks=False,
                 data_format="b01c",
                 year="VOC2012",
                 predictions=False,
                 normalize=True,
                 with_teacher=False,
                 teacher_temperature=1,
                 minibatch_size=1, *args, **kwargs):

        self.which_set = which_set
        self.with_filenames = with_filenames
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'PASCAL-VOC',
            'VOCdevkit')
        self.sharedpath = '/data/lisa/data/PASCAL-VOC/VOCdevkit'
        self.year = year

        if self.which_set == "test" and year == "2012":
            raise ValueError("No test set for other than 2012 year")

        self.threshold_masks = threshold_masks
        self.data_format = data_format
        self.random_state = np.random.RandomState(1999)
        self.predictions = predictions
        self.normalize = normalize
        self.with_teacher = with_teacher
        self.teacher_temperature = teacher_temperature
        self.n_classes = 21
        self.image_shape = [3, None, None]

        self.image_shape_out = [self.n_classes] + self.image_shape[1:]

        if self.which_set == "test":
            testing = True
        else:
            testing = False

        # TEMPORARY
        self.path = self.sharedpath

        self.txt_path = os.path.join(self.path, self.year,
                                     "ImageSets", "Segmentation")
        self.image_path = os.path.join(self.path, self.year, "JPEGImages")
        self.mask_path = os.path.join(self.path, self.year,
                                      "SegmentationClass")
        self.pred_path = os.path.join(self.path, self.which_set,
                                      "teacher_pred_temp1")

        data_list = self.get_names()

        super(VOCdataset, self).__init__(minibatch_size, data_list, testing)

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

        # Return images
        return image_names

    def fetch_from_dataset(self, batch_to_load):
        image_batch = []
        mask_batch = []
        pred_batch = []
        filename_batch = []

        for img_name in batch_to_load:
            # Load image
            try:
                img = io.imread(os.path.join(self.image_path, img_name +
                                             ".jpg"))
            except IOError:
                print("Failed to load " + img_name + ".jpg")
                continue

            # Load mask
            try:
                if self.which_set != "test":
                    mask = np.asarray(Image.open(os.path.join(self.mask_path,
                                                              img_name +
                                                              ".png")))
            except IOError:
                print("Failed to load " + img_name + ".png")
                continue
            # Load teacher predictions and soft predictions
            try:
                if self.predictions:
                    # Predictions
                    pred = np.load(os.path.join(self.pred_path, img_name +
                                                ".npy"))

            except IOError:
                print("Failed to load teacher prediction" +
                      img_name + ".npy")
                continue

            # Convert image to c01 format
            if self.data_format == "bc01":
                img = convert_01c_to_c01(img)

            # Normalize
            if self.normalize:
                mean_subs = np.asarray([122.67891434,
                                        116.66876762,
                                        104.00698793])
                img = np.float32(img - mean_subs[None, None, :])

            # Add to minibatch
            image_batch.append(img)
            if self.which_set != "test":
                mask_batch.append(mask)
            if self.predictions:
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
        return ret + other

    def get_image_shape(self):
        return self.image_shape

    def get_image_shape_out(self):
        return self.image_shape_out

    def get_n_classes(self):
        return self.n_classes


if __name__ == '__main__':
    d = VOCdataset(which_set="train",
                   data_format="bc01",
                   minibatch_size=500,
                   predictions=False,
                   normalize=True,
                   teacher_temperature=10)

    start = time.time()
    max_epochs = 2
    n_batches = d.get_n_batches()

    d = threaded_generator(d, 10)

    for epoch in range(max_epochs):
        for mb in range(n_batches):
            image_batch, mask_batch, filenames, pred_batch = d.next()
            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))
