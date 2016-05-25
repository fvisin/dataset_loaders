import os
import time
import numpy as np
from skimage import io
from dataset_helpers import random_crop, convert_01c_to_c01
from batchiterator import BatchIterator, threaded_generator


class CAMVIDdataset(BatchIterator):
    def __init__(self,
                 camvid_path="/Tmp/romerosa/datasets/camvid/segnet/",
                 which_set="train",
                 data_format="b01c",
                 minibatch_size=10,
                 crop_size="no_crop",
                 normalize=False,
                 predictions=False,
                 teacher_temperature=1.):

        self.which_set = which_set
        self.data_format = data_format
        self.camvid_path = camvid_path
        self.n_classes = 12
        self.image_shape = [3, 360, 480]
        self.image_shape_out = [21, 360, 480]
        self.crop_size = crop_size
        self.normalize = normalize
        self.data_format = data_format
        self.minibatch_size = minibatch_size
        self.random_state = np.random.RandomState(0xbeef)
        self.predictions = predictions
        self.teacher_temperature = teacher_temperature

        if self.which_set == "train":
            self.image_path = os.path.join(self.camvid_path, "train")
            self.mask_path = os.path.join(self.camvid_path, "trainannot")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.camvid_path, "val")
            self.mask_path = os.path.join(self.camvid_path, "valannot")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.camvid_path, "test")
            self.mask_path = os.path.join(self.camvid_path, "testannot")

        self.pred_path = os.path.join(self.camvid_path,
                                      self.which_set +
                                      "_teacher_temp1")

        data_list = self.get_names()

        if self.which_set == "test":
            testing = True
        else:
            testing = False

        super(CAMVIDdataset, self).__init__(minibatch_size,
                                            data_list,
                                            testing)

    def get_image_shape(self):
        return self.image_shape

    def get_image_shape_out(self):
        return self.image_shape_out

    def get_n_classes(self):
        return self.n_classes

    def get_names(self):
        filenames = []

        for directory, _, images in os.walk(self.image_path):
            filenames.extend([im for im in images if ".png" in im])

        filenames = sorted(filenames)

        return filenames

    def fetch_from_dataset(self, batch_to_load):

        image_batch = []
        mask_batch = []
        pred_batch = []

        for img_name in batch_to_load:

            # Load image
            try:
                img = io.imread(os.path.join(self.image_path,
                                             img_name))
            except IOError:
                print("Failed to load image " + img_name)
                continue

            # Load mask
            try:
                mask = io.imread(os.path.join(self.mask_path,
                                              img_name))
            except IOError:
                print("Failed to load mask " + img_name)

            # Load teacher predictions and soft predictions
            try:
                if self.predictions:
                    # Predictions
                    pred = np.load(os.path.join(self.pred_path,
                                                img_name[:-3] +
                                                "npy"))

            except IOError:
                print("Failed to load teacher (soft) prediction " +
                      img_name[:-3] + "npy")
                continue

            # Convert image to c01 format
            if self.data_format == "bc01":
                img = convert_01c_to_c01(img)

            # Normalize
            if self.normalize:
                raise NotImplementedError

            # Crop if necessary
            if not self.crop_size == 'no_crop' and \
               not self.which_set == "test":
                if self.predictions:
                    img, mask, pred, soft = random_crop(img,
                                                        mask,
                                                        self.random_state,
                                                        self.crop_size,
                                                        teacher_pred=pred)
                else:
                    img, mask = random_crop(img,
                                            mask,
                                            self.random_state,
                                            self.crop_size)

            image_batch.append(img)
            mask_batch.append(mask)

            if self.predictions:
                pred_batch.append(pred)

        return image_batch, mask_batch, batch_to_load, pred_batch


if __name__ == '__main__':
    batchiter = CAMVIDdataset(which_set="train",
                              minibatch_size=100,
                              data_format="bc01",
                              predictions=False)

    start = time.time()
    max_epochs = 2
    n_batches = batchiter.get_n_batches()

    batchiter = threaded_generator(batchiter, 10)

    for epoch in range(max_epochs):
        for mb in range(n_batches):
            image_batch, _, filenames, _ = batchiter.next()
            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))
