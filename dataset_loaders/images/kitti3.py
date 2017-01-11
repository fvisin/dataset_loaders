import os
import time

import numpy as np
from PIL import Image

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class KITTIdataset3(ThreadedDataset):
    name = 'kitti3'
    non_void_nclasses = 11
    path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'kitti3')
    sharedpath = '/data/lisatmp4/romerosa/datasets/kitti3/'
    _void_labels = [11]

    # mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
    #    'float32')
    mean = [0.35675976, 0.37380189, 0.3764753]
    std = [0.32064945, 0.32098866, 0.32325324]
    _cmap = {
        0: (128, 128, 128),    # Sky
        1: (128, 0, 0),        # Building
        2: (128, 64, 128),     # Road
        3: (0, 0, 192),        # Sidewalk
        4: (64, 64, 128),      # Fence
        5: (128, 128, 0),      # Vegetation
        6: (192, 192, 128),    # Pole
        7: (64, 0, 128),       # Car
        8: (192, 128, 128),    # Sign
        9: (64, 64, 0),        # Pedestrian
        10: (0, 128, 192),     # Cyclist
        11: (255, 255, 255)    # void
    }
    _mask_labels = {0: 'Sky', 1: 'Building', 2: 'Pole', 3: 'Road',
                    4: 'Sidewalk', 5: 'Vegetation', 6: 'Sign', 7: 'Fence',
                    8: 'Car', 9: 'Pedestrian', 10: 'Cyclist', 11: 'Void'}

    _filenames = None

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.png")
            file_names = glob.glob(file_pattern)
            # print (str(file_names))

            # Get raw filenames from file names list
            for file_name in file_names:
                path, file_name = os.path.split(file_name)
                file_name, ext = os.path.splitext(file_name)
                filenames.append(file_name)
                # print (file_name)

            # Save the filenames list
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set="train", *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        if self.which_set not in ("train", "val", 'test', 'trainval'):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        if self.which_set == 'train':
            set_folder = 'Training_00/'
        elif self.which_set == 'val':
            set_folder = 'valid/'
        elif self.which_set == 'test':
            set_folder = 'Validation_07/'
        elif self.which_set == 'trainval':
            set_folder = 'trainval/'
        else:
            raise ValueError

        self.image_path = os.path.join(self.path, set_folder, "RGB")
        self.mask_path = os.path.join(self.path, set_folder, "GT_ind")

        super(KITTIdataset3, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset."""
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

        for prefix, img_name in sequence:
            # Load image
            img = io.imread(os.path.join(self.image_path, img_name + ".png"))
            img = img.astype(floatX) / 255.

            # Load mask
            mask = np.array(Image.open(
                    os.path.join(self.mask_path, img_name + ".png")))
            mask = mask.astype('int32')

            # Add to minibatch
            image_batch.append(img)
            mask_batch.append(mask)
            filename_batch.append(img_name)

        ret = {}
        ret['data'] = np.array(image_batch)
        ret['labels'] = np.array(mask_batch)
        ret['subset'] = prefix
        ret['filenames'] = np.array(filename_batch)
        return ret

        # Add to minibatch
        image_batch.append(img)
        mask_batch.append(mask)
        if self.with_filenames:
            filename_batch.append(img_name)

        image_batch = np.array(image_batch)
        mask_batch = np.array(mask_batch)
        filename_batch = np.array(filename_batch)

        if self.with_filenames:
            return image_batch, mask_batch, filename_batch
        else:
            return image_batch, mask_batch


def test():
    trainiter = KITTIdataset3(
        which_set='train',
        batch_size=10,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        use_threads=True)

    validiter = KITTIdataset3(
        which_set='valid',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_one_hot=True,
        return_01c=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nbatches = trainiter.nbatches
    print("Train %d" % (train_nsamples))

    valid_nsamples = validiter.nsamples
    print("Valid %d" % (valid_nsamples))

    # Simulate training
    max_epochs = 2
    start_training = time.time()
    for epoch in range(max_epochs):
        start_epoch = time.time()
        for mb in range(nbatches):
            start_batch = time.time()
            trainiter.next()

            print("Minibatch {}: {} seg".format(mb, (time.time() -
                                                     start_batch)))
        print("Epoch time: %s" % str(time.time() - start_epoch))
    print("Training time: %s" % str(time.time() - start_training))


def run_tests():
    test()


if __name__ == '__main__':
    run_tests()
