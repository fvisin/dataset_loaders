import os
import numpy as np
from PIL import Image
from getpass import getuser
import time

from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class KITTIdataset2(ThreadedDataset):
    name = 'kitti'
    nclasses = 11
    debug_shape = (375, 500, 3)

    data_shape = (None, None, 3)
    # mean = np.asarray([122.67891434, 116.66876762, 104.00698793]).astype(
    #    'float32')
    std = 1.
    _void_labels = [11] # [255] (TODO: No void class???)
    GTclasses = range(11) + _void_labels

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
        10: (0, 128, 192)      # Cyclist
        # 255: (255, 255, 255)   # void
    }

    _mask_labels = {0: 'Sky', 1: 'Building', 2:  'Pole', 3:  'Road',
                    4:  'Sidewalk', 5:  'Vegetation', 6:  'Sign',
                    7:  'Fence', 8:  'Car', 9:  'Pedestrian', 10:
                    'Cyclist',
                    11: 'Void'}
                    # 255: 'void'}

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

    def __init__(self,
                 which_set="train",
                 with_filenames=False,
                 *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.with_filenames = with_filenames
        usr = getuser()
        # self.path = '/home/michal/KITTI2/'
        # self.sharedpath = '/home/michal/KITTI2/'
        self.path = '/Tmp/'+usr+'/datasets/KITTI2/'
        self.sharedpath = '/data/lisatmp4/romerosa/datasets/KITTI2'

        if self.which_set not in ("train", "val", "test"):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        if self.which_set == "train":
            set_folder = 'Training_00/'
        elif self.which_set == "test":
            set_folder = 'Validation_07/'
        elif self.which_set == 'val':
            set_folder = 'valid/'
        elif self.which_set == 'trainval':
            set_folder = 'trainval/'
        else:
            raise ValueError('Unknown set.')

        self.image_path = os.path.join(self.path, set_folder, "RGB")
        self.mask_path = os.path.join(self.path, set_folder, "GT_ind")
        print(self.image_path)
        print(self.mask_path)

        self.mu_camvid = [0.39068785, 0.40521392, 0.41434407]
        self.sigma_camvid = [0.29652068, 0.30514979, 0.30080369]

        self.mu_kitti = [0.35675976, 0.37380189, 0.3764753]
        self.sigma_kitti = [0.32064945, 0.32098866, 0.32325324]

        super(KITTIdataset2, self).__init__(*args, **kwargs)

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
        filename_batch = []

        if self.seq_length != 1:
            raise NotImplementedError()

        # Load image
        img = io.imread(os.path.join(self.image_path, img_name + ".png"))
        img = img.astype(floatX) / 255.

        if '_' in img_name:
            img -= self.mu_camvid
            img /= self.sigma_camvid
        else:
            img -= self.mu_kitti
            img /= self.sigma_kitti

        # Load mask
        mask = np.array(Image.open(
                os.path.join(self.mask_path, img_name + ".png")))
        mask = mask.astype('int32')

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
    trainiter = KITTIdataset2(
        which_set='train',
        batch_size=10,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    validiter = KITTIdataset2(
        which_set='valid',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.get_n_classes()
    nbatches = trainiter.get_n_batches()
    train_batch_size = trainiter.get_batch_size()
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


if __name__ == '__main__':
    test()
