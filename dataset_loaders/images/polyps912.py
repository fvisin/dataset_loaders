import numpy as np
import os
import time
from getpass import getuser

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
floatX = 'float32'


class Polyps912Dataset(ThreadedDataset):
    # Set basic info from the dataset
    name = 'Polyps912'
    nclasses = 3
    debug_shape = (384, 288, 3)
    data_shape = (384, 288, 3)
    mean = np.asarray([0., 0., 0.]).astype('float32')
    std = 1.
    _void_labels = [3]
    _cmap = {
        0: (128, 128, 0),       # Background
        1: (128, 0, 0),         # Polyp
        2: (128, 64, 128),      # Lumen
        3: (0, 0, 0),           # Void
        }
    _mask_labels = {0: 'Background', 1: 'Polyp', 2: 'Lumen', 3: 'Void'}
    _filenames = None
    labels = ('Background', 'Polyp', 'Lumen')

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.bmp")
            file_names = glob.glob(file_pattern)
            # print (str(file_names))

            # Get raw filenames from file names list
            for file_name in file_names:
                path, file_name = os.path.split(file_name)
                file_name, ext = os.path.splitext(file_name)
                raw_name = file_name.strip()
                filenames.append(file_name)
                # print (file_name)

            # Save the filenames list
            self._filenames = filenames
        return self._filenames

    def __init__(self, which_set='train', with_filenames=False,
                 *args, **kwargs):

        # Get dataset path
        usr = getuser()
        self.path = '/Tmp/'+usr+'/datasets/polyps912/'
        self.sharedpath = '/data/lisa/exp/vazquezd/datasets/polyps_split5/CVC-912/'

        # Put the which_set in the cannonical form ("training", "validation", "testing")
        self.which_set = which_set
        self.which_set = "train" if self.which_set in ("train", "training") else self.which_set
        self.which_set = "valid" if self.which_set in ("val", "valid", "validation") else self.which_set
        self.which_set = "test" if self.which_set in ("test", "testing") else self.which_set
        if self.which_set not in ("train", "valid", "test"):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)

        # Define the images and mask paths
        self.image_path = os.path.join(self.path, self.which_set, 'images')
        self.mask_path = os.path.join(self.path, self.which_set, 'masks4')

        # Other stuff
        self.with_filenames = with_filenames

        super(Polyps912Dataset, self).__init__(*args, **kwargs)

    def get_names(self):

        # Limit to the number of videos we want
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        video_length = len(self.filenames)
        max_num_sequences = video_length - seq_length + 1
        if (not self.seq_length or not self.seq_per_video or
                self.seq_length >= video_length):
            # Use all possible frames
            sequences = self.filenames[:max_num_sequences]
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
                sequences.append(self.filenames[i])

        # Return images
        return np.array(sequences)

    def load_sequence(self, img_name):
        from skimage import io
        image_batch, mask_batch = [], []
        pred_batch, filename_batch = [], []

        if self.seq_length != 1:
            raise NotImplementedError()

        # Load image
        img = io.imread(os.path.join(self.image_path, img_name + ".bmp"))
        img = img.astype(floatX) / 255.
        # print 'Image shape: ' + str(img.shape)

        # Load mask
        import numpy as np
        mask = np.array(io.imread(os.path.join(self.mask_path, img_name + ".tif")))
        mask = mask.astype('int32')
        # print 'Mask shape: ' + str(mask.shape)

        # THIS IS FOR DEBUG PROUPOSES
        # # Imports
        # from skimage.color import label2rgb, rgb2gray, gray2rgb
        # from skimage import img_as_float
        # import numpy as np
        # import scipy.misc
        # import seaborn as sns
        #
        # # Converts a label mask to RGB to be shown
        # def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
        #     output = np.zeros(labels.shape + (3,), dtype=np.float64)
        #     for i in range(len(colors)):
        #         if i != bglabel:
        #             output[(labels == i).nonzero()] = colors[i]
        #     if bglabel is not None:
        #         output[(labels == bglabel).nonzero()] = bg_color
        #     return output
        #
        #
        # # Converts a label mask to RGB to be shown and overlaps over an image
        # def my_label2rgboverlay(labels, colors, image, bglabel=None,
        #                         bg_color=(0., 0., 0.), alpha=0.2):
        #     image_float = gray2rgb(img_as_float(rgb2gray(image)))
        #     label_image = my_label2rgb(labels, colors, bglabel=bglabel,
        #                                bg_color=bg_color)
        #     output = image_float * alpha + label_image * (1 - alpha)
        #     return output
        #
        #
        # # Save image
        # def save_img(img, mask, fname, color_map, void_label):
        #     # img = img / 255.
        #
        #     label_mask = my_label2rgboverlay(mask,
        #                                      colors=color_map,
        #                                      image=img,
        #                                      bglabel=void_label,
        #                                      alpha=0.2)
        #
        #     combined_image = np.concatenate((img, label_mask), axis=1)
        #     scipy.misc.toimage(combined_image).save(fname)
        #
        # color_map = sns.hls_palette(7)
        # fname = './' + img_name + '.png'
        # save_img(img, mask, fname, color_map, self.get_void_labels()[0])
        # exit()

        # Add to minibatch
        image_batch.append(img)
        mask_batch.append(mask)
        if self.with_filenames:
            filename_batch.append(img_name)

        ret = [np.array(image_batch), np.array(mask_batch)]

        other = []
        if self.with_filenames:
            other += [np.array(filename_batch)]

        # return image_batch, mask_batch, batch_to_load, pred_batch
        return tuple(ret + other)


def test(max_epochs=2):
    trainiter = Polyps912Dataset(
        which_set='train',
        batch_size=10,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=True)

    validiter = Polyps912Dataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        crop_size=None,
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    testiter = Polyps912Dataset(
        which_set='test',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        crop_size=None,
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    # Get number of classes
    nclasses = trainiter.get_n_classes()
    print ("N classes: " + str(nclasses))
    void_label = trainiter.get_void_labels()
    print ("Void label: " + str(void_label))

    # Training info
    train_nsamples = trainiter.nsamples
    train_batch_size = trainiter.get_batch_size()
    train_nbatches = trainiter.get_n_batches()
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(train_nsamples,
                                                                     train_batch_size,
                                                                     train_nbatches))

    # Validation info
    valid_nsamples = validiter.nsamples
    valid_batch_size = validiter.get_batch_size()
    valid_nbatches = validiter.get_n_batches()
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(valid_nsamples,
                                                                 valid_batch_size,
                                                                 valid_nbatches))

    # Testing info
    test_nsamples = testiter.nsamples
    test_batch_size = testiter.get_batch_size()
    test_nbatches = testiter.get_n_batches()
    print("Test n_images: {}, batch_size: {}, n_batches: {}".format(test_nsamples,
                                                                 test_batch_size,
                                                                 test_nbatches))

    for epoch in range(max_epochs):
        epoch_start = time.time()
        for mb in range(train_nbatches):
            mb_start = time.time()
            train_batch = trainiter.next()
            print("Minibatch {}: {:.3f} seg".format(mb, time.time() - mb_start))
        print("End epoch {}: {:.3f} seg".format(epoch, time.time() - epoch_start))


if __name__ == '__main__':
    test()
