import os
import time

import numpy as np
from skimage import io
from theano import config

import dataset_loaders
from ..parallel_loader import ThreadedDataset
from ..utils_parallel_loader import natural_keys

floatX = config.floatX


class PolypVideoDataset(ThreadedDataset):
    name = 'colonoscopyVideos'
    nclasses = 2
    void_labels = []
    _is_one_hot = False
    _is_01c = True
    debug_shape = (288, 384, 3)

    cmap = np.array([
        (255, 255, 255),  # polyp
        (0, 0, 0)])  # background
    cmap = cmap / 255
    labels = ('polyp', 'background')

    def __init__(self,
                 which_set='train',
                 threshold_masks=True,
                 with_filenames=False,
                 split=.75, *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.threshold_masks = threshold_masks
        self.with_filenames = with_filenames

        self.void_labels = PolypVideoDataset.void_labels

        # Prepare data paths
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'POLYP_VIDEOS')
        self.sharedpath = ('/data/lisatmp4/dejoieti/data/data_colo/')
        if self.which_set == "train" or self.which_set == "val":
            self.image_path = os.path.join(self.path,
                                           'polyp_video_frames',
                                           'Images', 'Original')
            self.mask_path = os.path.join(self.path,
                                          'polyp_video_frames',
                                          'Images', 'Ground_Truth')
            self.split = split if self.which_set == "train" else (1 - split)
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path,
                                           'polyp_video_frames_test',
                                           'noBlackBand',
                                           'Original')
            self.mask_path = os.path.join(self.path,
                                          'poly_video_frames_test',
                                          'noBlackBand',
                                          'Ground_Truth')
            self.split = 1.

        # Get file names for this set
        self.filenames = os.listdir(self.image_path)
        self.filenames.sort(key=natural_keys)

        super(PolypVideoDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length

        all_prefix_list = np.unique(np.array([el[:el.index('_')]
                                              for el in self.filenames]))

        nvideos = len(all_prefix_list)
        nvideos_set = int(nvideos*self.split)
        prefix_list = all_prefix_list[-nvideos_set:] \
            if self.which_set == "val" else all_prefix_list[:nvideos_set]

        # update filenames list
        self.filenames = [f for f in self.filenames if f[:f.index('_')]
                          in prefix_list]

        self.video_length = {}
        # cycle through the different videos
        for prefix in prefix_list:
            seq_per_video = self.seq_per_video
            new_prefix = prefix + '_'
            frames = [el for el in self.filenames if new_prefix in el and
                      el.index(prefix+'_') == 0]
            video_length = len(frames)
            self.video_length[prefix] = video_length

            # Fill sequences with (prefix, frame_idx)
            if (not self.seq_length or not self.seq_per_video or
                    self.seq_length >= video_length):
                # Use all possible frames
                for el in [(prefix, f) for f in frames]:
                    sequences.append(el)
            else:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                max_num_sequences = video_length - seq_length + 1
                if max_num_sequences < seq_per_video:
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, seq_length,
                                          prefix, video_length))
                    seq_per_video = max_num_sequences

                # pick `seq_per_video` random indexes between 0 and
                # (video length - sequence length)
                first_frame_indexes = np.random.permutation(range(
                    max_num_sequences))[0:seq_per_video]

                for i in first_frame_indexes:
                    sequences.append((prefix, frames[i]))

        return np.array(sequences)

    def load_sequence(self, first_frame):
        """
        Load ONE clip/sequence
        Auxiliary function which loads a sequence of frames with
        the corresponding ground truth and potentially filenames.
        Returns images in [0, 1]
        """
        X = []
        Y = []
        F = []

        prefix, first_frame_name = first_frame

        if (self.seq_length is None or
                self.seq_length > self.video_length[prefix]):
            seq_length = self.video_length[prefix]
        else:
            seq_length = self.seq_length

        start_idx = self.filenames.index(first_frame_name)
        for frame in self.filenames[start_idx:start_idx + seq_length]:
            img = io.imread(os.path.join(self.image_path, frame))
            mask = io.imread(os.path.join(self.mask_path, frame))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            if self.threshold_masks:
                masklow = mask < 127
                maskhigh = mask >= 127
                mask[masklow] = 0
                mask[maskhigh] = 1

            X.append(img)
            Y.append(mask)
            F.append(frame)

        if self.with_filenames:
            return np.array(X), np.array(Y), np.array(F)
        else:
            return np.array(X), np.array(Y)

    def get_void_labels(self):
        return self.void_labels

if __name__ == '__main__':
    trainiter = PolypVideoDataset(
        which_set='train',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        split=.75)
    validiter = PolypVideoDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        split=.75)
    train_nsamples = trainiter.get_n_samples()
    valid_nsamples = validiter.get_n_samples()

    print("Train %d, valid %d" % (train_nsamples, valid_nsamples))

    start = time.time()
    n_minibatches_to_run = 1000
    itr = 1
    while True:
        train_group = trainiter.next()
        valid_group = validiter.next()

        if train_group is None or valid_group is None:
            raise ValueError('.next() returned None!')
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        tot = stop - start
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        print("Tot time: %s" % (tot))
        itr += 1
        # test
        if itr >= n_minibatches_to_run:
            break
