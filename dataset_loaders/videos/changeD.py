import time
import os

import numpy as np
from skimage import io
from theano import config

import dataset_loaders
from ..parallel_loader import ThreadedDataset
from ..utils_parallel_loader import get_frame_size, get_video_size


floatX = config.floatX
class_ids = {'0': 0, '50': 1, '85': 4, '170': 2, '255': 3}


class ChangeDetectionDataset(ThreadedDataset):
    name = 'change_detection'
    nclasses = 5
    _is_one_hot = False
    _is_01c = True
    debug_shape = (360, 640, 3)  # TODO MODIFY WITH AN APPROPRIATE SHAPE

    void_labels = [4]

    def __init__(self, which_set='train', threshold_masks=False,
                 seq_per_video=20, sequence_length=20, crop_size=None,
                 video_indexes=range(0, 53), split=0.75, *args, **kwargs):

        self.which_set = which_set
        self.threshold_masks = threshold_masks
        self.seq_per_video = seq_per_video
        self.sequence_length = sequence_length
        if crop_size and tuple(crop_size) == (0, 0):
            crop_size = None
        self.crop_size = crop_size
        self.video_indexes = list(video_indexes)
        self.split = split
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'Change_Detection',
            'Images')
        self.sharedpath = ('/data/lisatmp4/dejoieti/data/Change_Detection/'
                           'Images')

        if which_set in ['train', 'valid']:
            rng = np.random.RandomState(16)
            rng.shuffle(self.video_indexes)
            len_train = int(split*len(self.video_indexes))
            if which_set == 'train':
                self.video_indexes = self.video_indexes[:len_train]
            else:
                self.video_indexes = self.video_indexes[len_train:]
        elif which_set in ['test', 'test_all']:
            print('No mask for the test set!!')
        else:
            raise RuntimeError('unknown set')

        super(ChangeDetectionDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_per_video = self.seq_per_video
        sequence_length = self.sequence_length

        _, self.lengths = get_video_size(self.path)
        # cycle through the different videos
        for video_index in self.video_indexes:
            video_length = self.lengths[video_index]

            # Fill sequences with (video_idx, frame_idx)
            if (self.sequence_length is None or
                    self.which_set == 'test_all' or
                    self.sequence_length >= video_length):
                sequences.append((video_index, 0))
            else:
                if video_length - sequence_length < seq_per_video:
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, sequence_length,
                                          video_index, video_length))
                    seq_per_video = video_length - self.sequence_length

                # pick `seq_per_video` random indexes between 0 and
                # (video length = sequence length)
                first_frame_indexes = np.random.permutation(range(
                    video_length - sequence_length))[0:seq_per_video]

                for i in first_frame_indexes:
                    sequences.append((video_index, i))

        return np.array(sequences)

    def fetch_from_dataset(self, to_load):
        """
        This returns a list of sequences or clips or batches.
        """
        X = []
        Y = []
        for el in to_load:
            if el is None:
                continue
            video_index, first_frame_index = el
            height, width = get_frame_size(self.path, video_index, 'jpg')
            if not self.crop_size:
                top = 0
                left = 0
            else:
                top = np.random.randint(height - self.crop_size[0])
                left = np.random.randint(width - self.crop_size[1])
            top_left = [top, left]

            sequence_original, sequence_ground_truth = self.load_sequence(
                video_index,
                first_frame_index,
                top_left)

            X.append(sequence_original)
            Y.append(sequence_ground_truth)

        return np.array(X), np.array(Y)

    def load_sequence(
            self,
            video_index,
            first_frame_index,
            top_left):
        """
        Loading a sequence.

        Auxiliary function which loads a squence of frames width
        the corresponding ground truth.

        :video_index: index of the video in which the sequence is taken
        :crop_size: (height, width) or None, frame cropping size
        :top_left: (top, left), top left corner for cropping purposes
        :path: path of the dataset
        """
        # TODO : take rgb, 0, 1 and 0, 1, rgb into account

        im_path = os.path.join(self.path, 'Original')
        mask_path = os.path.join(self.path, 'Ground_Truth')

        sequence_image = []
        sequence_ground_truth = []

        video_length = self.lengths[video_index]
        if (self.sequence_length is None or
                self.which_set == 'test_all'or
                self.sequence_length > video_length):
            sequence_length = video_length
        else:
            sequence_length = self.sequence_length

        for frame_index in range(
                    first_frame_index, first_frame_index + sequence_length):
            filename = str(video_index) + "_" + str(frame_index) + ".jpg"
            img = io.imread(os.path.join(im_path, filename))
            mask = io.imread(os.path.join(mask_path, filename[:-4]+".png"))

            if self.crop_size:
                img = img[top_left[0]:top_left[0]+self.crop_size[0],
                          top_left[1]:top_left[1]+self.crop_size[1]]
                mask = mask[top_left[0]:top_left[0]+self.crop_size[0],
                            top_left[1]:top_left[1]+self.crop_size[1]]

            if self.threshold_masks:
                masklow = mask < 128
                maskhigh = mask >= 128
                mask[masklow] = 0
                mask[maskhigh] = 1

                assert 0. <= np.min(mask) <= 1
                assert 0 <= np.max(mask) <= 1

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            for gt_id, c_id in class_ids.iteritems():
                mask[mask == int(gt_id)] = c_id

            # TODO : raise an error when top_left_corner + crop_size is
            # out of bound.
            sequence_image.append(img)
            sequence_ground_truth.append(mask)

        return np.array(sequence_image), np.array(sequence_ground_truth)


if __name__ == '__main__':
    d = ChangeDetectionDataset(
        which_set='train',
        minibatch_size=5,
        seq_per_video=4,
        sequence_length=20,
        crop_size=(224, 224))
    start = time.time()
    n_minibatches_to_run = 1000
    itr = 1
    while True:
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()
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
