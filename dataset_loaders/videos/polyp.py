import os
import time

import numpy as np
from skimage import io
from theano import config

import dataset_loaders
from ..parallel_loader import ThreadedDataset
from ..utils_parallel_loader import get_frame_size, get_video_size


floatX = config.floatX


class PolypDataset(ThreadedDataset):
    name = 'polyp'
    nclasses = 2

    def __init__(self,
                 threshold_masks=False, seq_per_video=200,
                 sequence_length=20, crop_size=(224, 224),
                 video_indexes=range(20), *args, **kwargs):
        self.video_indexes = video_indexes
        self.seq_per_video = seq_per_video
        self.sequence_length = sequence_length
        if crop_size and tuple(crop_size) == (0, 0):
            crop_size = None
        self.crop_size = crop_size
        self.threshold_masks = threshold_masks
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets',
            'polyp_video_frames', 'Images')
        self.sharedpath = ('/data/lisatmp4/dejoieti/data/data_colo/' +
                           'polyp_video_frames/Images')

        super(PolypDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []

        _, lengths = get_video_size(self.path)
        for video_index in self.video_indexes:
            video_length = lengths[video_index]

            seq_per_video = self.seq_per_video
            if video_length - self.sequence_length < seq_per_video:
                print("/!\ Warning : the video_length - sequence_length is"
                      " smaller than the number_of_sequences_per_video")
                seq_per_video = video_length - self.sequence_length
            first_frame_indexes = np.random.permutation(range(
                video_length-self.sequence_length))[0:seq_per_video]

            for i in first_frame_indexes:
                sequences.append((video_index, i))

        return np.array(sequences)

    def fetch_from_dataset(self, to_load):
        X = []
        Y = []
        for video_index, first_frame_index in to_load:
            height, width = get_frame_size(self.path, video_index)
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

        for frame_index in range(
                first_frame_index, first_frame_index + self.sequence_length):
            filename = str(video_index) + "_" + str(frame_index) + ".tiff"
            img = io.imread(os.path.join(im_path, filename))
            mask = io.imread(os.path.join(mask_path, filename))

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

            img = np.array(img).astype(floatX) / 255.

            # TODO : raise an error when top_left_corner + crop_size is
            # out of bound.
            sequence_image.append(img)
            sequence_ground_truth.append(mask)

        return np.array(sequence_image), np.array(sequence_ground_truth)


if __name__ == '__main__':
    d = PolypDataset(
        seq_per_video=50,
        sequence_length=2,
        crop_size=(224, 224),
        is_one_hot=False,
        n_classes=2)
    start = time.time()
    n_minibatches_to_run = 100
    itr = 1
    while True:
        image_group = d.next()
        # time.sleep approximates running some model
        time.sleep(1)
        stop = time.time()
        tot = stop - start
        print("Threaded time: %s" % (tot))
        print("Minibatch %s" % str(itr))
        print("Time ratio (s per minibatch): %s" % (tot / float(itr)))
        itr += 1
        # test
        if itr >= n_minibatches_to_run:
            break
