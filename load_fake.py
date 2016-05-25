import numpy as np
import os
from skimage import io
import time

from theano import config

from parallel_loader import ThreadedDataset
from utils_parallel_loader import get_frame_size


floatX = config.floatX


class FakeDataset(ThreadedDataset):
    nclasses = 9
    # wtf, sky, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    cmap = np.array([
        (255, 128, 0),      # wtf
        (255, 0, 0),        # sky (red)
        (0, 130, 180),      # ground (blue)
        (0, 255, 0),        # solid (buildings, etc) (green)
        (255, 255, 0),      # porous (yellow)
        (120, 0, 255),      # cars
        (255, 0, 255),      # humans (purple)
        (160, 160, 160),    # vert mix
        (64, 64, 64)])      # main mix
    cmap = cmap / 255.
    labels = ('wtf', 'sky', 'ground', 'solid', 'porous',
              'cars', 'humans', 'vert mix', 'gen mix')

    def __init__(self, which_set='train', threshold_masks=False,
                 seq_per_video=20, sequence_length=20, crop_size=None,
                 video_indexes=range(1, 64), split=0.75, *args, **kwargs):
        kwargs.update({'shuffle_at_each_epoch': False})
        self.video_indexes = [4]
        self.seq_per_video = 2
        self.sequence_length = sequence_length
        if crop_size and tuple(crop_size) == (0, 0):
            crop_size = None
        self.crop_size = crop_size
        self.threshold_masks = threshold_masks
        self.path = '/home/michal/data/GATECH/Images'

        print('FAKE DATASET')
        # set dataset specific arguments
        kwargs.update({
            'is_one_hot': False,
            'nclasses': FakeDataset.nclasses,
            'data_dim_ordering': 'tf',
            'shuffle_at_each_epoch': False})

        super(FakeDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        # sequences = [(4, 0), (4, 19)]
        sequences = [(4, 0)]

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
            height, width = get_frame_size(self.path, video_index)
            if not self.crop_size:
                top = 0
                left = 0
            else:
                top = np.random.randint(height - self.crop_size[0])
                left = np.random.randint(width - self.crop_size[1])
            top_left = [top, left]
            top_left = [44, 140]

            sequence_original, sequence_ground_truth = self.load_sequence(
                video_index,
                first_frame_index,
                top_left)

            X.append(sequence_original)
            Y.append(sequence_ground_truth)

        # if self.crop_size == 'no_crop' and len(to_load) > 1:
        #     # Work around Keras assumptions with list inputs
        #     # see https://github.com/fchollet/keras/issues/2539
        #     X = [X]
        #     Y = [Y]

        return np.array(X), np.array(Y)
        # return [X], [Y]  # np.array(X), np.array(Y)
        # return {'l_in': X}, {'y': Y}

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
            else:
                mask = mask - 1  # classes go from 1 to 63

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            # TODO : raise an error when top_left_corner + crop_size is
            # out of bound.
            sequence_image.append(img)
            sequence_ground_truth.append(mask)

        return np.array(sequence_image), np.array(sequence_ground_truth)


if __name__ == '__main__':
    d = FakeDataset(
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
