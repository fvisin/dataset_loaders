import numpy as np
import os
from skimage import io
import time

from theano import config

from parallel_loader import ThreadedDataset


floatX = config.floatX


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


class CamvidDataset(ThreadedDataset):
    nclasses = 12
    void_labels = [11]
    cmap = np.array([
        (255, 128, 0),   # sky
        (255, 0, 0),     # building
        (0, 130, 180),   # column_pole
        (0, 255, 0),     # road
        (255, 255, 0),   # sidewalk
        (120, 0, 255),   # tree
        (255, 0, 255),   # sign
        (160, 160, 160), # fence
        (64, 64, 64),    # car
        (0, 0, 255),     # pedestrian
        (128, 128, 128), # byciclist
        (0, 0, 0)])      # void
    cmap = cmap / 255.
    labels = ('sky', 'building', 'column_pole', 'road', 'sidewalk',
              'tree', 'sign', 'fence', 'car', 'pedestrian', 'byciclist', 'void')

    def __init__(self, which_set='train', threshold_masks=False,
                 seq_per_video=20, sequence_length=20, crop_size=None,
                 video_indexes=range(1, 3), *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.threshold_masks = threshold_masks
        self.seq_per_video = seq_per_video
        self.sequence_length = sequence_length
        if crop_size and tuple(crop_size) == (0, 0):
            crop_size = None
        self.crop_size = crop_size
        self.video_indexes = list(video_indexes)

        self.path = "/data/lisa/exp/visin/datasets/camvid/segnet/"

        if self.which_set == "train":
            self.image_path = os.path.join(self.path, "train")
            self.mask_path = os.path.join(self.path, "trainannot")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.path, "val")
            self.mask_path = os.path.join(self.path, "valannot")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "test")
            self.mask_path = os.path.join(self.path, "testannot")

        # Get file names for this set and year
        filenames = []
        with open(self.path + self.which_set + ".txt") as f:
            for fi in f.readlines():
                raw_name = fi.strip()
                raw_name = raw_name.split("/")[4]
                raw_name = raw_name.strip()
                filenames.append(raw_name)
        self.filenames = filenames

        # set dataset specific arguments
        kwargs.update({
            'is_one_hot': False,
            'nclasses': CamvidDataset.nclasses,
            'data_dim_ordering': 'tf'})

        print(self.video_indexes)
        super(CamvidDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_per_video = self.seq_per_video
        sequence_length = self.sequence_length

        prefix_list = np.unique(np.array([el[:6] for el in self.filenames]))

        self.video_length = {}
        # cycle through the different videos
        for prefix in prefix_list:
            frames = [el for el in self.filenames if prefix in el]
            video_length = len(frames)
            self.video_length[prefix] = video_length

            # Fill sequences with (video_idx, frame_idx)
            if (self.sequence_length is None or
                    self.sequence_length >= video_length):
                sequences.append(frames[0])
            else:
                if video_length - sequence_length < seq_per_video:
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, sequence_length,
                                          prefix, video_length))
                    seq_per_video = video_length - self.sequence_length

                # pick `seq_per_video` random indexes between 0 and
                # (video length = sequence length)
                first_frame_indexes = np.random.permutation(range(
                    video_length - sequence_length))[0:seq_per_video]

                for i in first_frame_indexes:
                    sequences.append(frames[i])

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
            first_frame_name = el
            height, width = io.imread(os.path.join(self.image_path,
                                                   first_frame_name)).shape[:2]
            if not self.crop_size:
                top = 0
                left = 0
            else:
                top = np.random.randint(height - self.crop_size[0])
                left = np.random.randint(width - self.crop_size[1])
            top_left = [top, left]

            sequence_original, sequence_ground_truth = self.load_sequence(
                first_frame_name,
                top_left)

            X.append(sequence_original)
            Y.append(sequence_ground_truth)

        return np.array(X), np.array(Y)

    def load_sequence(
            self,
            first_frame_name,
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

        sequence_image = []
        sequence_ground_truth = []
        prefix = first_frame_name[:6]

        if (self.sequence_length is None or
                self.sequence_length > self.video_length[prefix]):
            sequence_length = self.video_length[prefix]
        else:
            sequence_length = self.sequence_length

        start_idx = self.filenames.index(first_frame_name)
        for frame in self.filenames[start_idx:start_idx + sequence_length]:
            img = io.imread(os.path.join(self.image_path, frame))
            mask = io.imread(os.path.join(self.mask_path, frame))

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

            # TODO : raise an error when top_left_corner + crop_size is
            # out of bound.
            sequence_image.append(img)
            sequence_ground_truth.append(mask)

        return np.array(sequence_image), np.array(sequence_ground_truth)


if __name__ == '__main__':
    d = CamvidDataset(
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
