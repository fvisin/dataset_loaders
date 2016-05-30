import numpy as np
import os
from skimage import io
import time

from theano import config

from ..parallel_loader import ThreadedDataset


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
    name = 'camvid'
    input_shape = (360, 480, 3)
    nclasses = 12
    void_labels = [11]
    cmap = np.array([
        (255, 128, 0),    # sky
        (255, 0, 0),      # building
        (0, 130, 180),    # column_pole
        (0, 255, 0),      # road
        (255, 255, 0),    # sidewalk
        (120, 0, 255),    # tree
        (255, 0, 255),    # sign
        (160, 160, 160),  # fence
        (64, 64, 64),     # car
        (0, 0, 255),      # pedestrian
        (128, 128, 128),  # byciclist
        (0, 0, 0)])       # void
    cmap = cmap / 255.
    labels = ('sky', 'building', 'column_pole', 'road', 'sidewalk',
              'tree', 'sign', 'fence', 'car', 'pedestrian', 'byciclist',
              'void')

    def __init__(self, which_set='train', with_filenames=False, *args,
                 **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.with_filenames = with_filenames
        self.path = "./datasets/camvid/segnet/"

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

        super(CamvidDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length

        prefix_list = np.unique(np.array([el[:6] for el in self.filenames]))

        self.video_length = {}
        # cycle through the different videos
        for prefix in prefix_list:
            seq_per_video = self.seq_per_video
            frames = [el for el in self.filenames if prefix in el]
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
                first_frame_indexes = self.rng.random.permutation(range(
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

            X.append(img)
            Y.append(mask)
            F.append(frame)

        if self.with_filenames:
            return np.array(X), np.array(Y), np.array(F)
        else:
            return np.array(X), np.array(Y)


if __name__ == '__main__':
    d = CamvidDataset(
        which_set='train',
        batch_size=5,
        seq_per_video=4,
        seq_length=20,
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
