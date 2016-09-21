import numpy as np
import os
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
from dataset_loaders.utils_parallel_loader import natural_keys


floatX = 'float32'

"""
To prepare the dataset with the correct class mapping we use the scritps
provided by the authors https://github.com/mcordts/cityscapesScripts

NOTE: if we want to change the class mapping we can modify
cityscapesscripts/helpers/labels.py
and run the script cityscapesscripts/preparation/createTrainIdLabelImgs.py
in order to generate the new ground truth images.
Then we need to modify the dictionaries "_cmap" and "_mask_labels"
in the definition of CityscapesDataset.

id = -1  means that we do not consider the class when we create the gt
id = 255 means that the class is considered as void/unlabeled

TODO: before to submit the results to the evaluation server we need to
remap the training ID classes to the original class mapping from 0 to 33.

"""


class CityscapesDataset(ThreadedDataset):
    name = 'cityscapes'
    nclasses = 19
    debug_shape = (512, 512, 3)

    # optional arguments
    # data_shape = (512, 512, 3)

    GT_classes = range(nclasses)
    GT_classes = GT_classes + [255]

    _void_labels = [255]

    _cmap = {
        # 255 , (0, 0, 0),      # unlabeled
        # 255 , (0, 0, 0),      # ego vehicle
        # 255 : (0, 0, 0),      # rectification border
        # 255 : (0, 0, 0),      # out of roi
        # 255 : (0, 0, 0),      # static
        # 255 : (111, 74,  0),  # dynamic
        # 255 : (81,  0, 81),   # ground
        0: (128, 64, 128),      # road
        1: (244, 35, 232),      # sidewalk
        # 255: (250,170,160),   # parking
        # 255: (230,150,140),   # rail track
        2: (70, 70, 70),        # building
        3: (102, 102, 156),     # wall
        4: (190, 153, 153),     # fence
        # 255: (180,165,180),   # guard rail
        # 255: (150,100,100),   # bridge
        # 255: (150,120, 90),   # tunnel
        5: (153, 153, 153),     # pole
        # 255: (153,153,153),   # polegroup
        6: (250, 170, 30),      # traffic light
        7: (220, 220,  0),      # traffic sign
        8: (107, 142, 35),      # vegetation
        9: (152, 251, 152),     # terrain
        10: (0, 130, 180),      # sky
        11: (220, 20, 60),      # person
        12: (255, 0,  0),       # rider
        13: (0, 0, 142),        # car
        14: (0, 0, 70),         # truck
        15: (0, 60, 100),       # bus
        # 255: (0,  0, 90),     # caravan
        # 255: (0,  0,110),     # trailer
        16: (0, 80, 100),       # train
        17: (0, 0, 230),        # motorcycle
        18: (119, 11, 32),      # bicycle
        19: (0, 0, 0),          # void
        # -1: (0, 0, 142)       # license plate
        }

    _mask_labels = {
        # 0: 'unlabeled',
        # 1: 'ego vehicle',
        # 2: 'rectification border',
        # 3: 'out of roi',
        # 4: 'static',
        # 5: 'dynamic',
        # 6: 'ground',
        0: 'road',
        1: 'sidewalk',
        # 9: 'parking',
        # 10: 'rail track',
        2: 'building',
        3: 'wall',
        4: 'fence',
        # 14: 'guard rail',
        # 15: 'bridge',
        # 16: 'tunnel',
        5: 'pole',
        # 18: 'polegroup',
        6: 'traffic light',
        7: 'traffic sign',
        8: 'vegetation',
        9: 'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        # 29: 'caravan',
        # 30: 'trailer',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        19: 'void'
        # -1: 'license plate'
    }

    _filenames = None

    @property
    def filenames(self):
        if self._filenames is None:
            self._filenames = []
            # Get file names for this set
            for root, dirs, files in os.walk(self.image_path):
                for name in files:
                        self._filenames.append(os.path.join(
                          root[-root[::-1].index('/'):], name))

            self._filenames.sort(key=natural_keys)

            # Note: will get modified by prefix_list
        return self._filenames

    def __init__(self, which_set='train', with_filenames=False, *args,
                 **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.with_filenames = with_filenames
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'cityscapes')

        self.sharedpath = '/data/lisa/exp/visin/_datasets/cityscapes'

        if self.which_set == "train":
            self.image_path = os.path.join(self.path,
                                           "leftImg8bit_trainvaltest",
                                           "leftImg8bit",
                                           "train")
            self.mask_path = os.path.join(self.path,
                                          "gtFine_trainvaltest",
                                          "gtFine",
                                          "train")
        elif self.which_set == "val":
            self.image_path = os.path.join(self.path,
                                           "leftImg8bit_trainvaltest",
                                           "leftImg8bit",
                                           "val")
            self.mask_path = os.path.join(self.path,
                                          "gtFine_trainvaltest",
                                          "gtFine",
                                          "val")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path,
                                           "leftImg8bit_trainvaltest",
                                           "leftImg8bit",
                                           "test")
            self.mask_path = os.path.join(self.path,
                                          "gtFine_trainvaltest",
                                          "gtFine",
                                          "test")
            self.has_GT = False

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(CityscapesDataset, self).__init__(*args, **kwargs)

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
            max_num_sequences = video_length - seq_length + 1
            if (not self.seq_length or not self.seq_per_video or
                    self.seq_length >= video_length):
                # Use all possible frames
                for el in [(prefix, f) for f in frames[
                        :max_num_sequences:self.seq_length - self.overlap]]:
                    sequences.append(el)
            else:
                if max_num_sequences < seq_per_video:
                    # If there are not enough frames, cap seq_per_video to
                    # the number of available frames
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, seq_length,
                                          prefix, video_length))
                    seq_per_video = max_num_sequences

                if self.overlap != self.seq_length - 1:
                    raise('Overlap other than seq_length - 1 is not '
                          'implemented')

                # pick `seq_per_video` random indexes between 0 and
                # (video length - sequence length)
                first_frame_indexes = self.rng.permutation(range(
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
        from skimage import io
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
            img = img.astype(floatX) / 255.
            X.append(img)
            F.append(frame)

            if self.has_GT:
                mask_filename = frame.replace("leftImg8bit",
                                              "gtFine_labelTrainIds")
                mask = io.imread(os.path.join(self.mask_path, mask_filename))
                mask = mask.astype('int32')
                Y.append(mask)

        X = np.array(X)
        Y = np.array(Y)
        F = np.array(F)

        if self.with_filenames:
            return X, Y, F
        else:
            return X, Y


def test3():
    trainiter = CityscapesDataset(
        which_set='val',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=False,
        get_01c=True,
        use_threads=True,
        nthreads=8)
    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = 500
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))
    trainiter.get_cmap()
    max_epochs = 5

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            start = time.time()
            train_group = trainiter.next()

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1] == 224
            assert train_group[0].shape[2] == 224
            assert train_group[0].shape[3] == 3
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1

            if trainiter.has_GT:
                assert train_group[1].shape[0] <= train_batch_size
                assert train_group[1].shape[1] == 224
                assert train_group[1].shape[2] == 224

                if trainiter.get_one_hot:
                    assert train_group[1].ndim == 4
                    assert train_group[1].shape[3] == nclasses
                else:
                    assert train_group[1].ndim == 3

            # time.sleep approximates running some model
            time.sleep(0.1)
            stop = time.time()
            tot = stop - start
            print("Threaded time: %s" % (tot))
            print("Minibatch %s" % str(mb))
        print('ended epoch --> should reset!')
        time.sleep(2)


def test1():
    d = CityscapesDataset(
        which_set='train',
        batch_size=5,
        seq_per_video=4,
        seq_length=0,
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


def test2():
    d = CityscapesDataset(
        which_set='train',
        with_filenames=True,
        batch_size=5,
        seq_per_video=0,
        seq_length=10,
        overlap=10,
        get_one_hot=True,
        crop_size=(224, 224))
    for i, _ in enumerate(range(d.epoch_length)):
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()
        sh = image_group[0].shape
        print(image_group[2])
        if sh[1] != 2:
            raise RuntimeError()


if __name__ == '__main__':
    test3()
