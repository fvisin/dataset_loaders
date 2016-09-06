import numpy as np
import os
from scipy import interpolate
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


def displacement_vecs(std, grid_size):
    disp_x = np.reshape(
        map(int, np.random.randn(np.prod(np.asarray(grid_size)))*std),
        grid_size)
    disp_y = np.reshape(
        map(int, np.random.randn(np.prod(np.asarray(grid_size)))*std),
        grid_size)
    return disp_x, disp_y


def _elastic_def(image, fx, fy):
    imsize = image.shape[-2:]
    x = np.arange(imsize[0]-1)
    y = np.arange(imsize[1]-1)
    xx, yy = np.meshgrid(x, y)
    z_1 = fx(x, y).astype(int)
    z_2 = fy(x, y).astype(int)
    img = np.zeros(image.shape, dtype='uint8')
    if image.ndim == 3:  # multi-channel
        img[:, xx, yy] = image[:, np.clip(xx - z_1, 0, imsize[0]-1),
                               np.clip(yy - z_2, 0, imsize[1]-1)]
    if image.ndim == 2:  # one channel
        img[xx, yy] = image[np.clip(xx - z_1, 0, imsize[0]-1),
                            np.clip(yy - z_2, 0, imsize[1]-1)]
    return img


def batch_elastic_def(im_batch, disp_x, disp_y, grid_size=(3, 3),
                      interpol_kind='linear'):
    '''im_batch should have batch dimension
       even if it contains a single image'''
    imsize = im_batch.shape[-2:]
    x = np.linspace(0, imsize[0]-1, grid_size[0]).astype(int)
    y = np.linspace(0, imsize[1]-1, grid_size[1]).astype(int)
    fx = interpolate.interp2d(x, y, disp_x, kind=interpol_kind)
    fy = interpolate.interp2d(x, y, disp_y, kind=interpol_kind)
    if im_batch.ndim == 3 or im_batch.ndim == 4:
        im_elast_def = np.asarray([_elastic_def(im, fx, fy)
                                   for im in im_batch])
    if im_batch.ndim == 2:  # if no batch in im_batch
        im_elast_def = _elastic_def(im_batch, fx, fy)
    return im_elast_def


class IsbiEmStacksDataset(ThreadedDataset):
    name = 'isbi_em_stacks'
    nclasses = 2
    debug_shape = (512, 512, 1)

    # optional arguments
    data_shape = (512, 512, 1)
    _void_labels = []
    _cmap = {
        0: (0, 0, 0), # Non-membranes
        1: (255, 255, 255)} # Membranes
    _mask_labels = {0: 'Non-membranes', 1: 'Membranes'}

    _filenames = None


    def __init__(self, which_set='train', start=0, end=30,
                 elastic_deform=False, h_flipping=False, v_flipping=False,
                 shearing_range=0.0, *args, **kwargs):

        assert which_set in ["train", "test"]
        self.which_set = which_set

        assert start >= 0 and end <= 30 and start < end
        self.start = start
        self.end = end

        self.elastic_deform = elastic_deform
        self.h_flipping = h_flipping
        self.v_flipping = v_flipping
        self.shearing_range = shearing_range

        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'em_stacks')
        self.sharedpath = '/data/lisa/data/isbi_challenge_em_stacks/'

        if self.which_set == "train":
            self.image_path = os.path.join(self.path, "train-volume.tif")
            self.target_path = os.path.join(self.path, "train-labels.tif")
        elif self.which_set == "test":
            self.image_path = os.path.join(self.path, "test-volume.tif")
            self.target_path = None

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(IsbiEmStacksDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length

        self.video_length = {}
        prefix = ""
        video_length = self.end - self.start
        frames = range(video_length)

        seq_per_video = self.seq_per_video
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
                      "frames each but the dataset only has {} "
                      "frames".format(seq_per_video, seq_length, video_length))
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

        self.epoch_length = ((len(sequences) + self.batch_size - 1) /
                             self.batch_size)
        return np.array(sequences)

    def load_sequence(self, first_frame):
        """
        Load ONE clip/sequence

        Auxiliary function which loads a sequence of frames with
        the corresponding ground truth and potentially filenames.
        Returns images in [0, 1]
        """
        from PIL import Image

        X = []
        Y = []

        load_labels = self.target_path is not None

        prefix, first_frame_name = first_frame
        first_frame_name = int(first_frame_name)

        if (self.seq_length is None or
                self.seq_length > self.video_length[prefix]):
            seq_length = self.video_length[prefix]
        else:
            seq_length = self.seq_length

        start_idx = first_frame_name
        end_idx = start_idx + seq_length

        for i in range(start_idx, end_idx):

            idx = i + self.start

            imgs = Image.open(self.image_path)
            imgs.seek(idx)
            X.append(np.array(imgs)[:, :, None].astype("uint8"))

            if load_labels:
                targets = Image.open(self.target_path)
                targets.seek(idx)
                Y.append(np.array(targets) / 255)

        X = np.array(X)
        Y = np.array(Y)

        # If needed, apply elastic deformation
        if self.elastic_deform:
            disp_x, disp_y = displacement_vecs(10, (3, 3))
            X = batch_elastic_def(X[:,:,:,0], disp_x, disp_y)[:,:,:,None]
            Y = batch_elastic_def(Y, disp_x, disp_y)

        # If needed, apply flipping
        # X is in format B01C, Y is in format B01
        for i in range(len(X)):
            if self.h_flipping and np.random.random() < 0.5:
                X[i] = X[i,:,::-1]
                Y[i] = Y[i,:,::-1]
            if self.v_flipping and np.random.random() < 0.5:
                X[i] = X[i,::-1,:]
                Y[i] = Y[i,::-1,:]

        # If needed, apply shearing deformation
        if self.shearing_range > 0.0:
            from keras.preprocessing.image import (apply_transform,
                                                   transform_matrix_offset_center)

            height = X.shape[2]
            width = X.shape[3]

            for i in range(len(X)):

                shear = np.random.uniform(-self.shearing_range,
                                          self.shearing_range)
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                         [0, np.cos(shear), 0],
                                         [0, 0, 1]])

                transform_matrix = transform_matrix_offset_center(shear_matrix,
                                                                  height, width)
                X[i] = apply_transform(X[i], transform_matrix, 2, 'nearest', 0.)
                Y[i] = apply_transform(Y[i,:,:,None], transform_matrix, 2,
                                       'nearest', 0.)[:,:,0]

        X = X.astype("float32") / 255

        if load_labels:
            return X, Y
        else:
            return X


def test2():
    d = IsbiEmStacksDataset(
        which_set='train',
        with_filenames=True,
        batch_size=5,
        seq_per_video=0,
        seq_length=10,
        overlap=9,
        get_one_hot=True,
        crop_size=(224, 224),
        elastic_deform=True)

    for i, _ in enumerate(range(d.epoch_length)):
        image_group = d.next()
        if image_group is None:
            raise NotImplementedError()
        sh = image_group[0].shape
        print sh
        if sh[1:] != (10, 1, 224, 224):
            raise RuntimeError()


if __name__ == '__main__':
    test2()
