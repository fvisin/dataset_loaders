import numpy as np
import os
import time

from scipy import interpolate

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
    non_void_nclasses = 2
    path = os.path.join(
        dataset_loaders.__path__[0], 'datasets', 'em_stacks')
    sharedpath = '/data/lisa/data/isbi_challenge_em_stacks/'
    _void_labels = []

    # optional arguments
    data_shape = (512, 512, 1)
    _cmap = {
        0: (0, 0, 0),  # Non-membranes
        1: (255, 255, 255)}  # Membranes
    _mask_labels = {0: 'Non-membranes', 1: 'Membranes'}

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
        """Return a dict of names, per prefix/subset."""
        return {'default': range(self.end - self.start)}

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        """
        from PIL import Image
        X = []
        Y = []
        F = []

        for prefix, idx in sequence:

            imgs = Image.open(self.image_path)
            imgs.seek(idx)
            imgs = np.array(imgs)[:, :, None].astype("uint8")

            if self.target_path is not None:
                targets = Image.open(self.target_path)
                targets.seek(idx)
                targets = np.array(targets) / 255

            X.append(imgs)
            Y.append(targets)
            F.append(idx)
        X = np.array(X)
        Y = np.array(Y)

        # If needed, apply elastic deformation
        if self.elastic_deform:
            disp_x, disp_y = displacement_vecs(10, (3, 3))
            X = batch_elastic_def(X[:, :, :, 0], disp_x, disp_y)[:, :, :, None]
            Y = batch_elastic_def(Y, disp_x, disp_y)

        # If needed, apply flipping
        # X is in format B01C, Y is in format B01
        for i in range(len(X)):
            if self.h_flipping and np.random.random() < 0.5:
                X[i] = X[i, :, ::-1]
                Y[i] = Y[i, :, ::-1]
            if self.v_flipping and np.random.random() < 0.5:
                X[i] = X[i, ::-1, :]
                Y[i] = Y[i, ::-1, :]

        # If needed, apply shearing deformation
        if self.shearing_range > 0.0:
            from keras.preprocessing.image import (
                apply_transform, transform_matrix_offset_center)

            height = X.shape[2]
            width = X.shape[3]

            for i in range(len(X)):

                shear = np.random.uniform(-self.shearing_range,
                                          self.shearing_range)
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                         [0, np.cos(shear), 0],
                                         [0, 0, 1]])

                transform_matrix = transform_matrix_offset_center(shear_matrix,
                                                                  height,
                                                                  width)
                X[i] = apply_transform(X[i], transform_matrix, 2,
                                       'nearest', 0.)
                Y[i] = apply_transform(Y[i, :, :, None], transform_matrix, 2,
                                       'nearest', 0.)[:, :, 0]

        X = X.astype("float32") / 255
        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        ret['subset'] = prefix
        ret['filenames'] = np.array(F)

        return ret


def test():
    trainiter = IsbiEmStacksDataset(
        which_set='train',
        batch_size=5,
        seq_per_subset=0,
        seq_length=0,
        overlap=9,
        return_one_hot=True,
        return_01c=True,
        data_augm_kwargs={
            'crop_size': (224, 224)},
        return_list=True,
        elastic_deform=True)

    train_nsamples = trainiter.nsamples
    nclasses = trainiter.nclasses
    nbatches = trainiter.nbatches
    train_batch_size = trainiter.batch_size
    print("Train %d" % (train_nsamples))

    start = time.time()
    tot = 0
    max_epochs = 2

    for epoch in range(max_epochs):
        for mb in range(nbatches):
            train_group = trainiter.next()

            # train_group checks
            assert train_group[0].ndim == 4
            assert train_group[0].shape[0] <= train_batch_size
            assert train_group[0].shape[1:] == (224, 224, 1)
            assert train_group[0].min() >= 0
            assert train_group[0].max() <= 1
            assert train_group[1].ndim == 4
            assert train_group[1].shape[0] <= train_batch_size
            assert train_group[1].shape[1:] == (224, 224, nclasses)

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s time: %s (%s)" % (str(mb), part, tot))


if __name__ == '__main__':
    test()
