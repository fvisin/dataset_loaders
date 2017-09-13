import numpy as np
import os
import copy
import h5py
import time
from scipy.ndimage.interpolation import zoom

from dataset_loaders.parallel_loader import ThreadedDataset


class MovingMNISTDataset(ThreadedDataset):
    # Adapted from
    # https://github.com/dbbert/dfn

    """ The moving MNIST dataset

    Parameters
    ----------
    which_set: string
        A string in ['train', 'valid', 'test'], corresponding to the
        subset from where the digits are taken to generate the video
        sequences
    seq_per_subset: int or np.inf
        "For this dataset, `seq_per_subset` specifies the number of generated
        sequences rather than the number of sequences to be used as usual. Note
        that by setting it to `np.inf` the dataset will return an infinite
        number of sequences (i.e., it will generate random sequences forever,
        without repetition)"
    frame_size: list of int
        The size [h, w] of the frames
    num_digits: int
        The number of moving digits
    digits_sizes: list of int
        The size [h, w] of the digits
    random_background: bool
        The background will be black if False, otherwise it will be
        initialized randomly
    binarize: bool
        If True the frames in the sequence will be returned as binary
        images {0, 1}, otherwise the images will be in the range [0, 1].
        To generate the binary images from the non-binarized original
        ones a threshold has been set. All the values below the threshold
        are set to 0 otherwise to 1. The value used for threshold is 0.8
        since it was empirically proved to be able to generate good
        non-blurry images.
    init_speed_range: list of float or list of lists of float
        The range of the initial linear speed of each digit. A digit
        speed in that range will be randomly sampled for each digit. If
        a single list is provided all the digits will have an initial
        speed sampled in the same range (although with different
        samples). Otherwise a list specifying the range of initial speed
        for each digit is expected.
    delta_speed_range: list of float or list of lists of float
        The range from which the variation of speed is sampled for each new
        frame. The higher the value the more 'complex' the sequence trajectory
        will be. When `delta_speed_range` is [0, 0] the digits move at constant
        speed
    steering_prob: float or list of float
        The probability for all the digits (when float) or for each digit (when
        list of floats) to randomly change direction at each frame. When zero,
        the digits will always move in the same direction (until they bounce on
        a frame border)
    rng:
        The random number generator (rng) used to stochastically select the
        speed and direction of the digits. If not specified, a default rng
        (seed = 1) is created instead.
    """
    name = 'movingMNIST'
    # The number of *non void* classes
    non_void_nclasses = None

    # A list of the ids of the void labels
    _void_labels = []

    def __init__(self, which_set='train', frame_size=[64, 64],
                 num_digits=1, digits_sizes=[28, 28], random_background=False,
                 init_speed_range=[-0.3, 0.3], delta_speed_range=[-0.1, 0.1],
                 steering_prob=0., binarize=True, rng=None, *args, **kwargs):

        self.data_shape = frame_size + [1]

        self.which_set = 'validation' if 'valid' in which_set else which_set
        self.frame_size = frame_size
        self.num_digits = num_digits
        self.digits_sizes = digits_sizes
        self.random_background = random_background
        self.set_has_GT = False
        if not kwargs['seq_per_subset'] or kwargs['seq_per_subset'] == 0:
            raise RuntimeError('seq_per_subset must be an int or np.inf!')
        if isinstance(init_speed_range[0], list):
            assert len(init_speed_range) == self.num_digits, (
                'When `digits_speed` is a list of lists its length '
                'must be equal to the number of digits')
            self.init_speed_range = init_speed_range
        else:
            self.init_speed_range = [init_speed_range for _ in
                                     range(self.num_digits)]
        if isinstance(delta_speed_range[0], list):
            assert len(delta_speed_range) == self.num_digits, (
                'When `delta_speed_range` is a list of lists its length '
                'must be equal to the number of digits')
            self.delta_speed_range = delta_speed_range
        else:
            self.delta_speed_range = [delta_speed_range for _ in
                                      range(self.num_digits)]
        if isinstance(steering_prob, list):
            assert len(steering_prob) == self.num_digits, (
                'When `steering_prob` is a list of lists its length '
                'must be equal to the number of digits')
            self.steering_prob = steering_prob
        else:
            self.steering_prob = [steering_prob for _ in
                                  range(self.num_digits)]
        self.binarize = binarize
        self._rng = rng if rng else np.random.RandomState(1)
        # Used to reset 'self.rng' to the initial rng when the dataset
        # is reset
        self._initial_rng = copy.deepcopy(self._rng)
        try:
            f = h5py.File(os.path.join(self.path, 'mnist.h5'))
        except:
            raise RuntimeError('Failed to load dataset file from: %s' %
                               os.path.join(self.path, 'mnist.h5'))

        self._MNIST_data = f[self.which_set].value.reshape(-1, 28, 28)
        f.close()

        super(MovingMNISTDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        """Return a dict of names, per prefix/subset.
           Note: called only when seq_per_subset is not inf."""
        n_seq = self.seq_per_subset if self.seq_per_subset else 1
        n_imgs = self.seq_length * n_seq
        return {'default': ['gen_%i' % i for i in range(n_imgs)]}

    def _get_random_trajectory(self):
        # Increase the sequence by one, to account for the extra frame
        # that will be used as a target
        ext_seq_length = self.seq_length + 1
        canvas_size = self.frame_size - np.max(self.digits_sizes)

        # Initial position uniform random inside the box.
        y = self._rng.rand(self.num_digits)
        x = self._rng.rand(self.num_digits)
        # Initial speed module and angle
        speed_module = [self._rng.uniform(low=init_range[0],
                                          high=init_range[1])
                        for init_range in self.init_speed_range]
        theta = [self._rng.uniform(high=2*np.pi)
                 for _ in range(self.num_digits)]
        # Compute the speed components along the (x, y) axis
        speed_x = np.cos(theta) * speed_module
        speed_y = np.sin(theta) * speed_module

        # This will contain the trajectory along the (x, y) axis
        trajectory_x = np.zeros((ext_seq_length, self.num_digits))
        trajectory_y = np.zeros((ext_seq_length, self.num_digits))

        for frame_id in range(ext_seq_length):

            delta_x = [self._rng.uniform(low=d_speed_range[0],
                                         high=d_speed_range[1])
                       for d_speed_range in self.delta_speed_range]
            delta_y = [self._rng.uniform(low=d_speed_range[0],
                                         high=d_speed_range[1])
                       for d_speed_range in self.delta_speed_range]

            do_steer = [self._rng.binomial(1, steering_prob)
                        for steering_prob in self.steering_prob]

            # Compute the delta speed components along the (x, y) axis
            speed_x += np.multiply(delta_x, do_steer)
            speed_y += np.multiply(delta_y, do_steer)

            # Take a step along velocity.
            x += speed_x
            y += speed_y
            # Bounce off edges.
            for j in range(self.num_digits):
                if x[j] <= 0.:
                    x[j] = -x[j]
                    speed_x[j] = -speed_x[j]
                elif x[j] >= 1.0:
                    x[j] = 2.0 - x[j]
                    speed_x[j] = -speed_x[j]
                if y[j] <= 0.:
                    y[j] = -y[j]
                    speed_y[j] = -speed_y[j]
                elif y[j] >= 1.0:
                    y[j] = 2.0 - y[j]
                    speed_y[j] = -speed_y[j]

            trajectory_x[frame_id] = x
            trajectory_y[frame_id] = y

        # Map to pixel coordinates in the canvas
        trajectory_x = (canvas_size[1] * trajectory_x).astype(np.int32)
        trajectory_y = (canvas_size[0] * trajectory_y).astype(np.int32)
        return trajectory_x, trajectory_y

    def _get_sequence(self, verbose=False):
        trajectory_x, trajectory_y = self._get_random_trajectory()

        # Minibatch data
        if self.random_background:
            out_sequence = self._rng.rand(self.seq_length + 1,
                                          self.frame_size[0],
                                          self.frame_size[1], 1)
        else:
            out_sequence = np.zeros((self.seq_length + 1,
                                     self.frame_size[0],
                                     self.frame_size[1], 1),
                                    dtype=np.float32)

        for digit_id in range(self.num_digits):

            # Get random digit from dataset
            curr_data_idx = self._rng.randint(0, self._MNIST_data.shape[0] - 1)
            digit_image = self._MNIST_data[curr_data_idx]

            zoom_factor = int(self.digits_sizes[digit_id] / 28)
            if zoom_factor != 1:
                digit_image = zoom(digit_image, zoom_factor)
            digit_size = digit_image.shape[0]

            # Generate video
            digit_image = np.expand_dims(digit_image, -1)
            # Iterate over seq_length + 1 to account for the extra frame
            # that is returned as a target
            for i, (top, left) in enumerate(zip(trajectory_y[:, digit_id],
                                                trajectory_x[:, digit_id])):
                bottom = top + digit_size
                right = left + digit_size
                out_sequence[i, top:bottom, left:right, :] = np.maximum(
                    out_sequence[i, top:bottom, left:right, :], digit_image)
        return out_sequence

    def _reset(self, *args, **kwargs):
        """ This function is used to reset the rng and data index from
        where the digits are sampled to create the video sequence
        """
        self._rng = copy.deepcopy(self._initial_rng)

    def _fill_names_batches(self, *args, **kwargs):
        # Reset the digit generator at the end of the epoch
        self._reset(self, *args, **kwargs)
        super(MovingMNISTDataset, self)._fill_names_batches(*args, **kwargs)

    def load_sequence(self, sequence):
        """Load a sequence of images/frames

        Auxiliary function that loads a sequence of frames with
        the corresponding ground truth and their filenames.
        Returns a dict with the images in [0, 1], their corresponding
        labels, their subset (i.e. category, clip, prefix) and their
        filenames.
        train_nsamples = trainiter.nsamples
        """
        F = []
        # if self.seq...
        if self.seq_per_subset == np.inf:
            for f_idx in range(self.seq_length):
                F.append("gen_%i" % f_idx)
        else:
            for frame_name, _ in sequence:
                F.append(frame_name)

        sequence = self._get_sequence()
        X = sequence[:-1]
        Y = sequence[-1]
        Y = Y[np.newaxis, ...]

        if self.binarize:
            X[X < 0.8] = 0
            Y[Y < 0.8] = 0
            X[X >= 0.8] = 1
            Y[Y >= 0.8] = 1

        ret = {}
        ret['data'] = X
        ret['labels'] = Y
        ret['subset'] = 'default'
        ret['filenames'] = np.array(F)
        return ret


def test(inf_dataset=True):
    trainiter = MovingMNISTDataset(
        which_set='train',
        batch_size=3,
        seq_per_subset=np.inf if inf_dataset else 6,
        seq_length=7,
        overlap=0,
        data_augm_kwargs={'crop_size': None},
        return_one_hot=True,
        return_01c=True,
        use_threads=True,
        shuffle_at_each_epoch=False)

    start = time.time()
    tot = 0
    max_epochs = 2
    for epoch in range(max_epochs):
        for mb in range(2):
            train_group = trainiter.next()

            # time.sleep approximates running some model
            time.sleep(1)
            stop = time.time()
            part = stop - start - 1
            start = stop
            tot += part
            print("Minibatch %s (inf %s) - Threaded time: %s (%s)" %
                  (str(mb), str(inf_dataset), part, tot))


def run_tests():
    test(False)
    test(True)


if __name__ == '__main__':
    run_tests()
