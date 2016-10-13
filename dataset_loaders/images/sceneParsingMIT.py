# Imports
import os
import numpy as np
from PIL import Image
from getpass import getuser
import time
from dataset_loaders.parallel_loader import ThreadedDataset
floatX = 'float32'


# Load file with the description of the dataset classes
def load_class_names(file_name):
    # Read the csv data
    from numpy import genfromtxt
    csv_data = genfromtxt(file_name, delimiter='\t', dtype=None, skip_header=1)
    # print str(csv_data)

    # Create the mask labels dictionary
    mask_labels = {}
    for line in csv_data:
        mask_labels[int(line[0])] = line[4]
    # print(str(mask_labels))

    # TODO: Other data can be taken from here like class frecuency

    return mask_labels


class sceneParsingMIT(ThreadedDataset):

    # Set basic info from the dataset
    name = 'scene_parsing_MIT'
    nclasses = 150
    debug_shape = (375, 500, 3)
    data_shape = (None, None, 3)
    mean = np.asarray([0., 0., 0.]).astype('float32')
    std = 1.
    _void_labels = [-1]   # TODO: Check
    GTclasses = range(nclasses) + _void_labels
    _filenames = None
    _cmap = {}
    _mask_labels = {}

    @property
    def filenames(self):
        import glob

        if self._filenames is None:
            # Load filenames
            filenames = []

            # Get file names from images folder
            file_pattern = os.path.join(self.image_path, "*.jpg")
            file_names = glob.glob(file_pattern)
            # print (str(file_names))

            # Get raw filenames from file names list
            for file_name in file_names:
                path, file_name = os.path.split(file_name)
                file_name, ext = os.path.splitext(file_name)
                raw_name = file_name.strip()
                filenames.append(file_name)
                # print (file_name)

            # Save the filenames list
            self._filenames = filenames
        return self._filenames

    def __init__(self,
                 which_set="train",
                 with_filenames=False,
                 *args, **kwargs):

        # Get dataset path
        usr = getuser()
        self.path = '/Tmp/'+usr+'/datasets/SceneParsingMIT/'
        self.sharedpath = '/data/lisa/exp/vazquezd/datasets/SceneParsingMIT/'

        # Put the which_set in the cannonical form ("training", "validation", "testing")
        self.which_set = which_set
        self.which_set = "training" if self.which_set in ("train", "training") else self.which_set
        self.which_set = "validation" if self.which_set in ("val", "valid", "validation") else self.which_set
        self.which_set = "testing" if self.which_set in ("test", "testing") else self.which_set
        if self.which_set not in ("training", "validation", "testing"):
            raise ValueError("Unknown argument to which_set %s" %
                             self.which_set)
        if self.which_set == 'testing':
            self.has_GT = False

        # Define the txt, images and mask paths
        self.txt_path = os.path.join(self.path, "objectInfo150.txt")
        self.image_path = os.path.join(self.path, "images", self.which_set)
        self.mask_path = os.path.join(self.path, "annotations", self.which_set)

        # Other stuff
        self.with_filenames = with_filenames

        # Load info from the classes
        _mask_labels = load_class_names(self.txt_path)

        super(sceneParsingMIT, self).__init__(*args, **kwargs)

    def get_names(self):

        # # Load info from the classes
        # _mask_labels = load_class_names(self.txt_path)

        # Limit to the number of videos we want
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        image_names = self.filenames
        video_length = len(image_names)
        max_num_sequences = video_length - seq_length + 1
        if (not self.seq_length or not self.seq_per_video or
                self.seq_length >= video_length):
            # Use all possible frames
            sequences = image_names[:max_num_sequences:
                                    self.seq_length - self.overlap]
        else:
            if max_num_sequences < seq_per_video:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                print("/!\ Warning : you asked {} sequences of {} "
                      "frames each but the dataset only has {} "
                      "frames".format(seq_per_video, seq_length,
                                      video_length))
                seq_per_video = max_num_sequences

            if self.overlap != self.seq_length - 1:
                raise('Overlap other than seq_length - 1 is not '
                      'implemented')
            # pick `seq_per_video` random indexes between 0 and
            # (video length - sequence length)
            first_frame_indexes = self.rng.permutation(range(
                max_num_sequences))[0:seq_per_video]

            for i in first_frame_indexes:
                sequences.append(image_names[i])

        # Return images
        return np.array(sequences)

    def load_sequence(self, img_name):
        from skimage import io
        image_batch = []
        mask_batch = []
        filename_batch = []

        if self.seq_length != 1:
            raise NotImplementedError()

        # Load image
        img = io.imread(os.path.join(self.image_path, img_name +
                                     ".jpg"))
        img = img.astype(floatX) / 255.

        # Load mask
        if self.which_set != "test":
            mask = np.array(Image.open(
                os.path.join(self.mask_path, img_name + ".png")))
            mask = mask.astype('int32')

        # Add to minibatch
        image_batch.append(img)
        if self.which_set != "test":
            mask_batch.append(mask)
        if self.with_filenames:
            filename_batch.append(img_name)

        image_batch = np.array(image_batch)
        mask_batch = np.array(mask_batch)
        filename_batch = np.array(filename_batch)

        if self.with_filenames:
            return image_batch, mask_batch, filename_batch
        else:
            return image_batch, mask_batch


def test():
    trainiter = sceneParsingMIT(
        which_set='train',
        batch_size=100,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=True)

    validiter = sceneParsingMIT(
        which_set='valid',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    testiter = sceneParsingMIT(
        which_set='test',
        batch_size=5,
        seq_per_video=0,
        seq_length=0,
        crop_size=(224, 224),
        get_one_hot=True,
        get_01c=True,
        use_threads=False)

    # Get number of classes
    nclasses = trainiter.get_n_classes()
    print ("N classes: " + str(nclasses))

    # Training info
    train_nsamples = trainiter.nsamples
    train_batch_size = trainiter.get_batch_size()
    train_nbatches = trainiter.get_n_batches()
    print("Train n_images: {}, batch_size{}, n_batches{}".format(train_nsamples,
                                                                 train_batch_size,
                                                                 train_nbatches))

    # Validation info
    valid_nsamples = validiter.nsamples
    valid_batch_size = validiter.get_batch_size()
    valid_nbatches = validiter.get_n_batches()
    print("Train n_images: {}, batch_size: {}, n_batches: {}".format(valid_nsamples,
                                                                 valid_batch_size,
                                                                 valid_nbatches))

    # Testing info
    test_nsamples = testiter.nsamples
    test_batch_size = testiter.get_batch_size()
    test_nbatches = testiter.get_n_batches()
    print("Test n_images: {}, batch_size: {}, n_batches: {}".format(test_nsamples,
                                                                 test_batch_size,
                                                                 test_nbatches))

    max_epochs = 2

    for epoch in range(max_epochs):
        epoch_start = time.time()
        for mb in range(train_nbatches):
            mb_start = time.time()
            train_batch = trainiter.next()
            print("Minibatch {}: {:.3f} seg".format(mb, time.time() - mb_start))
        print("End epoch {}: {:.3f} seg".format(epoch, time.time() - epoch_start))


if __name__ == '__main__':
    test()
