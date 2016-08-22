import numpy as np
import os
import time

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class IsbiEmStacksDataset(ThreadedDataset):
    name = 'isbi_em_stacks'
    nclasses = 2
    debug_shape = (512, 512, 1)

    # optional arguments
    data_shape = (512, 512, 1)
    _void_labels = []
    cmap = np.array([
        (255, 255, 255),    # Membranes
        (0, 0, 0)])         # Non-membranes
    cmap = cmap / 255.
    labels = ('membranes', 'non-membranes')

    _filenames = None


    def __init__(self, which_set='train', start=0, end=30, *args, **kwargs):
        
        assert which_set in ["train", "test"]
        self.which_set = which_set
        
        assert start >= 0 and end <= 30 and start < end
        self.start = start
        self.end = end

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
            X.append(np.array(imgs)[:, :, None].astype(floatX) / 255)
            
            if load_labels:
                targets = Image.open(self.target_path)
                targets.seek(idx)
                Y.append(np.array(targets) / 255)

        if load_labels:
            print np.array(X).shape
            print np.array(Y).shape
            return np.array(X), np.array(Y)
        else:
            return np.array(X)     


def test2():
    d = IsbiEmStacksDataset(
        which_set='train',
        with_filenames=True,
        batch_size=5,
        seq_per_video=0,
        seq_length=10,
        overlap=9,
        get_one_hot=True,
        crop_size=(224, 224))
    
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
