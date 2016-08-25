import numpy as np
import os
import time
import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset
floatX = 'float32'


class Polyps300Dataset(ThreadedDataset):
    name = 'Polyps300'
    nclasses = 4
    debug_shape = (384, 288, 3)

    # optional arguments
    data_shape = (384, 288, 3)

    _void_labels = [0]
    _cmap = {
        0: (0, 0, 0),           # Void
        1: (128, 128, 0),       # Background
        2: (128, 0, 0),         # Polyp
        3: (192, 192, 128),     # Specularity
        4: (128, 64, 128)}      # Lumen
    _mask_labels = {0: 'Void', 1: 'Background', 2: 'Polyp', 3: 'Specularity',
                    4: 'Lumen'}

    def __init__(self, which_set='train', with_filenames=False,
                 select='sequences', *args, **kwargs):

        self.which_set = "val" if which_set == "valid" else which_set
        self.with_filenames = with_filenames
        self.select = select
        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'polyps')
        self.sharedpath = '/data/lisa/exp/vazquezd/datasets/polyps/'

        self.image_path_300 = os.path.join(self.path, 'CVC-300', "bbdd")
        self.mask_path_300 = os.path.join(self.path, 'CVC-300', "labelled")
        self.image_path_612 = os.path.join(self.path, 'CVC-612', "bbdd")
        self.mask_path_612 = os.path.join(self.path, 'CVC-612', "labelled")

        super(Polyps300Dataset, self).__init__(*args, **kwargs)

    def get_names(self):

        # Select elements where the column 'c' is in ids
        def select_elements(data, c, ids):
            select = data[np.logical_or.reduce([data[:, c] == x
                                               for x in ids])].astype(int)
            # print "Select: " + str(select)
            return select

        # Get file names from the selected files
        def select_filenames(select):
            filenames = []
            for i in range(select.shape[0]):
                filenames.append(str(select[i, 0]))
            # print "Filenames: " + str(filenames)
            return filenames

        # Get file names in this frame ids
        def by_frame(data, ids):
            return select_filenames(select_elements(data, 0, ids))

        # Get file names in this sequence ids
        def by_sequence(data, ids):
            return select_filenames(select_elements(data, 3, ids))

        # Get file names in this sequence ids
        def by_patience(data, ids):
            return select_filenames(select_elements(data, 1, ids))

        def get_file_names_by_frame(data, id_first, id_last):
            return by_frame(data, range(id_first, id_last))

        def get_file_names_by_sequence(data, id_first, id_last):
            return by_sequence(data, range(id_first, id_last))

        def get_file_names_by_patience(data, id_first, id_last):
            return by_patience(data, range(id_first, id_last))

        # Get the metadata info from the dataset
        def read_csv(file_name):
            from numpy import genfromtxt
            csv_data = genfromtxt(file_name, delimiter=';')
            return csv_data
            # [Frame ID, Patiend ID, Polyp ID, Polyp ID2]

        # Get the metadata info from the dataset
        self.CVC_300_data = read_csv(os.path.join(self.sharedpath,
                                                  "CVC-300.csv"))
        self.CVC_612_data = read_csv(os.path.join(self.sharedpath,
                                                  "CVC-612.csv"))

        if self.select == 'frames':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         1, 401)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         401, 501)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_frame(self.CVC_612_data,
                                                         501, 613)
                self.is_300 = False
            else:
                print 'EROR: Incorret set: ' + self.filenames
                exit()
        elif self.select == 'sequences':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            1, 21)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            21, 26)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_sequence(self.CVC_612_data,
                                                            26, 32)
                self.is_300 = False
            else:
                print 'EROR: Incorret set: ' + self.filenames
                exit()
        elif self.select == 'patience':
            # Get file names for this set
            if self.which_set == "train":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            1, 16)
                self.is_300 = False
            elif self.which_set == "val":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            16, 21)
                self.is_300 = False
            elif self.which_set == "test":
                self.filenames = get_file_names_by_patience(self.CVC_612_data,
                                                            21, 26)
                self.is_300 = False
            else:
                print 'EROR: Incorret set: ' + self.filenames
                exit()
        else:
            print 'EROR: Incorret select: ' + self.select
            exit()

        # Limit to the number of videos we want
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        video_length = len(self.filenames)
        max_num_sequences = video_length - seq_length + 1
        if (not self.seq_length or not self.seq_per_video or
                self.seq_length >= video_length):
            # Use all possible frames
            sequences = self.filenames[:max_num_sequences]
        else:
            if max_num_sequences < seq_per_video:
                # If there are not enough frames, cap seq_per_video to
                # the number of available frames
                print("/!\ Warning : you asked {} sequences of {} "
                      "frames each but the dataset only has {} "
                      "frames".format(seq_per_video, seq_length,
                                      video_length))
                seq_per_video = max_num_sequences

            # pick `seq_per_video` random indexes between 0 and
            # (video length - sequence length)
            first_frame_indexes = self.rng.permutation(range(
                max_num_sequences))[0:seq_per_video]

            for i in first_frame_indexes:
                sequences.append(self.filenames[i])

        # Return images
        return np.array(sequences)

    def load_sequence(self, img_name):
        from skimage import io
        image_batch = []
        mask_batch = []
        pred_batch = []
        filename_batch = []

        if self.seq_length != 1:
            raise NotImplementedError()

        image_path = self.image_path_300 if self.is_300 else self.image_path_612
        mask_path = self.mask_path_300 if self.is_300 else self.mask_path_612

        # Load image
        img = io.imread(os.path.join(image_path, img_name + ".bmp"))
        img = img.astype(floatX) / 255.
        # print 'Image shape: ' + str(img.shape)

        # Load mask
        mask = np.array(io.imread(os.path.join(mask_path, img_name + ".tif")))
        mask = mask.astype('int32')
        #mask[mask == 0] = 5  # Set the void mask value to 5 instead to 0
        #mask = mask - 1  # Start the first class as 0 instead in 1
        # print 'Mask shape: ' + str(mask.shape)
        # print 'Mask: ' + str(mask)

        from skimage.color import label2rgb
        import scipy.misc
        import seaborn as sns

        # color_map = sns.hls_palette(4 + 1)
        # image = img.transpose((1, 2, 0))
        # image = img
        # print 'Image shape: ' + str(image.shape)
        # label_mask = label2rgb(mask, bg_label=0, colors=color_map)
        # np.set_printoptions(threshold=np.inf)
        # # print 'mask: ' + str(mask)
        # print 'mask shape: ' + str(mask.shape)
        # print 'mask max: ' + str(np.max(mask))
        # print 'mask min: ' + str(np.min(mask))
        # combined_image = np.concatenate((image, label_mask), axis=1)
        # print 'combined_image shape: ' + str(combined_image.shape)
        # scipy.misc.toimage(combined_image).save('./debugImage' + img_name + '.png')
        # exit()
        # import pdb; pdb.set_trace()

        # Add to minibatch
        image_batch.append(img)
        mask_batch.append(mask)
        if self.with_filenames:
            filename_batch.append(img_name)

        ret = [np.array(image_batch), np.array(mask_batch)]

        other = []
        if self.with_filenames:
            other += [np.array(filename_batch)]

        # return image_batch, mask_batch, batch_to_load, pred_batch
        return tuple(ret + other)


def test1():
    dd = Polyps300Dataset(which_set='test',
                          batch_size=10,
                          seq_per_video=0,
                          seq_length=0,
                          crop_size=(224, 224),
                          get_one_hot=False,
                          get_01c=False,
                          use_threads=False)

    print('Tot {}'.format(dd.epoch_length))
    for i, _ in enumerate(range(dd.epoch_length)):
        dd.next()
        # if i % 20 == 0:
        print str(i)


if __name__ == '__main__':
    test1()
