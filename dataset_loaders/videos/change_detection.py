import time
import os

import numpy as np
from PIL import Image

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset

floatX = 'float32'


class ChangeDetectionDataset(ThreadedDataset):
    '''The change detection 2014 dataset

    Multiple categories are given, each with multiple videos.
    Each video has associated a temporalROI and a ROI. The temporalROI
    determines which frames are to be used for training and test. The
    ROI defines the area of the frame that we are interested in.
    '''
    name = 'changeD'
    nclasses = 4
    _void_labels = [85]
    mean = [0.45483398, 0.4387207, 0.40405273]
    std = [0.04758175, 0.04148954, 0.05489637]
    _is_one_hot = False
    _is_01c = True

    debug_shape = (500, 500, 3)  # whatever...
    GTclasses = [0, 50, 85, 170, 255]
    categories = ['badWeather', 'baseline', 'cameraJitter',
                  'dynamicBackground', 'intermittentObjectMotion',
                  'lowFramerate', 'nightVideos', 'PTZ', 'shadow', 'thermal',
                  'turbulence']
    # with void [0.81749293, 0.0053444, 0.55085129, 0.03961262, 0.45756284]
    class_freqs = [0.81749293, 0.0053444, 0.03961262, 0.45756284]

    # static, shadow, ground, solid (buildings, etc), porous, cars, humans,
    # vert mix, main mix
    _cmap = {
        0: (0, 0, 0),           # static
        50: (255, 0, 0),        # shadow (red)
        170: (0, 255, 0),       # unknown (green)
        255: (255, 255, 255),   # moving (white)
        85: (127, 127, 127)}    # non-roi (grey)
    _mask_labels = {0: 'static', 50: 'shadow', 170: 'unknown', 255: 'moving',
                    85: 'non-roi'}

    _filenames = None

    @property
    def filenames(self):
        if self._filenames is None:
            inspect_dataset_properties = False  # debugging purpose
            self._filenames = {}
            ROIS = {}
            ROIS2 = {}
            tempROIS = {}
            cat_videos = {}
            for root, dd, ff in os.walk(self.path):
                if ff == [] or 'README' in ff:
                    # Root or category dir
                    dd.sort(key=str.lower)
                elif 'ROI.jpg' in ff:
                    # Video dir
                    category, video = root.split('/')[-2:]
                    cat_videos.setdefault(category, []).append(video)
                    ROI = np.array(Image.open(os.path.join(root, 'ROI.jpg')))
                    ROI2 = np.array(Image.open(os.path.join(root,
                                                            'ROI.bmp.tif')))
                    ROIS[video] = ROI
                    ROIS2[video] = ROI2
                    tempROIS[video] = open(os.path.join(
                        root, 'temporalROI.txt'), 'r').readline().split(' ')
                else:
                    # Images or GT dir
                    category, video, kind = root.split('/')[-3:]

                    if (category not in self.which_category or
                            video not in self.which_video):
                        continue

                    ff.sort(key=str.lower)
                    ff = [fname for fname in ff if 'Thumbs.db' not in fname]

                    if not inspect_dataset_properties:
                        # 1-indexed, inclusive
                        s, e = [int(el) - 1 for el in tempROIS[video]]
                        if self.which_set == 'test':
                            # anything out of tempROI
                            ff = ff[0:s] + ff[e+1:]
                        elif self.which_set == 'train':
                            d = int((e+1-s)*(1 - self.split))  # valid_delta
                            ff = ff[s+d:e+1]
                        else:
                            d = int((e+1-s)*(1 - self.split))  # valid delta
                            ff = ff[s:s+d]

                    if kind == 'input':
                        print('Loading {}..'.format(root[len(self.path):-6]))

                        self._filenames.setdefault(video, {}).update(
                            {'category': category,
                             'root': root[:-6],  # remove '/input'
                             'images': ff,
                             'ROI': ROIS[video],
                             'tempROI': tempROIS[video]})
                    else:
                        self._filenames.setdefault(video, {}).update(
                            {'GTs': ff})

            # Dataset properties:
            if inspect_dataset_properties:
                kk = self._filenames.keys()
                for k in kk:
                    tempROI = self._filenames[k]['tempROI']
                    # temporalROI is either at the beginning or at the end of
                    # the sequence
                    assert (int(tempROI[0]) == 001 or
                            int(tempROI[1]) == len(
                                self._filenames[k]['images'])), k
                    # First gt is gt000001.png
                    assert self._filenames[k]['GTs'][0] == 'gt000001.png', k
                    # First im is in000001.jpg
                    assert self._filenames[k]['images'][0] == 'in000001.jpg', k
                    # GT outside of tempROI is always 85 (void)
                    for i, f in enumerate(self._filenames[k]['images']):
                        if i < tempROI[0] or i > tempROI[1]:
                            continue  # consider only frames in tempROI
                        path = self._filenames[k]['root'] + '/' + f
                        gt = np.array(Image.open(path))
                        labels = np.unique(gt)
                        if len(labels) != 1:
                            print('k {} i {} labels {}'.format(k, i, labels))
                        if labels[0] != 85:
                            print('Non 85: k {} i {}'.format(k, i))

        return self._filenames

    def __init__(self,
                 which_set='train',
                 with_filenames=False,
                 split=.75,
                 which_category=('badWeather', 'baseline',
                                 'cameraJitter',
                                 'dynamicBackground',
                                 'intermittentObjectMotion', 'lowFramerate',
                                 'nightVideos', 'PTZ', 'shadow', 'thermal',
                                 'turbulence'),
                 which_video=(
                     # badWeather
                     'blizzard', 'skating', 'snowFall', 'wetSnow',
                     # baseline
                     'highway', 'office', 'pedestrians', 'PETS2006',
                     # cameraJitter
                     'badminton', 'boulevard', 'sidewalk', 'traffic',
                     # dynamicBackground
                     'boats', 'canoe', 'fall', 'fountain01', 'fountain02',
                     'overpass',
                     # intermittentObjectMotion
                     'abandonedBox', 'parking', 'sofa', 'streetLight',
                     'tramstop', 'winterDriveway',
                     # lowFramerate
                     'port_0_17fps', 'tramCrossroad_1fps',
                     'tunnelExit_0_35fps', 'turnpike_0_5fps',
                     # nightVideos
                     'bridgeEntry', 'busyBoulvard', 'fluidHighway',
                     'streetCornerAtNight', 'tramStation', 'winterStreet',
                     # PTZ
                     'continuousPan', 'intermittentPan',
                     'twoPositionPTZCam', 'zoomInZoomOut',
                     # shadow
                     'backdoor', 'bungalows', 'busStation', 'copyMachine',
                     'cubicle', 'peopleInShade',
                     # thermal
                     'corridor', 'diningRoom', 'lakeSide', 'library', 'park',
                     # turbulence
                     'turbulence0', 'turbulence1', 'turbulence2',
                     'turbulence3'),
                 *args, **kwargs):

        self.which_set = 'valid' if which_set == 'val' else which_set
        assert self.which_set in ['train', 'valid', 'test'], self.which_set
        self.with_filenames = with_filenames
        self.split = split
        self.which_category = which_category
        self.which_video = which_video

        # Prepare data paths
        self.path = os.path.join(dataset_loaders.__path__[0], 'datasets',
                                 'change_detection')
        self.sharedpath = '/u/visin/exp/_datasets/change_detection'

        if self.which_set == 'test':
            self.has_GT = False
            print('No mask for the test set!!')

        super(ChangeDetectionDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        sequences = []
        seq_length = self.seq_length
        seq_per_video = self.seq_per_video
        self.video_length = {}

        # cycle through the different videos
        for video, data in self.filenames.iteritems():
            video_length = len(data['images'])
            self.video_length[video] = video_length

            # Fill sequences with (video, frame_idx)
            max_num_sequences = video_length - seq_length + 1
            if (not self.seq_length or not self.seq_per_video or
                    self.seq_length >= video_length):
                # Use all possible frames
                for el in [(video, idx) for idx in range(
                        0, max_num_sequences, seq_length - self.overlap)]:
                    sequences.append(el)
            else:
                if max_num_sequences < seq_per_video:
                    # If there are not enough frames, cap seq_per_video to
                    # the number of available frames
                    print("/!\ Warning : you asked {} sequences of {} "
                          "frames each but video {} only has {} "
                          "frames".format(seq_per_video, seq_length,
                                          video, video_length))
                    seq_per_video = max_num_sequences

                if self.overlap != self.seq_length - 1:
                    raise('Overlap other than seq_length - 1 is not '
                          'implemented')

                # pick `seq_per_video` random indexes between 0 and
                # (video length - sequence length)
                first_frame_indexes = self.rng.permutation(range(
                    max_num_sequences))[0:seq_per_video]

                for i in first_frame_indexes:
                    sequences.append((video, i))

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

        video, idx = first_frame
        idx = int(idx)

        if (self.seq_length is None or
                self.seq_length > self.video_length[video]):
            seq_length = self.video_length[video]
        else:
            seq_length = self.seq_length

        data = self.filenames[video]
        root = data['root']
        for im, gt in zip(data['images'][idx:idx+seq_length],
                          data['GTs'][idx:idx+seq_length]):

            img = io.imread(os.path.join(self.path, root, 'input', im))
            mask = io.imread(os.path.join(self.path, root, 'groundtruth', gt))

            img = img.astype(floatX) / 255.
            mask = mask.astype('int32')

            X.append(img)
            Y.append(mask)
            F.append(os.path.join(root, 'input', im))

        # test only has void. No point in considering the mask
        Y = [] if self.which_set == 'test' else Y

        ret = {}
        ret['data'] = np.array(X)
        ret['labels'] = np.array(Y)
        if self.with_filenames:
            ret['filenames'] = np.array(F)
        return ret


if __name__ == '__main__':
    train = ChangeDetectionDataset(
        which_set='train',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        with_filenames=True,
        split=.75)
    valid = ChangeDetectionDataset(
        which_set='valid',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        with_filenames=True,
        split=.75)
    test = ChangeDetectionDataset(
        which_set='test',
        batch_size=1,
        seq_per_video=0,
        seq_length=0,
        shuffle_at_each_epoch=False,
        infinite_iterator=False,
        with_filenames=True,
        split=.75)
    train_nsamples = train.nsamples
    valid_nsamples = valid.nsamples
    test_nsamples = test.nsamples
    data = {'train': train, 'valid': valid, 'test': test}

    print("#samples: train %d, valid %d, test %d" % (train_nsamples,
                                                     valid_nsamples,
                                                     test_nsamples))

    start = time.time()
    for split in ['train', 'valid', 'test']:
        for i, el in enumerate(data[split]):
            if el[0].min() < 0:
                print('Image {} of {} is smaller than 0'.format(
                    el[2], split, el[0].min()))
            if el[0].max() > 1:
                print('Image {} of {} is greater than 1'.format(
                    el[2], split, el[0].max()))
            if split is not 'test' and el[1].max() > 4:
                print('Mask {} of {} is greater than 4: {}'.format(
                    el[2], split, el[1].max())),
            if split is not 'test' and np.unique(el[1]).tolist() == [4]:
                # check if the images is actually all void
                filename = el[2][0, 0, 0]
                f = filename[-10:-3]
                mask_f = filename[:-18] + 'groundtruth/gt' + f + 'png'
                un = np.unique(Image.open(mask_f))
                if un.tolist() != [85]:  # discard test
                    print('Image {} of {} is not test and is all void:{}. '
                          'It should be {}'.format(
                              el[2], split, np.unique(el[1]), un))
                # else:
                #     print('Image {} of {} is not test and is all void. '
                #           'Weird, but not an issue of the wrapper')
            if i % 1000 == 0:
                print('Sample {}/{} of {}'.format(i, data[split].nsamples,
                                                  split))
        print('Split {} done!'.format(split))
