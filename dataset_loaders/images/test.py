import numpy as np
import os

import dataset_loaders
from dataset_loaders.parallel_loader import ThreadedDataset


floatX = 'float32'


class TestDataset(ThreadedDataset):
    name = 'test'
    nclasses = 10
    debug_shape = (360, 480, 3)

    # optional arguments
    data_shape = (360, 480, 3)

    _void_labels = [3, 5]

    _filenames = None

    @property
    def filenames(self):
        return ['test']

    def __init__(self, *args, **kwargs):

        self.path = os.path.join(
            dataset_loaders.__path__[0], 'datasets', 'camvid', 'segnet')
        self.sharedpath = '/data/lisa/exp/visin/_datasets/camvid/segnet'

        # constructing the ThreadedDataset
        # it also creates/copies the dataset in self.path if not already there
        super(TestDataset, self).__init__(*args, **kwargs)

    def get_names(self):
        return np.array(['test'])

    def load_sequence(self, first_frame):
        max_label = self.nclasses + len(self._void_labels)
        mask = np.array(range(max_label) * 8).reshape((max_label, 8))
        img = np.random.random((self.nclasses, self.nclasses, 1))
        print('Mask before remap: \n{}'.format(mask))
        print('Max: {} \nExpected max after remap: {}'.format(mask.max(),
                                                              self.nclasses))
        print('Void labels: {}'.format(self._void_labels))

        ret = {}
        ret['data'] = np.array([img])
        ret['labels'] = np.array([mask])
        return ret


def test_one_hot_mapping():
    from dataset_loaders.images.test import TestDataset
    dd = TestDataset()
    dd.nclasses = 10
    dd._void_labels = [3, 5]
    mapping = {0: 0, 1: 1, 2: 2, 3: 10, 4: 3, 5: 10, 6: 4, 7: 5, 8: 6,
               9: 7, 10: 8, 11: 9}
    inv_mapping = {0: 0, 1: 1, 2: 2, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10,
                   9: 11}

    max_label = dd.nclasses + len(dd._void_labels)
    original_aa = np.array(range(max_label) * 8).reshape((max_label, 8))
    aa = dd.next()[1][0]
    print(aa)
    print('Max after remap: {}'.format(aa.max()))

    for i in range(dd.nclasses + len(dd._void_labels)):
        assert all(aa[original_aa == i] == mapping[i]), (
            'Key {} failed. aa content: n{}'.format(
                i, aa[original_aa == i]))
    for i in range(dd.nclasses):
        assert all(original_aa[aa == i] == inv_mapping[i]), (
            'Key {} failed. original_aa content: n{}'.format(
                i, original_aa[aa == i]))
    assert all([el in dd._void_labels
                for el in original_aa[aa == dd.nclasses]])

    print('Test successful!')


def test_one_hot_mapping2():
    from dataset_loaders.images.test import TestDataset
    dd = TestDataset()
    dd.nclasses = 10
    dd._void_labels = [0, 3, 5, 11]
    mapping = {0: 10, 1: 0, 2: 1, 3: 10, 4: 2, 5: 10, 6: 3, 7: 4, 8: 5,
               9: 6, 10: 7, 11: 10, 12: 8, 13: 9, 14: 10}
    inv_mapping = {0: 1, 1: 2, 2: 4, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 12,
                   9: 13, 10: 14}

    max_label = dd.nclasses + len(dd._void_labels)
    original_aa = np.array(range(max_label) * 8).reshape((max_label, 8))
    aa = dd.next()[1][0]
    print(aa)
    print('Max after remap: {}'.format(aa.max()))

    for i in range(dd.nclasses + len(dd._void_labels)):
        assert all(aa[original_aa == i] == mapping[i]), (
            'Key {} failed. aa content: n{}'.format(
                i, aa[original_aa == i]))
    for i in range(dd.nclasses):
        assert all(original_aa[aa == i] == inv_mapping[i]), (
            'Key {} failed. original_aa content: n{}'.format(
                i, original_aa[aa == i]))
    assert all([el in dd._void_labels
                for el in original_aa[aa == dd.nclasses]])

    print('Test successful!')


if __name__ == "__main__":
    test_one_hot_mapping()
    test_one_hot_mapping2()
