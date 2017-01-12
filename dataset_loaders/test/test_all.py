import time
import unittest

from dataset_loaders import *


class TestDatasets(unittest.TestCase):

    def testAll(self, verbose=False):
        datasets = [el for el in dir() if 'Dataset' in el]
        for dataset in datasets:
            for which_set in ['train', 'val', 'test']:
                print('Testing ' + dataset.name + ' - ' + which_set)
                d = dataset(
                    which_set=which_set,
                    batch_size=1,
                    seq_length=0,  # 4D
                    raise_IOErrors=True)
                start = time.time()
                tot = 0

                for mb in range(d.epoch_length):
                    batch = d.next()
                    self.assertIsNotNone(batch, 'The batch was empty.')

                    # time.sleep approximates running some model
                    time.sleep(0.1)
                    if verbose:
                        stop = time.time()
                        part = stop - start - 1
                        start = stop
                        tot += part
                        print('Minibatch %s time: %s (%s)' % (
                            str(mb), part, tot))
