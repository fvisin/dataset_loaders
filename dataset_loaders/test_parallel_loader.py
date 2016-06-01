import numpy as np
import time

from theano import config

from parallel_loader import ThreadedDataset


floatX = config.floatX


class TestDataset(ThreadedDataset):
    def __init__(self, array_dim=5, *args, **kwargs):
        self.array_dim = array_dim
        super(TestDataset, self).__init__(*args, **kwargs)
        assert self.queues_size <= array_dim

    def get_names(self):
        return np.array(range(self.array_dim))

    def fetch_from_dataset(self, to_load):
        """
        This returns a list of sequences or clips or batches.
        """
        X = range(self.array_dim)
        ret = []
        for el in to_load:
            if el is None:
                continue
            ret.append(X[el])
        return ret

if __name__ == '__main__':
    d = TestDataset(use_threads=False, array_dim=100)
    start = time.time()
    n_minibatches_to_run = 50
    itr = 1
    while True:
        ret = []
        print('Begin itr' + str(itr))
        for el in d:
            for inner_el in el:
                ret.append(inner_el)
        # Check that every element is unique
        assert(len(ret) == len(set(ret)))
        # time.sleep approximates running some model
        itr += 1
        if itr > n_minibatches_to_run:
            break
