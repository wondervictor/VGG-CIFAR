# -*- coding: utf-8 -*-

import cPickle as pickle
import numpy as np
import os



def data_reader(path, n):
    def reader():
        with open(path, 'r') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3072).astype("float")
            Y = np.array(Y)

            size = 10000
            for i in range(size):
                yield {'image': X[i] / 255.0,
                       'label': int(Y[i])
                       }
    return reader
