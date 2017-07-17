__author__ = 'bacon'

import time
import numpy as np

# return k neighbors index
def navie_knn(dataSet,query,k):
    numSamples = dataSet.shape[0]
