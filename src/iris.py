__author__ = 'biswajeet'

from sklearn import datasets
import numpy as np

def load_data():
    data = datasets.load_iris()

    feature = data['data']
    target = data['target']
    return feature, target

def load_data_wrapper():

    tr_d, te_d = load_data()

    training_results = [vectorized_result(y) for y in te_d]
    training_data = zip(tr_d, training_results)
    print len(training_data)

    i = 0
    train_set = []
    test_set = []
    for item in training_data:
        if(i%5 == 0):
            test_set.append(item)
        else:
            train_set.append(item)
        i += 1

    return train_set, test_set

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((3, 1))
    e[j] = 1.0
    return e

print load_data_wrapper()