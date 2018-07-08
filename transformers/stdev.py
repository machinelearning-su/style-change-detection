import numpy as np

def stdev(X):
    print('Stdev...')
    return [np.std(vect, axis=0) for vect in X]
