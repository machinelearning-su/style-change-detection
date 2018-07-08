import numpy as np
from sklearn.preprocessing import StandardScaler

def max_diff(X, local_diff = False):
    transformed = []

    for vector in X:
        if(local_diff):
            vector_len = len(vector)
            vector_tmp = []
            for index in range(vector_len):
                if(index == vector_len - 1): break

                vector_tmp.append([float(abs(a - b)) for a, b in zip(vector[index], vector[index + 1])])
            vector = vector_tmp

        transformed.append(np.subtract(np.max(vector, axis=0), np.min(vector, axis=0)))

    return transformed
