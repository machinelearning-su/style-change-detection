from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import math


def gmm(X, params):
    print('GMM...')
    transformed = []

    for doc in X:
        segments_count = len(doc)
        multiplier = math.ceil(params['n_components']/segments_count)
        doc = np.array(doc).repeat(multiplier, axis=0)

        if segments_count < params['n_components']:
            print("Duplicating segments: ", segments_count, doc.shape)

        if params['pca']:
            doc = PCA(n_components=2).fit_transform(doc)

        n_components = np.arange(1, params['n_components'] + 1)
        models = [
            GaussianMixture(
                n, covariance_type=params['covariance_type'], random_state=0, verbose=0)
            .fit(doc) for n in n_components
        ]

        transformed.append([m.bic(doc) for m in models])

    scaled = MinMaxScaler().fit_transform(transformed)
    
    return np.array(scaled)
