from scipy.sparse import csr_matrix, hstack

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """

    return hstack([X, csr_matrix(feature_to_add).T], 'csr')