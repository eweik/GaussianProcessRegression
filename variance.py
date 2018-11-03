#
#
# Edison Weik
# Variance functions 
# variance.py
#
#
# Calculate variances & covariances of argument arrays using the given
# kernel function to compute the elements.

import numpy as np

def variance( X, k ):
    K = np.zeros( (len(X), len(X) ) )
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = k(X[i], X[j])
    return K
   
    
def covariance( X1, X2, k ):
    K = np.zeros( (len(X1), len(X2) ) )
    for i in range(len(X1)):
        for j in range(len(X2)):
            K[i, j] = k(X1[i], X2[j])
    return K