import numpy as np
from functools import reduce

def Dn(n):
    # Zero matrix
    D = np.zeros((n - 1, n))
    # Set coordinates to fill desire values (each row)
    row_index = np.arange(n - 1)
    # -1: column coordinates in [0, n - 1)
    # +1: column coordinates in [1, n)
    # Example (n = 5):
    # -1 +1  0  0  0
    #  0 -1 +1  0  0
    #  0  0 -1 +1  0
    #  0  0  0 -1 +1
    col_index = np.arange(n - 1)

    D[row_index, col_index] = -1
    D[row_index, col_index + 1] = 1

    return D

def WhittakerGraduation(z, w, h, n):
    # subscript: n - (z - 1) to n - (z - z)
    #        or: n - (z - i)
    # variable i: 1 to z (inclusive)
    DD = [Dn(n - z + i) for i in range(1, z + 1)]
    print("Number of Difference Matrices: {}".format(len(DD)))

    # Product of all Dn's
    k = reduce(lambda x, y: x.dot(y), DD)
    # Check k shape
    assert k.shape == (n - z, n)
    print("k_z shape: {}".format(k.shape))

    hkk = h * k.T.dot(k)

    return np.linalg.pinv(w + hkk)

if __name__ == "__main__":
    # Set numpy print precision to 0
    # suppress using scientific notation
    np.set_printoptions(precision=0, suppress=True)
    # Hyperparameters
    n = 10
    z = 2
    h = 100 #0.1
    # Define W matrix
    # Example: w_i = 1 for all i, which is identity matrix
    w = np.identity(n)

    # WhittakerGraduation
    smoothing = WhittakerGraduation(z, w, h, n)
    print("\nSmoothing matrix (%): \n{}".format(smoothing * 100))
