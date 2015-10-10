from scipy.io import loadmat
import sys
import numpy as np
import pandas as pd

if len(sys.argv) < 3:
    print "usage : to_csv matlabinputfile csvoutputfile [nan_representation]"
    print "default : nan_representation=-999"
    sys.exit(0)

if len(sys.argv) == 4:
    nan_repr = int(sys.argv[3])
else:
    nan_repr = -999

obj = loadmat(sys.argv[1])
X = obj['X']
y = obj['data_target']
h = obj['header']
h = ["labels"] + [h_col[0] for h_col in h[0]]


X = np.concatenate((y, X), axis=1)
df = pd.DataFrame(X)
df.to_csv(sys.argv[2], header=h, na_rep=nan_repr, index=False)
