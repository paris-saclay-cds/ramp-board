import numpy as np
from skimage.transform import resize

def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    x = resize(x, (64, 64), preserve_range=True)
    x = x.transpose((2, 0, 1))
    x /= 255.
    return x
