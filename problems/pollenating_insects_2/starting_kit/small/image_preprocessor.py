import numpy as np
from skimage.transform import resize

def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    x = resize(x, (32, 32), preserve_range=True)
    x = x / 255.
    x = x.transpose((2, 0, 1))
    return x
