import numpy as np
from skimage.transform import resize

def transform(x):
    return resize(x, (3, 64, 64), preserve_range=True)
