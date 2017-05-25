import numpy as np
from skimage.transform import resize
 
def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    x = resize(x, (224, 224), preserve_range=True)
    x = x.transpose((2, 0, 1))
    # 'RGB'->'BGR'
    x = x[::-1, :, :]
    # Zero-center by mean pixel
    x[0, :, :] -= 103.939
    x[1, :, :] -= 116.779
    x[2, :, :] -= 123.68
    return x
