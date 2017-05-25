import numpy as np
from skimage.transform import resize
 
def transform(x):
    if x.shape[2] == 4:
        x = x[:, :, 0:3]
    h, w = x.shape[:2]
    min_shape = min(h, w)
    x = x[h/2 - min_shape / 2:h/2 + min_shape / 2,
          w/2 - min_shape / 2:w/2 + min_shape / 2]
 
    x = resize(x, (224, 224), preserve_range=True)
    x = x.transpose((2, 0, 1))
    #using preprocessing from torchvision  : 
    #https://github.com/pytorch/examples/blob/master/imagenet/main.py
    x /= 255.
    x[0, :, :] -= 0.485
    x[0, :, :] /= 0.229
    x[1, :, :] -= 0.456
    x[1, :, :] /= 0.224
    x[2, :, :] -= 0.406
    x[2, :, :] /= 0.225
    return x
