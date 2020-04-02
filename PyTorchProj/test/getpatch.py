import random

import numpy as np

from PIL import Image

import torch


# youkuSR/data/common.py


def get_patch(img_in, patch_size):  # CAR的get_patch
    img_in = np.asarray(img_in)
    ih, iw = img_in.shape[-3:-1]  # N,H,W,C
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    img_out = img_in[iy:iy + ip, ix:ix + ip, :]
    img_out = Image.fromarray(img_out.astype(np.uint8))

    return img_out

img = Image.open('./test.png')
img = get_patch(img, 512)
img.save('./test_512.png')

