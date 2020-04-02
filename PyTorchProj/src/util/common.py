import random

import numpy as np

import torch


# youkuSR/data/common.py

def get_patch_SR(img_in, img_tar, patch_size, scale=2, multi_scale=False):
    multi_frames = len(img_in.shape) == 4
    # ih, iw = args[0].shape[:2]

    ih, iw = img_in.shape[-3:-1]
    p = scale if multi_scale else 1  # 是否支持多尺度超分。即上采样在网络前面，还是末尾。multi_scale==False，就是固定尺度的超分。
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    if multi_frames:
        img_in = img_in[:, iy:iy + ip, ix:ix + ip, :]
    else:
        img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar


def get_patch(img_in, img_tar, patch_size):  # CAR的get_patch
    multi_frames = len(img_in.shape) == 4  # 多帧恢复一帧，如EDVR
    # ih, iw = args[0].shape[:2]

    ih, iw = img_in.shape[-3:-1]  # N,H,W,C
    ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if multi_frames:
        img_in = img_in[:, iy:iy + ip, ix:ix + ip, :]
    else:
        img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[iy:iy + ip, ix:ix + ip, :]

    return img_in, img_tar


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            np_transpose = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
        else:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # https://zhuanlan.zhihu.com/p/59767914
        tensor = torch.from_numpy(np_transpose).float() / rgb_range

        return tensor

    return [_np2Tensor(_l) for _l in l]


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            if hflip: img = img[:, :, ::-1, :]
            if vflip: img = img[:, ::-1, :, :]
            if rot90: img = img.transpose(0, 2, 1, 3)
        else:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]
