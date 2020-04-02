# -*- coding:utf8 -*-
from __future__ import print_function
from math import log10, sqrt
import logging
import random
import numpy as np
from PIL import Image
import torch
import logging

logging.basicConfig(filename='./psnr_result.log', level=logging.INFO)

# youkuSR/data/common.py

def MSE(img1,img2):
	mse = np.mean((img1 - img2)**2)
	return mse

def cal_psnr(img1, img2):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    mse = MSE(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * log10(255.0 / sqrt(mse))

img1_path = './valGT001.png'
img2_path = './valoutput001.png'
img1, img2 = Image.open(img1_path), Image.open(img2_path)

psnr_1_2 = cal_psnr(img1, img2)

print('===> psnr between {} and {} is : {}'.format(img1_path[2:], img2_path[2:], psnr_1_2))
logging.info('===> psnr between {} and {} is : {}'.format(img1_path[2:], img2_path[2:], psnr_1_2))


