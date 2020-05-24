# -*- coding:utf8 -*-
from __future__ import print_function
import argparse

# Training settings
parser = argparse.ArgumentParser(description='CAR ConvLSTM baseline')
parser.add_argument('--crf', type=int, required=True, help='compression factor crf')
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate. default=0.01')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for dataloader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use.')
parser.add_argument('--patchSize', type=int, default=96, help='CenterCrop size')
parser.add_argument('--rgb_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument('--pre_train', type=str, default='', help='pre-trained model directory')
parser.add_argument('--step', type=int, default=20, help='lstm指定多长的序列能够构成一个上下文相关的序列')
parser.add_argument('--dataset', type=str, default='', help='指定数据集路径，方便一点。虽然暂时没用它')
parser.add_argument('--data_range', type=str, default='0-100/100-160', help='train/test data range')


opt = parser.parse_args()

# -----annotation-----
1. --data_range
data_range = [r.split('-') for r in opt.data_range.split('/')] 
then use data_range[0][0]~data_range[0][1], data_range[1][0]~data_range[1][1] to divide the dataset.

2. --testBatchSize
look at this issue: https://github.com/pytorch/examples/issues/522
compute the PSNR for each image in the batch separately.
YYY also suggested that usually do not set test batchsize, verification is very fast.

