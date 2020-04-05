# -*- coding:utf8 -*-
from __future__ import print_function
import datetime

import argparse
from math import log10
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from model.ByteCAR import BytecarNet
from model.Arcnn import ARCNN, weights_init
from data.data import get_training_set, get_test_set
from option import opt
from tqdm import tqdm
import logging

logging.basicConfig(filename='./LOG/' + 'experiment' + '.log', level=logging.INFO)

opt = opt
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # '0,1,2'
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)

device = torch.device("cuda" if opt.cuda else "cpu")

print('===> Loading datasets')
train_set = get_training_set()
test_set = get_test_set()

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = ARCNN().to(device)
# apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 就这一行
    model = nn.DataParallel(model)
    
model.apply(weights_init)

criterion = nn.MSELoss()

optimizer = optim.Adam([
    {'params': model.base.parameters()},
    {'params': model.last.parameters(), 'lr': opt.lr * 0.1},
], lr=opt.lr)


# optimizer = optim.Adam(model.parameters(), lr=opt.lr)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        # print('===>输入', input[0,1,:,:])
        # print('===>输出', model(input)[0,1,:,:])
        # print('===>标签', target[0,1,:,:])
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print('===> Epoch[{}]({}/{}): Loss: {:.4f}'.format(epoch, iteration, len(training_data_loader), loss.item()))
    print('===> Epoch {} Complete: Avg. Loss: {:.4f}'.format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    with torch.no_grad():
        for batch in tqdm(testing_data_loader):
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / (mse.item() / 3))  # 就是1，因为还是tensor,0~1。rgb三通道，mse除3。
            avg_psnr += psnr

    print('===> Avg. PSNR: {:.4f} dB'.format(avg_psnr / len(testing_data_loader)))
    logging.info('===> Avg. PSNR: {:.4f} dB'.format(avg_psnr / len(testing_data_loader)))

    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    model_out_path = 'model_epoch_{}.pth'.format(epoch)
    model_latest_path = 'model_latest.pth'

    model_out_path = os.path.join('.', 'experiment', model_out_path)
    model_latest_path = os.path.join('.', 'experiment', 'latestcheckpoint', model_latest_path)
    if epoch % 5 == 0:
        # torch.save(model, model_out_path)
        torch.save(model.state_dict(), model_out_path)
        print('Checkpoint saved to {}'.format(model_out_path))
    # torch.save(model, model_latest_path)
    torch.save(model.state_dict(), model_latest_path)
    print('Checkpoint saved to {}'.format(model_latest_path))


# 记录实验时间
nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logging.info('experiment in {}'.format(nowTime))

if opt.pre_train:
    print('===> Working directory :', os.getcwd())
    print('===> Loading pretrained model from latest checkpoint')
    # model = torch.load(opt.pre_train)
    model.load_state_dict(torch.load(opt.pre_train))
    # 如果是部分加载
    ‘’‘
    # 1. 利用字典的 update 方法进行加载
    Checkpoint = torch.load(Path)
    model_dict = model.state_dict()
    model_dict.update(Checkpoint)
    model.load_state_dict(model_dict)
    # 2. 利用 load_state_dict() 的 strict 参数进行部分加载
    model.load_state_dict(torch.load(PATH), strict=False)
    ’‘’
    
    
    best_psnr = 33.0
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        logging.info('===> in {}th epochs'.format(epoch))
        psnr = test()
        if psnr > best_psnr:
            best_psnr = psnr
            model_best_path = os.path.join('.', 'experiment', 'model_best.pth')
            logging.info('===> save the best model: reach {:.2f}dB PSNR'.format(best_psnr))
            # torch.save(model, model_best_path)
            torch.save(model.state_dict(),PATH)
        checkpoint(epoch)

else:
    best_psnr = 33.0
    # train ; test ; checkpoint
    for epoch in range(1, opt.nEpochs + 1):
        train(epoch)
        logging.info('===> in {}th epochs'.format(epoch))
        psnr = test()
        if psnr > best_psnr:
            best_psnr = psnr
            model_best_path = os.path.join('.', 'experiment', 'model_best.pth')
            logging.info('===> save the best model: reach {:.2f}dB PSNR'.format(best_psnr))
            # torch.save(model, model_best_path)
            torch.save(model.state_dict(), model_best_path)
        checkpoint(epoch)
