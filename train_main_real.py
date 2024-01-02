#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by HongW 2020-08-05
# This is a simple coding framework. The original framework for CVPR2020 RCDNet is at https://github.com/hongwang01/RCDNet
# CVPR2020  A Model-driven Deep Neural Network for Single Image Rain Removal
from __future__ import print_function
import argparse
import os
import random
import torch.nn.functional as  F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from rcdnet import RCDNet
from torch.utils.data import DataLoader
from DerainDataset  import SPATrainDataset
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",type=str, default="./data/spa-data/",help='txt path to training spa-data')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--patchSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--batchnum', type=int, default=1500, help='the number of batch at every epoch, need to be adjusted according to the total number of training samples')
parser.add_argument('--niter', type=int, default=100, help='total number of training epochs')
parser.add_argument('--num_M', type=int, default=32, help='the number of rain maps')
parser.add_argument('--num_Z', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=17, help='the number of iterative stages in RCDNet')
parser.add_argument('--resume', type=int, default=0, help='continue to train from epoch')
parser.add_argument("--milestone", type=int, default=[25,50,75], help="When to decay learning rate")
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument('--log_dir', default='./spa_logs/', help='tensorboard logs')
parser.add_argument('--model_dir',default='./spa_models/',help='saving model')
parser.add_argument('--manualSeed', type=int, help='manual seed')
opt = parser.parse_args()


if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device("cuda:0")

# create path
try:
    os.makedirs(opt.log_dir)
except OSError:
    pass
try:
    os.makedirs(opt.model_dir)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True


def train_model(net, optimizer, lr_scheduler,datasets):
    data_loader = DataLoader(datasets, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers),
                             pin_memory=True)
    num_data = len(datasets)
    num_iter_epoch = ceil(num_data / opt.batchSize)
    writer = SummaryWriter(opt.log_dir)
    step = 0
    for epoch in range(opt.resume,opt.niter):
        mse_per_epoch = 0
        tic = time.time()
        # train stage
        lr = optimizer.param_groups[0]['lr']
        phase = 'train'
        for ii, data in enumerate(data_loader):
            im_rain, im_gt = [x.cuda() for x in data]
            net.train()
            optimizer.zero_grad()
            B0, ListB, ListR = net(im_rain)
            loss_Bs = 0
            loss_Rs = 0
            for j in range(opt.S):
                loss_Bs = loss_Bs + 0.1 * F.mse_loss(ListB[j], im_gt)                # 2022-09-19 fix the bug
                loss_Rs = loss_Rs + 0.1 * F.mse_loss(ListR[j], im_rain - im_gt)      # 2022-09-19 fix the bug
            lossB = F.mse_loss(ListB[-1], im_gt)
            lossR = 0.9 *F.mse_loss(ListR[-1], im_rain - im_gt)
            lossB0 = 0.1 *F.mse_loss(B0, im_gt)
            loss = lossB0 + loss_Bs + lossB  + loss_Rs +  lossR
            # back propagation
            loss.backward()
            optimizer.step()
            mse_iter = loss.item()
            mse_per_epoch += mse_iter
            if ii % 300 == 0:
                template = '[Epoch:{:>2d}/{:<2d}] {:0>5d}/{:0>5d}, Loss={:5.2e}, lr={:.2e}'
                print(template.format(epoch+1, opt.niter, ii, num_iter_epoch, mse_iter, lr))
                writer.add_scalar('Train Loss Iter', mse_iter, step)
                writer.add_scalar('lossB', lossB.item(), step)
                writer.add_scalar('lossR', lossR.item(), step)
            step += 1
        mse_per_epoch /= (ii+1)
        print('Epoch:{:>2d}, Derain_Loss={:+.2e}'.format(epoch + 1, mse_per_epoch))
        # adjust the learning rate
        lr_scheduler.step()
        # save model
        model_prefix = 'model_'
        save_path_model = os.path.join(opt.model_dir, model_prefix+str(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'step': step+1,
            }, save_path_model)
        model_prefix = 'DerainNet_state_'
        save_path_model = os.path.join(opt.model_dir, model_prefix+str(epoch+1)+'.pt')
        torch.save(net.state_dict(), save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc-tic))
        print('-' * 100)
    writer.close()
    print('Reach the maximal epochs! Finish training')

if __name__ == '__main__':
    netDerain = RCDNet(opt).cuda()
    optimizerDerain = optim.Adam(netDerain.parameters(), lr=opt.lr)
    schedulerDerain = optim.lr_scheduler.MultiStepLR(optimizerDerain, milestones=opt.milestone, gamma=0.2)  # learning rates
    # from opt.resume continue to train
    for _ in range(opt.resume):
        schedulerDerain.step()
    if opt.resume:
        checkpoint = torch.load(os.path.join(opt.model_dir, 'model_' + str(opt.resume)))
        netDerain.load_state_dict(torch.load(os.path.join(opt.model_dir, 'DerainNet_state_' + str(opt.resume) + '.pt')))
        print('loaded checkpoints, epoch{:d}'.format(checkpoint['epoch']))

    # load dataset
    spa_files = open(os.path.join(opt.data_path,'real_world.txt'), 'r').readlines()
    train_dataset = SPATrainDataset(opt.data_path, spa_files, opt.patchSize, int(opt.batchSize * 5000), len(spa_files))
    # train model
    train_model(netDerain, optimizerDerain, schedulerDerain, train_dataset)

