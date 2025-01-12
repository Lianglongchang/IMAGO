import argparse
import math
import os
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler

from get_dataset import get_ssl_datasets
from logger import setup_logger
import aslloss
from encoder import build_PreEncoder
from validation import evaluate_performance
from torch.nn.functional import normalize

import copy
import csv

from regularizer import Regularizer
import torch.nn.functional as F


def parser_args():
    parser = argparse.ArgumentParser(description='IMAGO self-supervised Training')
    parser.add_argument('--org', default='mouse', help='organism')
    parser.add_argument('--dataset_dir', help='dir of dataset',
                        default=r'~/IMAGO/Dataset/mouse')
    parser.add_argument('--aspect', type=str, choices=['P', 'F', 'C'], help='GO aspect')

    parser.add_argument('--output', default=r'~/IMAGO/Database/mouse/mouse_result',
                        metavar='DIR',
                        help='path to output folder')
    parser.add_argument('--num_class', default=45, type=int,
                        help="Number of class labels")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

    parser.add_argument('--seed', default=1329765522, type=int,
                        help='seed for initializing training. ')

    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length. ')

    parser.add_argument('--norm_norm', action='store_true', default=False,
                        help='using mormal scale to normalize input features')

    # * Transformer
    parser.add_argument('--attention_layers', default=6, type=int,
                        help="Number of layers of each multi-head attention module")

    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the multi-head attention blocks")
    parser.add_argument('--activation', default='gelu', type=str, choices=['relu', 'gelu', 'lrelu', 'sigmoid'],
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the multi-head attention module")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * raining
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args


def get_args():
    args = parser_args()
    return args


best_mAP = 0


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    torch.cuda.set_device(0)
    print('| Single GPU init (device 0)', flush=True)

    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=0, color=False, name="Q2L")

    args.h_n = 1
    return main_worker(args, logger)


def main_worker(args, logger):
    global best_mAP
    full_dataset, args.modesfeature_len = get_ssl_datasets(args)
    args.encode_structure = [1024]

    # build model
    pre_model = build_PreEncoder(args)
    pre_model = pre_model.cuda()

    # criterion
    pre_criterion = aslloss.pretrainLossOptimized(
        clip=args.loss_clip,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = 1
    # if args.optim == 'AdamW':
    pre_model_param_dicts = [
        {"params": [p for n, p in pre_model.named_parameters() if p.requires_grad]},
    ]
    pre_model_optimizer = getattr(torch.optim, 'AdamW')(
        pre_model_param_dicts,
        args.lr_mult * args.lr,
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    # tensorboard

    full_sampler = None
    full_loader = torch.utils.data.DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=(full_sampler is not None),
        num_workers=args.workers, pin_memory=True, sampler=full_sampler, drop_last=False)

    if args.evaluate:
        _, perf = validate(full_loader, pre_model, pre_criterion, args, logger)
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, losses_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    # pre_model_scheduler = lr_scheduler.OneCycleLR(pre_model_optimizer, max_lr=args.lr, steps_per_epoch=full_dataset[0].shape[0]//args.batch_size, epochs=args.epochs, pct_start=0.2)

    end = time.time()

    torch.cuda.empty_cache()

    pre_loss_f = args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_lr_' + str(
        args.lr) + '_seed_' + str(args.seed) + \
                 '_activation_' + str(args.activation) + '_pre_loss.csv'
    with open(pre_loss_f, 'w') as f:
        csv.writer(f).writerow(['pre_loss'])
    steplr = lr_scheduler.StepLR(pre_model_optimizer, 2500)
    for epoch in range(args.start_epoch, args.epochs):
        print('epoch=', epoch)
        pre_loss = pre_train(full_loader, pre_model, pre_criterion, pre_model_optimizer, steplr, epoch, args, logger)

        print('epoch={}, pre_loss={}'.format(epoch, pre_loss))

        with open(pre_loss_f, 'a') as f:
            csv.writer(f).writerow([pre_loss, pre_model_optimizer.param_groups[0]['lr']])

    torch.save(pre_model,
               args.output + '/' + args.org + '_attention_layers_' + str(args.attention_layers) + '_lr_' + str(
                   args.lr) + '_seed_' + str(args.seed) + \
               '_activation_' + str(args.activation) + '_model.pkl')


def pre_train(full_loader, pre_model, pre_criterion, optimizer, steplr, epoch, args, logger, clamp=0.01, embed_dim=512,
              reg_hidden_dim_2=32, reg_hidden_dim_1=64):
    losses = AverageMeter('Loss', ':5.3f')
    regularizer = Regularizer(embed_dim,
                              reg_hidden_dim_2, reg_hidden_dim_1)
    regularizer_optimizer = torch.optim.Adam(regularizer.parameters(),
                                             lr=args.lr)

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    pre_model.train()
    try:
        for i, proteins in enumerate(full_loader):
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            ori = copy.deepcopy(proteins)
            rec, hs = pre_model(proteins)

            x = hs[0]
            # print('x.shape=',x.shape)
            z = hs[1]
            # print('z.shape=',z.shape)
            for j in range(1):
                f_x = regularizer(x)
                f_z = regularizer(z)
                r1 = torch.normal(
                    0.0, 1.0, [x.shape[0], x.shape[1]]).cuda()
                f_r1 = regularizer(r1)
                r2 = torch.normal(
                    0.0, 1.0, [z.shape[0], z.shape[1]]).cuda()
                f_r2 = regularizer(r2)
                reg_loss = - f_r1.mean() + f_x.mean() - f_r2.mean() + f_z.mean()
                regularizer_optimizer.zero_grad()
                reg_loss.backward(retain_graph=True)
                regularizer_optimizer.step()

                for p in regularizer.parameters():
                    p.data.clamp_(-clamp, clamp)
            f_x = regularizer(x)
            # print('f_x=', f_x)
            f_z = regularizer(z)
            # print('f_z=', f_z)
            reg = -(f_x.mean() + f_z.mean())
            # print('x=', torch.numel(x))
            # print('z=', torch.numel(z))
            # print('f_x=', torch.numel(f_x))
            # print('f_z=', torch.numel(f_z)) + ((1 / len(x)) * cosine_similarity_loss(x, ori[0]) + (1 / len(z)) * cosine_similarity_loss(z, ori[1]))
            loss = pre_criterion(ori, rec, hs) + reg.item()
            if args.loss_dev > 0:
                loss *= args.loss_dev

            # record loss
            losses.update(loss.item(), rec[0].size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        steplr.step()
    except Exception as e:
        print(f"Error loading data: {e}")
    return losses.avg


##################################################################################
def add_weight_decay(pre_model, decoder_modal, weight_decay=1e-4, skip_list=()):
    pre_decay = []
    pre_no_decay = []
    decode_decay = []
    decode_no_decay = []
    for name, param in pre_model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            pre_no_decay.append(param)
        else:
            pre_decay.append(param)
    for name, param in decoder_modal.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            decode_no_decay.append(param)
        else:
            decode_decay.append(param)
    return [
        {'params': pre_no_decay, 'weight_decay': 0.},
        {'params': pre_decay, 'weight_decay': weight_decay},
        {'params': decode_no_decay, 'weight_decay': 0.},
        {'params': decode_decay, 'weight_decay': weight_decay}]


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""

    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name,
                             val=str(datetime.timedelta(seconds=int(self.val))),
                             sum=str(datetime.timedelta(seconds=int(self.sum))))


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class myRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data, generator, shuffle=True):
        self.data = data
        self.generator = generator
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            return iter(torch.randperm(n, generator=self.generator).tolist())
        else:
            return iter(list(range(n)))

    def __len__(self):
        return len(self.data)


def kill_process(filename: str, holdpid: int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True,
                                  cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


def cosine_similarity_loss(x, y):
    """
    Calculate cosine similarity loss
    """
    if x.size(1) != y.size(1):
        # If the dimensions are different, reduce the dimension of Y.
        y = y[:, :x.size(1)]

    x_norm = F.normalize(x, p=2, dim=-1)
    y_norm = F.normalize(y, p=2, dim=-1)
    cos_sim = torch.sum(x_norm * y_norm, dim=-1)
    return torch.mean(1 - cos_sim)


if __name__ == '__main__':
    main()
