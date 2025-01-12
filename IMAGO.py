import argparse
import math
import os
import random
import datetime
import time
from typing import List
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from get_dataset import get_datasets
from encoder import build_PreEncoder
import aslloss
from predictor_module import build_predictor
from validation import evaluate_performance
from torch.nn.functional import normalize
from logger import setup_logger
import copy
import csv

from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE

os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

def parser_args():
    parser = argparse.ArgumentParser(description='IMAGO main')
    parser.add_argument('--org', help='organism', default='human')
    parser.add_argument('--dataset_dir', help='dir of dataset', default= r'~/IMAGO/Dataset/human')
    parser.add_argument('--aspect', type=str, choices=['P', 'F', 'C'], help='GO aspect', default='P')
    parser.add_argument('--pretrained_model', type=str, help='pretrained self-supervide learning model', default= r'~/IMAGO/Database/human/human_result/human_attention_layers_6_lr_1e-05_seed_1329765522_activation_gelu_model.pkl')

    parser.add_argument('--output', metavar='DIR', default= r'~/IMAGO/Database/human/human_result',
                        help='path to output folder')
    parser.add_argument('--num_class', default=45, type=int,
                        help="Number of class labels")
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False,
                        help='disable_torch_grad_focal_loss in asl')
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                        help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                        help='scale factor for clip')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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

    parser.add_argument('--seed', default=1329765522, type=int,
                        help='seed for initializing training.')

    # data aug
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=-1,
                        help='length of the holes. suggest to use default setting -1.')
    parser.add_argument('--cut_fact', type=float, default=0.5,
                        help='mutual exclusion with length.')

    parser.add_argument('--norm_norm', action='store_true', default=False,
                        help='using normal scale to normalize input features')

    # * Transformer
    parser.add_argument('--attention_layers', default=6, type=int,
                        help="Number of layers of each multi-head attention module")

    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the multi-head attention blocks")
    parser.add_argument('--activation', default='gelu', type=str, choices=['relu', 'gelu', 'lrelu', 'sigmoid'],
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--dropout', default=0.3, type=float,
                        help="Dropout applied in the multi-head attention module")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the multi-head attention module's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * training
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


def set_rand_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = get_args()

    if args.seed is not None:
        set_rand_seed(args.seed)

    torch.cuda.set_device(0)
    cudnn.benchmark = True

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, color=False, name="IMAGO")

    args.h_n = 1
    return main_worker(args, logger)


def main_worker(args, logger):
    global best_mAP
    train_dataset, test_dataset, args.modesfeature_len = get_datasets(args)
    criterion = aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 32

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)

    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, losses_ema],
        prefix='=> Test Epoch: ')

    end = time.time()
    best_epoch = -1
    best_finetune_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_regular_finetune_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()

    fn = os.path.join(args.output,
                      f'{args.org}_attention_layers_{args.attention_layers}_aspect_{args.aspect}_fintune_seed_{args.seed}_act_{args.activation}.csv')
    with open(fn, 'w') as f:
        csv.writer(f).writerow(['m-aupr', 'M-aupr', 'F1', 'acc', 'Fmax'])

    for epoch in range(1):
        if args.seed is not None:
            set_rand_seed(args.seed)
        torch.cuda.empty_cache()

        finetune_pre_model = torch.load(args.pretrained_model)
        predictor_model = build_predictor(finetune_pre_model, args)
        predictor_model = predictor_model.cuda()

        predictor_model_param_dicts = [
            {"params": [p for n, p in predictor_model.pre_model.named_parameters() if p.requires_grad], "lr": 1e-5},
            {"params": [p for n, p in predictor_model.fc_decoder.named_parameters() if p.requires_grad]}
        ]

        predictor_model_optimizer = getattr(torch.optim, 'AdamW')(
            predictor_model_param_dicts,
            lr=args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
        steplr = lr_scheduler.StepLR(predictor_model_optimizer, 50)
        patience = 10
        changed_lr = False
        for epoch_train in range(100):
            # train for one epoch
            train_loss = train(train_loader, predictor_model, criterion,
                               predictor_model_optimizer, steplr, epoch_train, args, logger)
            print('loss = ', train_loss)

            old_loss = train_loss

        # 在训练完成后调用 evaluate
        loss, perf = evaluate(test_loader, predictor_model, criterion, args, logger)

        with open(fn, 'a') as f:
            csv.writer(f).writerow([perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc'], perf['Fmax']])

    # 合并训练集和测试集的数据，进行 t-SNE 可视化
    visualize_embeddings(train_loader, test_loader, predictor_model, args)

    return 0


def train(train_loader, predictor_model, criterion, optimizer, steplr, epoch_train, args, logger):
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    def get_learning_rate(optimizer):
        return optimizer.param_groups[1]["lr"]

    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    predictor_model.train()

    if epoch_train >= 50:
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = True
    else:
        for p in predictor_model.pre_model.parameters():
            p.requires_grad = False

    end = time.time()
    for i, (proteins, label) in enumerate(train_loader):
        proteins[0] = proteins[0].cuda()
        proteins[1] = proteins[1].cuda()
        label = label.cuda()

        rec, output = predictor_model(proteins)
        loss = criterion(rec, output, label)
        if args.loss_dev > 0:
            loss *= args.loss_dev

        losses.update(loss.item(), proteins[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    steplr.step()
    return losses.avg

# Calculation of adding Davies-Bouldin score to evaluate function
@torch.no_grad()
def evaluate(test_loader, predictor_model, criterion, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, mem],
        prefix='Test: ')

    predictor_model.eval()
    saved_data = []
    embeddings = []
    labels = []

    with torch.no_grad():
        for i, (proteins, label) in enumerate(test_loader):
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            label = label.cuda()

            rec, output = predictor_model(proteins)
            loss = criterion(rec, output, label)
            if args.loss_dev > 0:
                loss *= args.loss_dev
            output_sm = nn.functional.sigmoid(output)
            if torch.isnan(loss):
                saveflag = True

            losses.update(loss.item(), proteins[0].size(0))

            _item = torch.cat((output_sm.detach().cpu(), label.detach().cpu()), 1)
            saved_data.append(_item)

            embeddings.append(output_sm.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())

            if i % args.print_freq == 0:
                progress.display(i, logger)

        loss_avg = losses.avg

        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.txt'
        np.savetxt(os.path.join(args.output, saved_name), saved_data)

        y_score = saved_data[:, 0:(saved_data.shape[1] // 2)]
        labels_val = saved_data[:, (saved_data.shape[1] // 2):]
        perf = evaluate_performance(labels_val, y_score, (y_score > 0.5).astype(int))
        print(
            '%0.5f %0.5f %0.5f %0.5f %0.5f\n' % (perf['m-aupr'], perf['M-aupr'], perf['F1'], perf['acc'], perf['Fmax']))

        # Calculate Davies-Bouldin score
        embeddings = np.concatenate(embeddings, axis=0)
        labels = np.concatenate(labels, axis=0)
        db_score = davies_bouldin_score(embeddings, labels.argmax(axis=1))
        print(f"Davies-Bouldin Score: {db_score}")

        fn = os.path.join(args.output, f'{args.org}_attention_layers_{args.attention_layers}_aspect_{args.aspect}_fintune_seed_{args.seed}_act_{args.activation}.csv')
        with open(fn, 'a') as f:
            csv.writer(f).writerow([db_score])

    return loss_avg, perf

# Visualization function
@torch.no_grad()
def visualize_embeddings(train_loader, test_loader, predictor_model, args):
    predictor_model.eval()

    train_embeddings = []
    train_labels = []
    test_embeddings = []
    test_labels = []

    with torch.no_grad():
        for proteins, label in train_loader:
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            label = label.cuda()

            rec, output = predictor_model(proteins)
            output_sm = nn.functional.sigmoid(output)

            train_embeddings.append(output_sm.detach().cpu().numpy())
            train_labels.append(label.detach().cpu().numpy())

        for proteins, label in test_loader:
            proteins[0] = proteins[0].cuda()
            proteins[1] = proteins[1].cuda()
            label = label.cuda()

            rec, output = predictor_model(proteins)
            output_sm = nn.functional.sigmoid(output)

            test_embeddings.append(output_sm.detach().cpu().numpy())
            test_labels.append(label.detach().cpu().numpy())

    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Merge the data of training set and test set.
    all_embeddings = np.concatenate([train_embeddings, test_embeddings], axis=0)
    all_labels = np.concatenate([train_labels, test_labels], axis=0)

    # Visualization using t-SNE
    print("Running t-SNE on combined embeddings...")
    tsne = TSNE(n_components=2, random_state=0)

    # print(f"Shape of all embeddings: {all_embeddings.shape}")
    # print(f"First row of embeddings: {all_embeddings[0]}")

    try:
        tsne_result = tsne.fit_transform(all_embeddings)
        print("t-SNE completed.")
    except Exception as e:
        print(f"Error during t-SNE computation: {e}")
        return

    # Normalize t-SNE results to the range of [0, 1]
    tsne_min = tsne_result.min(axis=0)
    tsne_max = tsne_result.max(axis=0)
    tsne_result = (tsne_result - tsne_min) / (tsne_max - tsne_min)

    plt.figure(figsize=(16, 16))
    sns.scatterplot(
        x=tsne_result[:, 0], y=tsne_result[:, 1],
        hue=all_labels.argmax(axis=1),
        palette=sns.color_palette("hsv", as_cmap=True),
        legend="full",
        alpha=0.6,
        s=50
    )
    output_filename = f'tsne_visualization_{args.org}_{args.aspect}.png'
    plt.savefig(os.path.join(args.output, output_filename))
    plt.close()

def add_weight_decay(pre_model, decoder_modal, weight_decay=1e-4, skip_list=()):
    pre_decay = []
    pre_no_decay = []
    decode_decay = []
    decode_no_decay = []
    for name, param in pre_model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            pre_no_decay.append(param)
        else:
            pre_decay.append(param)
    for name, param in decoder_modal.named_parameters():
        if not param.requires_grad:
            continue
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
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device
        if (self.device is not None):
            self.module.to(device=self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if (self.device is not None):
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=(lambda e, m: (self.decay * e) + ((1. - self.decay) * m)))

    def set(self, model):
        self._update(model, update_fn=(lambda e, m: m))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
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


if __name__ == '__main__':
    main()
