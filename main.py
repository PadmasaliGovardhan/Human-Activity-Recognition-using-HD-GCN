#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
import traceback
import glob
import csv
from collections import OrderedDict

import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from torchlight import DictAction

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError(
            f'Class {class_str} cannot be found:\n'
            f'{traceback.format_exc()}'
        )


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Unsupported value encountered.')


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')

    parser.add_argument('--work-dir', default='./work_dir/temp')
    parser.add_argument('--config', default=None)

    # processor
    parser.add_argument('--phase', default='train')
    parser.add_argument('--save-score', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--save-epoch', type=int, default=30)
    parser.add_argument('--eval-interval', type=int, default=5)
    parser.add_argument('--print-log', type=str2bool, default=True)
    parser.add_argument('--show-topk', type=int, nargs='+', default=[1, 5])

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder')
    parser.add_argument('--num-worker', type=int, default=32)
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict())
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict())

    # model
    parser.add_argument('--model', default=None)
    parser.add_argument('--model-args', action=DictAction, default=dict())
    parser.add_argument('--weights', default=None)
    parser.add_argument('--ignore-weights', nargs='+', default=[])

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01)
    parser.add_argument('--step', type=int, nargs='+', default=[20, 40, 60])
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--nesterov', type=str2bool, default=False)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--num-epoch', type=int, default=80)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--lr-ratio', type=float, default=0.001)
    parser.add_argument('--lr-decay-rate', type=float, default=0.1)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--loss-type', type=str, default='CE')

    return parser


# ------------------------------------------------------------------
# Processor
# ------------------------------------------------------------------
class Processor():

    def __init__(self, arg):
        self.arg = arg

        # -------- unified device handling --------
        if isinstance(arg.device, list) and len(arg.device) > 0 and torch.cuda.is_available():
            self.output_device = arg.device[0]
            self.device = torch.device(f'cuda:{self.output_device}')
        else:
            self.output_device = None
            self.device = torch.device('cpu')
        # -----------------------------------------

        self.save_arg()
        self.global_step = 0
        self.best_acc = 0
        self.best_acc_epoch = 0

        if arg.phase == 'train':
            self.train_writer = SummaryWriter(os.path.join(arg.work_dir, 'train'))
            self.val_writer = SummaryWriter(os.path.join(arg.work_dir, 'val'))

        self.load_model()
        self.load_optimizer()
        self.load_data()

        if isinstance(arg.device, list) and len(arg.device) > 1:
            self.model = nn.DataParallel(
                self.model,
                device_ids=arg.device,
                output_device=self.output_device
            )

    # ------------------------------------------------

    def load_model(self):
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)

        self.model = Model(**self.arg.model_args).to(self.device)

        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().to(self.device)
        else:
            self.loss = LabelSmoothingCrossEntropy(0.1).to(self.device)

        if self.arg.weights:
            self.print_log(f'Loading weights from {self.arg.weights}')
            weights = torch.load(self.arg.weights, map_location=self.device)

            weights = OrderedDict(
                (k.split('module.')[-1], v.to(self.device))
                for k, v in weights.items()
            )

            for w in self.arg.ignore_weights:
                for k in list(weights.keys()):
                    if w in k:
                        weights.pop(k)

            self.model.load_state_dict(weights, strict=False)

    # ------------------------------------------------

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer')

    # ------------------------------------------------

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = {}

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False)

    # ------------------------------------------------

    def save_arg(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.arg), f)

    # ------------------------------------------------

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_list = []
        acc_list = []

        for data, label, _ in tqdm(loader):
            data = data.float().to(self.device)
            label = label.long().to(self.device)

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())
            acc = (output.argmax(1) == label).float().mean().item()
            acc_list.append(acc)

        self.print_log(
            f'Epoch {epoch+1}: '
            f'Loss={np.mean(loss_list):.4f}, '
            f'Acc={np.mean(acc_list)*100:.2f}%'
        )

    # ------------------------------------------------

    def eval(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        correct, total = 0, 0

        with torch.no_grad():
            for data, label, _ in tqdm(loader):
                data = data.float().to(self.device)
                label = label.long().to(self.device)

                output = self.model(data)
                pred = output.argmax(1)

                correct += (pred == label).sum().item()
                total += label.size(0)

        acc = correct / total
        self.print_log(f'Validation accuracy: {acc*100:.2f}%')
        return acc

    # ------------------------------------------------

    def print_log(self, msg):
        print(msg)
        if self.arg.print_log:
            with open(f'{self.arg.work_dir}/log.txt', 'a') as f:
                print(msg, file=f)

    # ------------------------------------------------

    def start(self):
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train(epoch)
                acc = self.eval(epoch)

                if acc > self.best_acc:
                    self.best_acc = acc
                    self.best_acc_epoch = epoch + 1
                    torch.save(
                        self.model.state_dict(),
                        f'{self.arg.work_dir}/best_model.pt'
                    )

            self.print_log(f'Best Acc: {self.best_acc*100:.2f}% '
                           f'at epoch {self.best_acc_epoch}')

        else:
            self.eval(0)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == '__main__':
    parser = get_parser()
    p = parser.parse_args()

    if p.config:
        with open(p.config) as f:
            parser.set_defaults(**yaml.safe_load(f))

    arg = parser.parse_args()
    init_seed(arg.seed)

    processor = Processor(arg)
    processor.start()
