import math
from copy import deepcopy

import torch
from torch import nn
from torch.functional import F


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, init_lr * decay))


def poly_lr(optimizer, init_lr, curr_iter, max_iter, power=0.9):
    lr = init_lr * (1 - float(curr_iter) / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_poly(optimizer, init_lr, curr_iter, max_iter):
    warm_start_lr = 1e-7
    warm_steps = 1000

    if curr_iter <= warm_steps:
        warm_factor = (init_lr / warm_start_lr) ** (1 / warm_steps)
        warm_lr = warm_start_lr * warm_factor ** curr_iter
        for param_group in optimizer.param_groups:
            param_group['lr'] = warm_lr
    else:
        lr = init_lr * (1 - (curr_iter - warm_steps) / (max_iter - warm_steps)) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
