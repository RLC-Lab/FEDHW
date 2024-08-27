# -*- coding: utf-8 -*-
# python version: 3.11
import copy
from datetime import datetime
import pandas as pd
import pynvml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Optimizer
import numpy as np
import math


def inference(gpuid, model, test_dataset, loss_func):
    device = torch.device('cuda', gpuid)
    model.to(device)
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    if loss_func == 'nll':
        criterion = nn.NLLLoss().to(device)
    elif loss_func == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total

    return accuracy


def lstm_inference(gpuid, model, test_dataset):
    device = torch.device('cuda', gpuid)
    model.to(device)
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        labels = torch.squeeze(labels)
        inputs, labels = Variable(inputs.cuda()), labels.cuda()
        model.batch_size = len(labels)
        model.hidden = model.init_hidden()
        output = model(inputs.t())
        loss = criterion(output, Variable(labels))
        # calc testing acc
        _, predicted = torch.max(output.data, 1)
        total_acc += (predicted == labels).sum()
        total += len(labels)
        total_loss += loss.item()
    test_loss = total_loss / total
    test_acc = total_acc / total

    return test_acc


class cosine_annealing_lr:
    def __init__(self, init_lr, max_lr, min_lr, epochs):
        self.total_epochs = epochs
        self.current_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch != self.total_epochs:
            lr = self.min_lr + 1 / 2 * (self.max_lr - self.min_lr) * (1 + np.cos(epoch / self.total_epochs * np.pi))
        else:
            lr = self.min_lr

        return lr


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class FedadamOptimizer(Optimizer):
    def __init__(self, params, **kwargs):
        lr = kwargs.get('lr')
        v0 = kwargs.get('v0')
        tau = kwargs.get('tau')
        momentum = kwargs.get('betas')
        defaults = dict(lr=lr, momentum=momentum, v0=v0, tau=tau)
        Optimizer.__init__(self, params=params, defaults=defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for idx, group in enumerate(self.param_groups):
            (beta1, beta2) = group['momentum']
            tau = group['tau']
            lr = group['lr']
            v0 = group['v0']
            for param in group['params']:
                if param.grad is None:
                    continue
                delta = -param.grad.data

                if idx == 0:
                    if 'momentum_buffer1' not in self.state[param]:
                        self.state[param]['momentum_buffer1'] = torch.zeros_like(param).detach()
                    self.state[param]['momentum_buffer1'].mul_(beta1).add_(
                        delta.mul(1. - beta1))  # \beta1 * m_t + (1 - \beta1) * \Delta_t
                    m_new = self.state[param]['momentum_buffer1']

                    # calculate v_t
                    if 'momentum_buffer2' not in self.state[param]:
                        self.state[param]['momentum_buffer2'] = v0 * beta2 + delta.pow(2).mul(1. - beta2)
                    self.state[param]['momentum_buffer2'].mul_(beta2).add_(
                        delta.pow(2).mul(1. - beta2))  # \beta2 * v_t + (1 - \beta2) * \Delta_t^2
                    v_new = self.state[param]['momentum_buffer2']

                    param.data.add_(m_new.div(v_new.pow(0.5).add(tau)).mul(lr))
                elif idx == 1:  # idx == 1: buffers; just averaging
                    param.data.add_(delta)
        return loss

    def accumulate(self, client_count, local_layers_iterator, check_if=lambda name: 'num_batches_tracked' in name):
        for group in self.param_groups:
            for server_param, (name, local_signals) in zip(group['params'], local_layers_iterator):
                if check_if(name):
                    server_param.data.zero_()
                    server_param.data.grad = torch.zeros_like(server_param)
                    continue
                local_delta = (server_param - local_signals).mul(1/client_count).data.type(server_param.dtype)
                if server_param.grad is None:  # NOTE: grad buffer is used to accumulate local updates!
                    server_param.grad = local_delta
                else:
                    server_param.grad.data.add_(local_delta)


def get_gpu_power():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    pynvml.nvmlShutdown()
    return powerusage


def get_server_energy(ag_time1, ag_time2, gpu1, gpu2, gpu3):
    df = pd.read_csv("/home/kylepan/outputs.csv", header=1, dtype={'Time': str})
    duration = ag_time2 - ag_time1
    date_time = datetime.fromtimestamp(ag_time1)
    time1 = date_time.time()
    time1 = f"{time1.hour:02}:{time1.minute:02}:{time1.second:02}.{int(time1.microsecond / 10000):02}"
    date_time = datetime.fromtimestamp(ag_time2)
    time2 = date_time.time()
    time2 = f"{time2.hour:02}:{time2.minute:02}:{time2.second:02}.{int(time2.microsecond / 10000):02}"

    time_column = df['Time'].astype(str)
    i = 0
    begin = 0
    for time in time_column:
        ftime = time[:-1]
        if begin == 0 and ftime == time1:
            begin = i
            break
        else:
            i += 1
    energy_column = df['Proc Energy (Joules)']
    cpu_energy = 0.0
    j = begin
    while j < len(energy_column):
        if math.isnan(energy_column[j]):
            j += 1
            continue
        cpu_energy += float(energy_column[j]) - 0.1
        j += 1

    gpu_power = (gpu1+gpu2+gpu3) / 3
    gpu_energy = gpu_power * duration

    return cpu_energy, gpu_energy
