import pynvml
import pandas as pd
from datetime import datetime
import pdb
import math
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import copy


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


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


class cosine_annealing_lr:
    def __init__(self, init_lr, max_lr, min_lr, rounds):
        self.rounds = rounds
        self.current_lr = init_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch != self.rounds:
            lr = self.min_lr + 1 / 2 * (self.max_lr - self.min_lr) * (1 + np.cos(epoch / self.rounds * np.pi))
        else:
            lr = self.min_lr

        return lr


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
