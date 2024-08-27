import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
from src.utils.dataset import (DatasetSplit)


def train(args, model, train_dataset, train_idx, epochs, batch_size, iteration, lr):
    device = torch.device('cuda', args.gpuid)
    model.to(device)
    model.train()
    start = time.time()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataloader = get_train_loader(train_dataset, train_idx, batch_size)
    if args.criterion == 'nll':
        criterion = nn.NLLLoss().to(device)
    elif args.criterion == 'ce':
        criterion = nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in range(epochs):
        batch_loss = 0.0
        count = 0
        for batch_idx, (data, labels) in enumerate(train_dataloader):
            data, labels = data.to(device), labels.to(device)

            log_probs = model(data)
            loss = criterion(log_probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss.data.item()
            count += 1
        epoch_loss.append(batch_loss / count)
        print("Global Round: {} \t Local Epoch: {} \t Loss: {:.6f}".format(iteration, epoch+1, epoch_loss[-1]))
    end = time.time()
    duration = end - start

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss), duration


def lstm_train(args, model, train_dataset, train_idx, epochs, batch_size, iteration, lr):
    device = torch.device('cuda', args.gpuid)
    model.to(device)
    model.train()
    start = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(device)
    train_dataloader = get_train_loader(train_dataset, train_idx, batch_size)
    epoch_loss = []

    for epoch in range(epochs):
        batch_loss = 0.0
        count = 0.0
        for idx, (data, labels) in enumerate(train_dataloader):
            labels = torch.squeeze(labels)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            data, labels = Variable(data.cuda()), labels.cuda()

            optimizer.zero_grad()
            model.batch_size = len(labels)
            model.hidden = model.init_hidden()
            log_probs = model(data.t())
            loss = criterion(log_probs, Variable(labels))
            loss.backward()
            optimizer.step()

            count += 1
            batch_loss += loss.data.item()
        epoch_loss.append(batch_loss / count)
        print("Global Round: {} \t Local Epoch: {} \t Loss: {:.6f}".format(iteration, epoch + 1, epoch_loss[-1]))
    end = time.time()
    duration = end - start

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss), duration


def get_train_loader(train_dataset, train_idx, batch_size):
    trainloader = DataLoader(DatasetSplit(train_dataset, train_idx),  batch_size=batch_size, shuffle=True)
    return trainloader
