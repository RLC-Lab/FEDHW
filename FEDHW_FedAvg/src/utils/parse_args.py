# -*- coding: utf-8 -*-
# python version: 3.11

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--data_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--client_num', type=int, default=100, help='number of clients')
    parser.add_argument('--type', type=str, default='GPU', help='type of training device')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--criterion', type=str, default='nll', help='loss function')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--energyOpt', type=str, default=None, help='clm energy algorithm')
    parser.add_argument('--latencyOpt', type=str, default=None, help='clm latency algorithm')
    parser.add_argument('--gpuid', type=int, default=0, help='GPU ID')

    parser.add_argument('--rounds', type=int, default=100, help='number of global rounds')
    parser.add_argument('--epochs', type=int, default=20, help='number of local epochs of clients training')
    parser.add_argument('--b_size', type=int, default=10, help='batch size of data on the local clients')
    parser.add_argument('--fraction', type=float, default=0.1, help='selected clients involved in the training')

    parser.add_argument('--accuracy', type=float, default=0, help='set the accuracy goal')

    args = parser.parse_args()
    return args


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Train type: {args.type}')
    print(f'    Number of clients: {args.client_num}')
    print(f'    Learning rate    : {args.lr}')
    print(f'    Criterion        : {args.criterion}')
    print(f'    Optimizer        : {args.optimizer}')
    print(f'    Fraction of clients: {args.fraction}')
    print(f'    Local Epochs       : {args.epochs}')
    print(f'    Local Batch size   : {args.b_size}')
    if args.energyOpt is not None:
        print(f'    Energy Opt   : {args.energyOpt}')
    elif args.latencyOpt is not None:
        print(f'    Latency Opt   : {args.latencyOpt}')
    print(f'    Target Accuracy    : {args.accuracy}')

    return
