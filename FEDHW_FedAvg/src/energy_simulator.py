import torch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import os
import csv
import sys
import pickle

wifi_BIT_RATE = 26000000  # unit: bps
g4g_BIT_RATE = 10000000  # unit: bps
TX_Power = 31  # unit: dBm
g4g_Power = 2.295  # unit: W


def get_local_energy(args, epochs, batch_size, fraction, weights, dataset, duration):
    cpu_energy = get_localcpu_energy(epochs, batch_size, args.client_num, fraction, dataset)
    gpu_energy = get_localgpu_energy(epochs, batch_size, args.client_num, fraction, dataset)
    dram_energy = get_dram_energy(duration, args.model)
    trans_energy = get_trans_energy(weights, args.client_num, fraction)

    if args.type == 'GPU':
        total_energy = cpu_energy + gpu_energy + dram_energy + trans_energy
        result = [float(cpu_energy), float(gpu_energy), float(dram_energy), float(trans_energy)]
        return float(total_energy), result
    else:
        total_energy = cpu_energy + tpu_energy + dram_energy + trans_energy
        result = [float(cpu_energy), float(tpu_energy), float(dram_energy), float(trans_energy)]
        return float(total_energy), result


def get_localcpu_energy(epochs, batch_size, client_num, fraction, dataset):
    if dataset == 'cifar10':
        file = open('../save/simulate_model/client_cpu_model_cifar10.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'mnist':
        file = open('../save/simulate_model/client_cpu_model_mnist.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    elif dataset == 'esc50':
        file = open('../save/simulate_model/client_cpu_model_esc50.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'r8':
        file = open('../save/simulate_model/client_cpu_model_lstm1.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    else:
        Exception('No such dataset model!')
    model = pickle.load(file)
    X = np.array([epochs, batch_size]).reshape(1, -1)
    poly.fit(X)
    X = poly.transform(X)
    y = model.predict(X)
    energy = y * client_num * fraction
    file.close()
    return energy


def get_localgpu_energy(epochs, batch_size, client_num, fraction, dataset):
    if dataset == 'cifar10':
        file = open('../save/simulate_model/client_gpu_model_cifar10.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'mnist':
        file = open('../save/simulate_model/client_gpu_model_mnist.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    elif dataset == 'esc50':
        file = open('../save/simulate_model/client_gpu_model_esc50.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'r8':
        file = open('../save/simulate_model/client_gpu_model_lstm1.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    else:
        Exception('No such dataset model!')
    model = pickle.load(file)
    X = np.array([epochs, batch_size]).reshape(1, -1)
    poly.fit(X)
    X = poly.transform(X)
    y = model.predict(X)
    energy = y * client_num * fraction
    file.close()
    return energy


def get_dram_energy(duration, model):
    resnet18_cifar_unit = 1.2789  # unit: W
    cnn_mnist_unit = 1.1195  # unit: W
    resnet18_esc50_unit = 1.121  # unit: W
    lstm_r8_unit = 1.224 # unit: W
    if model == 'resnet18':
        power = resnet18_cifar_unit
    elif model == 'cnn':
        power = cnn_mnist_unit
    elif model == 'audio':
        power = resnet18_esc50_unit
    elif model == 'lstm':
        power = lstm_r8_unit
    energy = power * duration
    return energy


def get_trans_energy(weights, client_num, fraction):
    # assume weight in every client is the same
    size = sys.getsizeof(weights) * 8  # unit: bit
    duration = size / g4g_BIT_RATE  # unit: s
    trans_power = 10 ** (TX_Power / 10) * 0.001  # unit: W
    # trans_power = g4g_Power
    energy = trans_power * duration  # unit: J
    energy = energy * client_num * fraction
    return energy


def cal_energy(args, iteration, epochs, batch_size, fraction, weights, duration, server_energy, energy_sum):
    local_energy, result = get_local_energy(args, epochs, batch_size, fraction, weights, args.dataset, duration)
    energy_total = local_energy + sum(server_energy)
    energy_sum.append(energy_total)
    file_dir = '../save/logs/'
    file_title = '{}_{}_{}_energy_record.txt'.format(args.dataset, args.model, args.energyOpt)
    path = file_dir + file_title
    with open(path, "a+") as f:
        f.write("Global iteration: {} \t Total Energy: {:.4f}J".format(iteration, energy_total) + '\n')
        f.write(
            "Server CPU energy: {:.2f}J \t Server GPU energy: {:.4f}J \tCPU energy: {:.4f}J \t GPU energy: {:.4f}J \t "
            "DRAM energy: {:.4f}J \t Transmission energy: {:.4f}J".format(
                server_energy[0], server_energy[1], result[0], result[1], result[2], result[3]) + '\n')
    return local_energy
