import pickle
import sys
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

BIT_RATE = 26000000  # unit: bps


def get_local_latency(args, epoch, batch_size, fraction, info, dataset):
    train_latency = get_train_part(epoch, batch_size, args.client_num, fraction, dataset)
    trans_latency = get_trans_part(info)
    total_latency = train_latency + trans_latency
    result = [float(train_latency), float(trans_latency)]
    return float(total_latency), result


def get_train_part(epoch, batch_size, client_num, fraction, dataset):
    file = None
    if dataset == 'cifar10':
        file = open('../save/simulate_model/client_latency_model_cifar10.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'mnist':
        file = open('../save/simulate_model/client_latency_model_mnist.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    elif dataset == 'esc50':
        file = open('../save/simulate_model/client_latency_model_esc50.pkl', 'rb')
        poly = PolynomialFeatures(degree=2)
    elif dataset == 'r8':
        file = open('../save/simulate_model/client_latency_model_lstm1.pkl', 'rb')
        poly = PolynomialFeatures(degree=2, include_bias=False)
    else:
        print('No such dataset model!')
    model = pickle.load(file)
    X = np.array([epoch, batch_size]).reshape(1, -1)
    poly.fit(X)
    X = poly.transform(X)
    y = model.predict(X)
    latency = y
    file.close()

    return latency


def get_trans_part(info):
    size = sys.getsizeof(info) * 8  # unit: bit
    duration = size / BIT_RATE  # unit: s
    duration = duration * 2

    return duration


def cal_latency(args, iteration, epochs, batch_size, fraction, weights, server_latency, latency_sum):
    local_latency, result = get_local_latency(args, epochs, batch_size, fraction, weights, args.dataset)
    latency = local_latency + server_latency
    latency_sum.append(latency)
    file_dir = '../save/logs/'
    file_title = '{}_{}_{}_latency_record.txt'.format(args.dataset, args.model, args.latencyOpt)
    path = file_dir + file_title
    with open(path, "a+") as f:
        f.write("Global iteration: {} \t Training latency: {:.2f}s".format(iteration, latency) + '\n')
        f.write(
            "Server Latency: {:.2f}s \t Train Latency: {:.2f}s \t Trans Latency: {:.4f}s".format(
                server_latency, result[0], result[1]) + '\n')

    return local_latency
