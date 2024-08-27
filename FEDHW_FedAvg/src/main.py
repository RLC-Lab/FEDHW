# -*- coding: utf-8 -*-
# python version: 3.11
import copy
import gc
import time
import torch
import numpy as np
import subprocess
import warnings
from datetime import datetime
import pdb

warnings.filterwarnings("ignore")
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
from utils.parse_args import parse_args, exp_details
from utils.model_lib import get_model
from utils.dataset import get_data
from utils import local_client
import clever_little_man as clm
from server import get_server_energy, get_gpu_power, cosine_annealing_lr, inference, lstm_inference, average_weights
from energy_simulator import cal_energy
from latency_simulator import cal_latency


if __name__ == "__main__":
    print("FedAvgEF Start!")
    np.random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)

    # parse arguments
    args = parse_args()
    epochs = args.epochs
    batch_size = args.b_size
    fraction = args.fraction
    exp_details(args)

    # get model
    global_model = get_model(args)
    train_dataset, test_dataset, user_train_groups, user_test_groups = get_data(args.dataset, args.client_num)
    # set gpu:0 as the training device
    device = torch.device('cuda', args.gpuid)
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Train
    file_dir = '../save/logs/'
    if args.energyOpt is not None:
        file_title = '{}_{}_EnergyOpt{}_training_record.txt'.format(args.dataset, args.model, args.energyOpt)
    elif args.latencyOpt is not None:
        file_title = '{}_{}_LatencyOpt{}_training_record.txt'.format(args.dataset, args.model, args.latencyOpt)
    else:
        file_title = '{}_{}_E{}_B{}_F{}_training_record.txt'.format(args.dataset, args.model, args.epochs,
                                                                    args.b_size, args.fraction)
    path = file_dir + file_title
    start_time = time.time()
    count = 0
    last_accuracy = 0  # record the last accuracy
    accuracy = 0  # global model test accuracy
    iteration = 1  # record the training rounds
    lr = args.lr
    scheduler = cosine_annealing_lr(init_lr=lr, max_lr=lr, min_lr=1e-6, rounds=args.rounds)
    if args.latencyOpt is not None:
        latency_sum = []
        if args.latencyOpt == 'GA':
            GA_latency = clm.Latency_genetic_algorithm(batch_size_max=40, fraction_min=0.1)
    elif args.energyOpt is not None:
        energy_sum = []
        if args.energyOpt == 'GA':
            GA_energy = clm.Energy_genetic_algorithm()

    while accuracy < args.accuracy:
        local_weights, local_losses = [], []
        train_duration = 0.0
        print("\nGlobal Training Epoch: {}".format(iteration))
        global_weights = global_model.state_dict()
        chosen = max(int(fraction * args.client_num), 1)
        chosen_client = np.random.choice(range(args.client_num), chosen, replace=False)

        for client in chosen_client:
            local_model = get_model(args)
            local_model.load_state_dict(global_weights)
            print("Client {} is training".format(client))
            if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'audio' or args.model == 'resnet34':
                weight, loss, duration = local_client.train(args, local_model, train_dataset, user_train_groups[client],
                                                            epochs, batch_size, iteration, lr)
            elif args.model == 'lstm':
                weight, loss, duration = local_client.lstm_train(args, local_model, train_dataset,
                                                                 user_train_groups[client],
                                                                 epochs, batch_size, iteration, lr)
            local_weights.append(copy.deepcopy(weight))
            local_losses.append(copy.deepcopy(loss))
            train_duration += duration
            del local_model
            gc.collect()
            torch.cuda.empty_cache()

        if args.energyOpt is not None:
            server_energy = []
            count += 1
            pcm_com = ['/Github/pcm/build/bin/pcm', '0.01', '-r', '-csv=/home/username/outputs.csv']
            pcm_process = subprocess.Popen(pcm_com)
            time.sleep(3)
            gpu_power1 = get_gpu_power()
            ag_time1 = time.time()
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            gpu_power2 = get_gpu_power()
            if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'audio':
                test_acc = inference(args.gpuid, global_model, test_dataset, args.criterion)
            elif args.model == 'lstm':
                test_acc = lstm_inference(args.gpuid, global_model, test_dataset)
            former_accuracy = accuracy
            accuracy = test_acc
            print("Global Training Iteration: {} \t Train Accuracy: {:.2f}% \t Test Loss: {:.5f}".format(
                iteration, 100 * accuracy, np.mean(np.array(local_losses))))
            ag_time2 = time.time()
            ag_duration = ag_time2 - ag_time1
            gpu_power3 = get_gpu_power()
            pcm_process.kill()
            cpu_energy, gpu_energy = get_server_energy(ag_time1, ag_time2, gpu_power1, gpu_power2, gpu_power3)
            server_energy.append(cpu_energy)
            server_energy.append(gpu_energy)
            local_energy = cal_energy(args, iteration, epochs, batch_size, fraction, global_weights, train_duration,
                                      server_energy, energy_sum)
        elif args.latencyOpt is not None:
            ag_time1 = time.time()
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'audio' or args.model == 'resnet34':
                test_acc = inference(args.gpuid, global_model, test_dataset, args.criterion)
            elif args.model == 'lstm':
                test_acc = lstm_inference(args.gpuid, global_model, test_dataset)
            former_accuracy = accuracy
            accuracy = test_acc
            print("Global Training Iteration: {} \t Train Accuracy: {:.2f}% \t Test Loss: {:.5f}".format(
                iteration, 100 * accuracy, np.mean(np.array(local_losses))))
            ag_time2 = time.time()
            ag_duration = ag_time2 - ag_time1
            local_latency = cal_latency(args, iteration, epochs, batch_size, fraction, global_weights, ag_duration, latency_sum)
        else:
            global_weights = average_weights(local_weights)
            global_model.load_state_dict(global_weights)
            if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'audio' or args.model == 'resnet34':
                test_acc = inference(args.gpuid, global_model, test_dataset, args.criterion)
            elif args.model == 'lstm':
                test_acc = lstm_inference(args.gpuid, global_model, test_dataset)
            former_accuracy = accuracy
            accuracy = test_acc
            print("Global Training Iteration: {} \t Train Accuracy: {:.2f}% \t Test Loss: {:.5f}".format(
                iteration, 100 * accuracy, np.mean(np.array(local_losses))))

        with open(path, 'a+') as f:
            f.write("Global Training Iteration: {} \t Test Accuracy: {:.2f}% \t Train Loss: {:.5f}\n".format(
                iteration, 100 * accuracy, np.mean(np.array(local_losses))))
            f.write("Epoch: {} \t Batch size: {} \t Fraction: {} \t".format(epochs, batch_size, fraction) + '\n')
            f.write("Chosen clients: {}".format(chosen_client) + '\n')

        if accuracy < args.accuracy:
            if args.energyOpt == 'SA':
                epochs, batch_size, fraction = clm.Energy_sim_annealing(args, epochs, batch_size, fraction,
                                                                        former_accuracy, accuracy, global_model.state_dict(), train_duration, local_energy, energy_sum)
            elif args.energyOpt == 'GA':
                epochs, batch_size, fraction = GA_energy.run(args, epochs, batch_size, fraction, former_accuracy,
                    accuracy, global_model.state_dict(), train_duration, local_energy, energy_sum)
            elif args.latencyOpt == 'SA':
                epochs, batch_size, fraction = clm.Latency_sim_annealing(args, epochs, batch_size, fraction,
                    former_accuracy, accuracy, global_model.state_dict(), local_latency, latency_sum)
            elif args.latencyOpt == 'GA':
                epochs, batch_size, fraction = GA_latency.run(args, epochs, batch_size, fraction, former_accuracy,
                    accuracy, global_model.state_dict(), local_latency, latency_sum)
            iteration += 1
            lr = scheduler.step(iteration)

    # Test inference after completion of training
    if args.model == 'cnn' or args.model == 'resnet18' or args.model == 'audio':
        test_acc = inference(args.gpuid, global_model, test_dataset, args.criterion)
    elif args.model == 'lstm':
        test_acc = lstm_inference(args.gpuid, global_model, test_dataset)
    print(f'\nResults after {iteration} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # save the model
    if args.energyOpt is not None:
        torch.save(global_model.state_dict(),
                   '../save/{}_{}_EnergyOpt{}_model.pt'.format(args.dataset, args.model, args.energyOpt))
    elif args.latencyOpt is not None:
        torch.save(global_model.state_dict(),
                   '../save/{}_{}_LatencyOpt{}_model.pt'.format(args.dataset, args.model, args.latencyOpt))
    end_time = time.time()
    duration = end_time - start_time - (count * 3)
    duration = duration / 60
    print('Total Run Time: {0:0.2f} mins'.format(duration))
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, 'a+') as f:
        f.write(f'\nResults after {iteration} global rounds of training:' + '\n')
        f.write("|---- Avg Train Accuracy: {:.2f}%".format(100 * accuracy) + '\n')
        f.write("|---- Test Accuracy: {:.2f}%".format(100 * test_acc) + '\n')
        f.write('Total Run Time: {0:0.2f} mins'.format(duration) + '\n')
        f.write(f"Current date: {date}\n\n\n")
