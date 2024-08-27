# -*- coding: utf-8 -*-
# python version: 3.11
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np
import torch
import os
import copy
import librosa
from utils.cutout import Cutout
import pandas as pd
from sklearn import model_selection
import sys


sys.path.append('../')



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return data, label


class TxtDatasetProcessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)
        fp = open(txt_filepath, 'r')
        self.txt_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len

    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        clip = False
        for words in fp:
            for word in words.split():
                if word.strip() in self.corpus.dictionary.word2idx:
                    if count > self.sen_len - 1:
                        clip = True
                        break
                    txt[count] = self.corpus.dictionary.word2idx[word.strip()]
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])
        return txt, label

    def __len__(self):
        return len(self.txt_filename)


class ESC50Dataset(Dataset):
    def __init__(self, data, label, audio_dir, data_aug=False, _type='train'):
        self.label = label
        self.data_aug = data_aug
        self.data = data
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        x, fs = load_wave_data(self.audio_dir, self.data[idx])

        if self.data_aug:
            r = np.random.rand()
            if r < 0.3:
                x = add_white_noise(x)
            r = np.random.rand()
            if r < 0.3:
                x = shift_sound(x, rate=1+np.random.rand())
            r = np.random.rand()
            if r < 0.3:
                x = stretch_sound(x, rate=0.8+np.random.rand()*0.4)

        melsp = calculate_melsp(x)
        mean = np.mean(melsp)
        std = np.std(melsp)
        melsp -= mean
        melsp /= std

        melsp = np.asarray([melsp, melsp, melsp])
        melsp = torch.tensor(melsp)
        melsp = melsp.float()
        return melsp, label


def get_data(dataset, client_number):
    """
    :param dataset:
    :param client_number:
    :return: return the iid dataset. e.g. cifar-10, mnist
    """
    if dataset == 'cifar10':
        data_dir = '../data/cifar10/'
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=16)
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        user_train_group, user_test_group = split_dataset(train_dataset, test_dataset, client_number)
        return train_dataset, test_dataset, user_train_group, user_test_group
    elif dataset == 'mnist':
        data_dir = '../data/mnist/'
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        user_train_group, user_test_group = split_dataset(train_dataset, test_dataset, client_number)
        return train_dataset, test_dataset, user_train_group, user_test_group
    elif dataset == 'r8':
        data_dir = '../data/r8/'
        train_file = os.path.join(data_dir, 'train_txt.txt')
        test_file = os.path.join(data_dir, 'test_txt.txt')
        fp_train = open(train_file, 'r')
        train_filenames = [os.path.join('train_txt', line.strip()) for line in fp_train]
        filenames = copy.deepcopy(train_filenames)
        fp_train.close()
        fp_test = open(test_file, 'r')
        test_filenames = [os.path.join('test_txt', line.strip()) for line in fp_test]
        fp_test.close()
        filenames.extend(test_filenames)
        corpus = Corpus(data_dir, filenames)
        train_dataset = TxtDatasetProcessing(data_dir, 'train_txt', 'train_txt.txt', 'train_label.txt', 32, corpus)
        test_dataset = TxtDatasetProcessing(data_dir, 'test_txt', 'test_txt.txt', 'test_label.txt', 32, corpus)
        user_train_group, user_test_group = split_dataset(train_dataset, test_dataset, client_number)
        return train_dataset, test_dataset, user_train_group, user_test_group
    elif dataset == 'esc50':
        esc_dir = "../data/ESC-50-master/"
        meta_file = os.path.join(esc_dir, 'meta/esc50.csv')
        audio_dir = os.path.join(esc_dir, 'audio/')
        meta_data = pd.read_csv(meta_file)
        x = list(meta_data.loc[:, 'filename'])
        y = list(meta_data.loc[:, 'target'])
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.20, random_state=17)
        train_dataset = ESC50Dataset(x_train, y_train, audio_dir, data_aug=True)
        test_dataset = ESC50Dataset(x_test, y_test, audio_dir, data_aug=False)
        user_train_group, user_test_group = split_dataset(train_dataset, test_dataset, client_number)
        return train_dataset, test_dataset, user_train_group, user_test_group


def split_dataset(train_dataset, test_dataset, client_number):
    block_length = int(len(train_dataset) / client_number)
    train_users, all_idxs = {}, [i for i in range(len(train_dataset))]
    for i in range(client_number):
        train_users[i] = set(np.random.choice(all_idxs, block_length, replace=False))
        all_idxs = list(set(all_idxs) - train_users[i])

    block_length = int(len(test_dataset) / client_number)
    test_users, all_idxs = {}, [i for i in range(len(test_dataset))]
    for i in range(client_number):
        test_users[i] = set(np.random.choice(all_idxs, block_length, replace=False))
        all_idxs = list(set(all_idxs) - test_users[i])

    return train_users, test_users


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

        return ids


def load_wave_data(audio_dir, file_name):
    file_path = os.path.join(audio_dir, file_name)
    x, fs = librosa.load(file_path, sr=44100)
    return x, fs


def calculate_melsp(x, n_fft=1024, hop_length=128):
    stft = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))**2
    log_stft = librosa.power_to_db(stft)
    melsp = librosa.feature.melspectrogram(S=log_stft, n_mels=128)
    return melsp


def add_white_noise(x, rate=0.002):
    return x + rate*np.random.randn(len(x))


def shift_sound(x, rate):
    return np.roll(x, int(len(x)//rate))


def stretch_sound(x, rate):
    input_length = len(x)
    x = librosa.effects.time_stretch(y=x, rate=rate)
    if len(x) > input_length:
        return x[:input_length]
    else:
        return np.pad(x, (0, max(0, input_length-len(x))), "constant")
