# Benchmaking and Optimizing Federated Learning with Hardware-related Metrics

**This is the source code of the paper: "Benchmarking and Optimizing Federated Learning with Hardware-related Metrics", BMVC 2024.**

**FEDHW** is a optimization framework that can optimize FL algorithms in energy-constrained or latency-constrained situations with hardware-related metrics. In this project, we use the **FedAvg** and the **FedOpt** as the baseline algorithm, and implement them in **Python 3.11**.

## Build environments

In this project, we use following python packages:

```
PyTorch
TorchVision
scikit-learn
numpy
librosa
pandas
pynvml
```

We also provide env.yml for anaconda environment recreation. Using the following command:

```
conda env create -f env.yml
```

Besides, we need to use [Intel PCM](https://github.com/intel/pcm.git) to monitor the server CPU energy consumption.

## Project Structure

The project structure is the same for FEDHW_FedAvg and FEDHW_FedOpt. Use FEDHW_FedAvg as an example:

```
FEDHW_FedAvg
│  ├─data
│  ├─save
│  │  ├─logs
│  │  └─simulate_model
│  └─src
│      ├─utils
```

The data folder is used to store datasets. MNIST and CIFAR-10 can be downloaded using PyTorch. ESC-50 can be downloaded at https://github.com/karolpiczak/ESC-50.git. R8 can be downloaded at https://github.com/jiangqy/LSTM-Classification-pytorch.git. The save folder is used to store the regression models for each dataset. The src folder is the source code of this project. 

## Run the program

The program entry is the main.py in the src folder. There is an example command to run the program:

```
python main.py --dataset=mnist --data_classes=10 --client_num=100 --model=cnn --rounds=10 --lr=0.01 --epochs=20 --b_size=10 --fraction=0.1 --criterion=nll --optimizer=sgd --accuracy=0.98
```

This command is used to train the MNIST dataset in FedAvg.

The following command is used to train the MNIST dataset in FedOpt:

```
python main.py --dataset=mnist --data_classes=10 --client_num=100 --model=cnn --rounds=10 --lr=0.01 --serverlr=0.03 --epochs=20 --b_size=10 --fraction=0.1 --criterion=nll --optimizer=sgd --accuracy=0.98
```

It has an extra parameter 'serverlr', which is used to adjust learning rate of the server optimizer in the FedOpt algorithm. The full command used in this experiment is in the **command.txt** file.

After running the program, the result will be in the save/logs folder.
