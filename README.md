# STPGNN
This is an implementation of [Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting]

## Environment
- python 3.7.4
- torch 1.2.0
- numpy 1.17.2

## Dataset
Step 1： Download the processed dataset from [Baidu Yun](https://pan.baidu.com/s/1J5YvQiY6MfVWyRKDZ_1UyQ) (Access Code:luck).

Step 2: Put them into data directories.

## Train command

    # Train with PEMS03
    python train.py --data PEMS03 --topk 70

    # Train with PEMS04
    python train.py --data PEMS04 --topk 70 --dims 64

    # Train with PEMS07 (Reduce the batch_size appropriately if don't have enough memory.)
    python train.py --data PEMS07 --topk 175 

    # Train with PEMS08
    python train.py --data PEMS08 --topk 35

    # Train with ENG
    python train.py --data ENG --topk 35

    # Train with TaxiBJ
    python train.py --data bjtaxi --topk 200
