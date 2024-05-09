# STPGNN
This is an implementation of [Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting]

The code for this paper is currently being organized and is expected to be uploaded by late May.

## Environment
- python 3.7.4
- torch 1.2.0
- numpy 1.17.2
## Dataset
Step 1： Download the processed dataset from [Baidu Yun](https://pan.baidu.com/s/1J5YvQiY6MfVWyRKDZ_1UyQ) (Access Code:luck).

If needed, the origin dataset of PEMSD4 and PEMSD8 are available from [ASTGCN](https://github.com/Davidham3/ASTGCN).

Step 2: Put them into data directories.
## Train command
    # Train with PEMSD4
    python train.py --data=PEMSD4
    
    # Train with PEMSD8
    python train.py --data=PEMSD8
    
    # Train with England
    python train.py --data=England
