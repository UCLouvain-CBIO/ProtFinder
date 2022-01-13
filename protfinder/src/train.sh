#!/bin/bash

python ./train.py --trainpath ../data/train_all_0.6.csv --testpath ../data/test_all_0.6.csv --gpu --batchsize 64 --nclass 2 -lr 4e-4 --epochs 8 --save --modelpath ../models/final_lstm_model.pt 
