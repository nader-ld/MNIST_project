#! /Users/naderlaaldehghani/anaconda3/envs/ml-env/bin/python 

# src/data_fetch.py

from sklearn import datasets
from random import shuffle
import pandas as pd
import argparse

if __name__ == "__main__":
    data = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

    pixel_values, targets = data

    targets = targets.astype(int)

    parser = argparse.ArgumentParser()
    parser.add_argument("-train_size", type=float)
    args = parser.parse_args()

    idx = list(range(pixel_values.shape[0]))
    shuffle(idx)
    train_size = int(len(idx)*args.train_size)

    train = pd.concat([pixel_values.iloc[idx[:train_size],:],targets[idx[:train_size]]],axis=1)
    test = pd.concat([pixel_values.iloc[idx[train_size:],:],targets[idx[train_size:]]],axis=1)

    train.to_csv(r'../input/train.csv',index=False)
    test.to_csv(r'../input/test.csv',index=False)

