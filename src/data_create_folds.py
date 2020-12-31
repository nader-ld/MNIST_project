#! /Users/naderlaaldehghani/anaconda3/envs/ml-env/bin/python

#src/data_create_folds

import pandas as pd
from sklearn import model_selection
import argparse
import config


def create_folds(data_train, n_splits):
    X_train = data_train.drop('class',axis=1)
    y_train = data_train['class']

    data_train['Kfold'] = -1

    Kfold = model_selection.StratifiedKFold(n_splits=args.n_splits)
    for i, (t_, v_) in enumerate(Kfold.split(X=X_train, y=y_train)):
        data_train.loc[v_,'Kfold'] = i
    
    return data_train


if __name__ == "__main":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_splits',type=int)
    args = parser.parse_args()

    data_train = pd.read_csv(config.TRAINING_FILE,header=0)

    data_train_folds = create_folds(data_train=data_train, n_splits=args.n_splits)
    
    data_train_folds.to_csv(config.TRAINING_FOLDS_FILE,index=False)