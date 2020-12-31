#! /Users/naderlaaldehghani/anaconda3/envs/ml-env/bin/python

#src/data_create_folds

import pandas as pd
from sklearn import model_selection
import argparse

if __name__ == "__main":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_splits',type=int)
    args = parser.parse_args()

    data_train = pd.read_csv(r"../input/train.csv",header=0)


    X_train = data_train.drop('class',axis=1)
    y_train = data_train['class']

    data_train['Kfold'] = -1

    Kfold = model_selection.StratifiedKFold(n_splits=args.n_splits)
    for i, (t_, v_) in enumerate(Kfold.split(X=X_train, y=y_train)):
        data_train.loc[v_,'Kfold'] = i

    data_train.to_csv('../input/train_folds.csv',index=False)