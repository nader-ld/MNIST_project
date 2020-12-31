#! /Users/naderlaaldehghani/anaconda3/envs/ml-env/bin/python

# src/train.py

import joblib
import pandas as pd
import argparse
from sklearn import metrics
from sklearn import tree

import config
import model_dispatcher

def run(data, fold, model):
    
    data_train = data.loc[data['Kfold']!=fold,:].reset_index(drop=True)
    data_val = data.loc[data['Kfold']==fold,:].reset_index(drop=True)
    
    X_train = data_train.drop(['class','Kfold'],axis=1)
    y_train = data_train['class']

    X_val = data_val.drop(['class','Kfold'],axis=1)
    y_val = data_val['class']

    clf = model
    clf.fit(X_train,y_train)
    
    preds = clf.predict(X_val)

    accuracy = metrics.accuracy_score(y_val, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")
    return clf
    

if __name__ == "__main__":

    data_train_folds = pd.read_csv(config.TRAINING_FOLDS_FILE,header=0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-fold", type=int)
    parser.add_argument("-model", type=str)
    args = parser.parse_args()

    if args.model != None:
        model = model_dispatcher.models[args.model]
        model_name = args.model
    else:
        model = config.MODEL['model']
        model_name = config.MODEL['model_name']
    
    clf = run(data=data_train_folds, fold=args.fold, model=model)
    joblib.dump(clf,f"{config.MODELS_FOLDER}df_{model_name}_{args.fold}.bin")