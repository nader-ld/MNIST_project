# src/config.py

from sklearn import tree

TRAINING_FILE = "../input/train.csv"

TEST_FILE = "../input/test.csv"

TRAINING_FOLDS_FILE = "../input/train_folds.csv"

MODELS_FOLDER = "../models/"

MODEL = {'model_name':'dt_default',
         'model':tree.DecisionTreeClassifier()}



