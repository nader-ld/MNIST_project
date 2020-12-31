# src/model_dispatcher.py

from sklearn import tree
from sklearn import ensemble

models = {
        'dt_gini': tree.DecisionTreeClassifier(criterion='gini'),
        'dt_entropy': tree.DecisionTreeClassifier(criterion='entropy'),
        'rf': ensemble.RandomForestClassifier()
          }
