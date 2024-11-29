import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.inference import VariableElimination
from sklearn.naive_bayes import CategoricalNB
import numpy as np

data = pd.read_csv('2020_bn_nb_data.txt', sep='\t')

grade_map = {'AA': 10, 'AB': 9, 'BB': 8, 'BC': 7, 'CC': 6, 'CD': 5, 'DD': 4, 'F': 0}
data.replace(grade_map, inplace=True)
qualification_map = {'y': 1, 'n': 0}
data['QP'].replace(qualification_map, inplace=True)

data = data.apply(pd.to_numeric, errors='coerce')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

results = {'naive_bayes': [], 'bayesian_network': []}

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    nb_model = CategoricalNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    results['naive_bayes'].append(accuracy_score(y_test, y_pred_nb))

    hc = HillClimbSearch(pd.concat([X_train, y_train], axis=1))
    model = hc.estimate(scoring_method=BicScore(pd.concat([X_train, y_train], axis=1)))
    bn_model = BayesianNetwork(model.edges())
    bn_model.fit(pd.concat([X_train, y_train], axis=1))
    inference = VariableElimination(bn_model)
    y_pred_bn = [inference.map_query(variables=[y.name], evidence=dict(row))[y.name] for _, row in X_test.iterrows()]
    results['bayesian_network'].append(accuracy_score(y_test, y_pred_bn))

print("Naive Bayes Average Accuracy:", np.mean(results['naive_bayes']))
print("Bayesian Network Average Accuracy:", np.mean(results['bayesian_network']))
