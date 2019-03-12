import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from svm import cal_pearson


def mlp():
    data = pd.read_csv('./data/clf_AD.csv')
    y = data.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(data.values)
    mlpc = MLPClassifier(hidden_layer_sizes=(100,), activation='identity', solver='lbfgs', learning_rate='adaptive',batch_size=10, )
    cv = StratifiedKFold(n_splits=10, random_state=0)
    scoring = make_scorer(cal_pearson)
    scores = cross_val_score(mlpc, X, y, scoring=scoring, cv=cv)
    print(scores.mean())
    print(scores)


if __name__ == '__main__':
    mlp()
