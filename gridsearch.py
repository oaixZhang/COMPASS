import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold

import svm


def gridsearch():
    data = pd.read_csv('./data_genetic/clf_CN.csv')
    y = data.pop('DECLINED')
    X = data.values

    pipe = Pipeline(
        [('norm', MinMaxScaler(feature_range=(0, 1))),
         ('clf', SVC(kernel='poly', random_state=9))])

    params = [{'clf__C': [1, 10, 100, 1000],
               'clf__gamma': [0.1, 1, 10],
               'clf__degree': [1, 2],
               'clf__coef0': [0, 0.1, 1, 10, 100],
               'clf__class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}]}]

    scoring = make_scorer(svm.cal_pearson)
    # scoring = make_scorer(roc_auc_score)
    cv = StratifiedKFold(10, random_state=9)
    grid = GridSearchCV(pipe, params, scoring=scoring, cv=cv, iid=False, return_train_score=False)

    grid.fit(X, y)

    print('best score: ', grid.best_score_)
    print('best params:', grid.best_params_)

    pd.DataFrame(grid.cv_results_).to_csv('./grid/test1.csv', index=0)


if __name__ == "__main__":
    gridsearch()
