import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def cal_pearson(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    # sum x_list、y_list
    sum_xy = np.sum(np.multiply(x, y))
    # x_list square、y_list square
    sum_x2 = sum([pow(i, 2) for i in x])
    sum_y2 = sum([pow(j, 2) for j in y])
    molecular = sum_xy - (float(sum_x) * float(sum_y) / n)
    # calculate
    denominator = np.sqrt((sum_x2 - float(sum_x ** 2) / n) * (sum_y2 - float(sum_y ** 2) / n))
    if denominator != 0:
        return molecular / denominator
    else:
        return 0


def svc_crossvalidate(file, params, n_splits=5):
    print('*** {} ***'.format(file))
    data = pd.read_csv('./data_genetic/{}.csv'.format(file))
    y = data.pop('DECLINED').values
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
    scoring = make_scorer(cal_pearson)
    cv = StratifiedKFold(n_splits=n_splits, random_state=9)
    grid = GridSearchCV(SVC(random_state=0), params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    result = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    joblib.dump(result, './model_params/{}.m'.format(file))
    pd.DataFrame(grid.cv_results_).to_csv('./grid/{}.csv'.format(file), index=0)

    return result


"""
without ADNI-MEM ADNI-EF
CN

best parameters: 
gamma 1.5
random_state 1
coef0 10
degree 2
class_weight {0: 1, 1: 12}
kernel poly
"""


def cn_without_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [1, 10, 100, 1000],
               'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}]}]
    svc_crossvalidate('clf_CN', params, 4)


"""
without ADNI-MEM ADNI-EF
MCI

best parameters:
gamma 1.5
random_state 1
coef0 1
degree 2
class_weight balanced
kernel poly
 
"""


def mci_without_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'C': [0.1, 1, 10, 100, 1000]}]
    svc_crossvalidate('clf_MCI', params, 5)


"""
without ADNI-MEM ADNI-EF
AD


best parameters:
degree 2
gamma 0.1
coef0 100
kernel poly
class_weight balanced
"""


def ad_without_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2, 3], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}], 'C': [0.1, 1, 10, 100, 1000]}]
    svc_crossvalidate('clf_AD', params, 5)


"""
with ADNI-MEM ADNI-EF data
CN
best parameters: 
gamma 2
random_state 1
coef0 0
degree 1
class_weight balanced
kernel rbf
"""


def cn_with_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [0.1, 1, 10, 100, 1000],
               'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}]}]
    svc_crossvalidate('clf_CN_extra_data', params, 4)


"""
with ADNI-MEM ADNI-EF data
MCI

best parameters:
gamma 0.1
random_state 1
coef0 100
degree 2
class_weight {0: 1, 1: 2}
kernel poly
"""


def mci_with_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'C': [0.1, 1, 10, 100, 1000]}]
    svc_crossvalidate('clf_MCI_extra_data', params, 5)


"""
with ADNI-MEM ADNI-EF data
AD

best parameters:
gamma 1
random_state 1
coef0 100
degree 2
class_weight balanced
kernel poly
"""


def ad_with_extra_data():
    params = [{'kernel': ['poly'], 'degree': [1, 2, 3], 'C': [0.1, 1, 10, 100],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}], 'gamma': [0.01, 0.1, 1, 10],
               'coef0': [0, 0.1, 1, 10, 100]},
              {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10],
               'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}], 'C': [0.1, 1, 10, 100, 1000]}]
    svc_crossvalidate('clf_AD_extra_data', params, 5)


if __name__ == "__main__":
    print('#### without ADNI-MEM ADNI-EF data ####')
    cn_without_extra_data()
    mci_without_extra_data()
    ad_without_extra_data()
    print('#### with ADNI-MEM ADNI-EF data ####')
    cn_with_extra_data()
    mci_with_extra_data()
    ad_with_extra_data()
