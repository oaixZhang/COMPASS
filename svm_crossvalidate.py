import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.externals import joblib


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


# without ADNI-MEM ADNI-EF
# CN
"""
best parameters: 
gamma 1.5
random_state 1
coef0 10
degree 2
class_weight {0: 1, 1: 12}
kernel poly
  
"""


def cn_without_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN.csv')
    y_CN = CN.pop('DECLINED').values
    X_CN = MinMaxScaler().fit_transform(CN.values)

    svc_CN = SVC(C=1)

    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 12}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedKFold(n_splits=4, n_repeats=5)
    gridcn = GridSearchCV(svc_CN, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    gridcn.fit(X_CN, y_CN)
    cn = gridcn.best_estimator_
    print('best Pearson score:', gridcn.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])
    joblib.dump(cn, './model_params/clf_CN.m')
    pd.DataFrame(gridcn.cv_results_).to_csv('./grid/clf_CN.csv', index=0)


#  MCI
"""
best parameters:
gamma 1.5
random_state 1
coef0 1
degree 2
class_weight balanced
kernel poly
 
"""


def mci_without_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI.csv')
    y_MCI = MCI.pop('DECLINED').values
    X_MCI = MinMaxScaler(feature_range=(0, 1)).fit_transform(MCI.values)

    svc_MCI = SVC(C=1)
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedKFold(n_splits=10, n_repeats=5)
    grid = GridSearchCV(svc_MCI, param_grid=params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid = grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    joblib.dump(mci, './model_params/clf_MCI.m')
    pd.DataFrame(grid.cv_results_).to_csv('./grid/clf_MCI.csv', index=0)


# AD
"""
best parameters:
degree 2
gamma 0.1
coef0 100
kernel poly
class_weight balanced
"""


def ad_without_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD.csv')
    y_AD = AD.pop('DECLINED').values
    X_AD = MinMaxScaler(feature_range=(0, 1)).fit_transform(AD.values)
    svc_AD = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 2, 1: 1}, {0: 3, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedKFold(n_splits=10, n_repeats=5)
    gridad = GridSearchCV(svc_AD, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    gridad.fit(X_AD, y_AD)
    ad = gridad.best_estimator_
    print('best Pearson score:', gridad.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])
    joblib.dump(ad, './model_params/clf_AD.m')
    pd.DataFrame(gridad.cv_results_).to_csv('./grid/clf_AD.csv', index=0)


# with ADNI-MEM ADNI-EF data
# CN
"""
best parameters: 
gamma 2
random_state 1
coef0 0
degree 1
class_weight balanced
kernel rbf
"""


def cn_with_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN_extra_data.csv')
    y_CN = CN.pop('DECLINED').values
    X_CN = MinMaxScaler(feature_range=(0, 1)).fit_transform(CN.values)

    svc_CN = SVC(C=1)

    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 6}, {0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 12}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedKFold(n_splits=4, n_repeats=5)
    gridcn = GridSearchCV(svc_CN, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    gridcn.fit(X_CN, y_CN)
    cn = gridcn.best_estimator_
    print('best Pearson score:', gridcn.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])
    joblib.dump(cn, './model_params/clf_CN_extra_data.m')
    pd.DataFrame(gridcn.cv_results_).to_csv('./grid/clf_CN_extra_data.csv', index=0)


# MCI
"""
best parameters:
gamma 0.1
random_state 1
coef0 100
degree 2
class_weight {0: 1, 1: 2}
kernel poly
"""


def mci_with_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI_extra_Data.csv')
    y_MCI = MCI.pop('DECLINED').values
    X_MCI = MinMaxScaler(feature_range=(0, 1)).fit_transform(MCI.values)
    svc_MCI = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    # 10 fold grid search
    cv = RepeatedKFold(n_splits=10, n_repeats=5)
    grid = GridSearchCV(svc_MCI, param_grid=params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid = grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    joblib.dump(mci, './model_params/clf_MCI_extra_Data.m')
    pd.DataFrame(grid.cv_results_).to_csv('./grid/clf_MCI_extra_Data.csv', index=0)


# AD
"""
best parameters:
gamma 1
random_state 1
coef0 100
degree 2
class_weight balanced
kernel poly
"""


def ad_with_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD_extra_data.csv')
    y_AD = AD.pop('DECLINED').values
    X_AD = MinMaxScaler(feature_range=(0, 1)).fit_transform(AD.values)
    svc_AD = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 2, 1: 1}, {0: 3, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedKFold(n_splits=10, n_repeats=5)
    gridad = GridSearchCV(svc_AD, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    gridad.fit(X_AD, y_AD)
    ad = gridad.best_estimator_
    print('best Pearson score:', gridad.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])
    joblib.dump(ad, './model_params/clf_AD_extra_data.m')
    pd.DataFrame(gridad.cv_results_).to_csv('./grid/clf_AD_extra_data.csv', index=0)


if __name__ == "__main__":
    print('#### without ADNI-MEM ADNI-EF data ####')
    cn_without_extra_data()
    mci_without_extra_data()
    ad_without_extra_data()
    print('#### with ADNI-MEM ADNI-EF data ####')
    cn_with_extra_data()
    mci_with_extra_data()
    ad_with_extra_data()
