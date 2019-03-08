from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
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
accuracy on CN test data:  0.7959183673469388
precision on CN test data:  0.0
Pearson score on CN test data:  -0.1113692092779409  
"""


def cn_without_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN.csv')
    y_CN = CN.pop('DECLINED')
    scaler = MinMaxScaler()
    X_CN = scaler.fit_transform(CN.values)
    X_CN_train, X_CN_test, y_CN_train, y_CN_test = train_test_split(X_CN, y_CN.values, test_size=0.25, random_state=1)
    print("test set:", Counter(y_CN_test))
    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_CN_train))
    X_CN_train, y_CN_train = sm.fit_resample(X_CN_train, y_CN_train)
    print("train set after oversampling:", Counter(y_CN_train))

    svc_CN = SVC(C=1)

    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    gridcn = GridSearchCV(svc_CN, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    gridcn.fit(X_CN_train, y_CN_train)
    cn = gridcn.best_estimator_
    print('best Pearson score on train set:', gridcn.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])

    predict_CN = cn.predict(X_CN_test)
    print(y_CN_test)
    print(predict_CN)
    print("accuracy on CN test data: ", cn.score(X_CN_test, y_CN_test))
    print("precision on CN test data: ", precision_score(y_CN_test, predict_CN))
    print("Pearson score on CN test data: ", cal_pearson(y_CN_test, predict_CN), '\n')
    pd.DataFrame(gridcn.cv_results_).to_csv('./grid/grid_CN_o.csv')


#  MCI
"""
best parameters:
gamma 1.5
random_state 1
coef0 1
degree 2
class_weight balanced
kernel poly
accuracy on MCI test data:  0.5802469135802469
precision on MCI test data:  0.46511627906976744
Pearson score on MCI test data:  0.18033915132256592 
"""


def mci_without_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI.csv')
    y_MCI = MCI.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_MCI = scaler.fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, y_MCI_train, y_MCI_test = train_test_split(X_MCI, y_MCI.values, test_size=0.25,
                                                                        random_state=1)
    print("test set:", Counter(y_MCI_test))

    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_MCI_train))
    X_MCI_train, y_MCI_train = sm.fit_resample(X_MCI_train, y_MCI_train)
    print("train set after oversampling:", Counter(y_MCI_train))

    svc_MCI = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    grid = GridSearchCV(svc_MCI, param_grid=params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    grid = grid.fit(X_MCI_train, y_MCI_train)
    mci = grid.best_estimator_
    print('best Pearson score on train set:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])

    predict = mci.predict(X_MCI_test)
    print(y_MCI_test)
    print(predict)
    print("accuracy on MCI test data: ", mci.score(X_MCI_test, y_MCI_test))
    print("precision on MCI test data: ", precision_score(y_MCI_test, predict))
    print("Pearson score on MCI test data: ", cal_pearson(y_MCI_test, predict), '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_MCI_o.csv')


# AD
"""
best parameters:
degree 2
gamma 0.1
coef0 100
kernel poly
class_weight balanced
random_state 1
accuracy on AD test data:  0.56
precision on AD test data:  0.7777777777777778
Pearson score on AD test data:  0.21527777777777782
"""


def ad_without_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD.csv')
    y_AD = AD.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_AD = scaler.fit_transform(AD.values)
    X_AD_train, X_AD_test, y_AD_train, y_AD_test = train_test_split(X_AD, y_AD.values, test_size=0.25, random_state=1)
    print("test set:", Counter(y_AD_test))

    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_AD_train))
    X_AD_train, y_AD_train = sm.fit_resample(X_AD_train, y_AD_train)
    print("train set after oversampling:", Counter(y_AD_train))

    svc_AD = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(10, 5, random_state=0)
    gridad = GridSearchCV(svc_AD, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    gridad.fit(X_AD_train, y_AD_train)
    ad = gridad.best_estimator_
    print('best Pearson score:', gridad.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])

    predict_AD = ad.predict(X_AD_test)
    print(y_AD_test)
    print(predict_AD)
    print("accuracy on AD test data: ", ad.score(X_AD_test, y_AD_test))
    print("precision on AD test data: ", precision_score(y_AD_test, predict_AD))
    print("Pearson score on AD test data: ", cal_pearson(y_AD_test, predict_AD), '\n')
    pd.DataFrame(gridad.cv_results_).T.to_csv('./grid/grid_AD_o.csv')

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
    accuracy on CN test data:  0.7755102040816326
    precision on CN test data:  0.0
    Pearson score on CN test data:  -0.12592155012732867 
    """


def cn_with_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN_extra_data.csv')
    y_CN = CN.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_CN = scaler.fit_transform(CN.values)
    X_CN_train, X_CN_test, y_CN_train, y_CN_test = train_test_split(X_CN, y_CN.values, test_size=0.25, random_state=1)
    print("test set:", Counter(y_CN_test))
    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_CN_train))
    X_CN_train, y_CN_train = sm.fit_resample(X_CN_train, y_CN_train)
    print("train set after oversampling:", Counter(y_CN_train))

    svc_CN = SVC(C=1)

    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    gridcn = GridSearchCV(svc_CN, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    gridcn.fit(X_CN_train, y_CN_train)
    cn = gridcn.best_estimator_
    print('best Pearson score on train set:', gridcn.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])

    predict_CN = cn.predict(X_CN_test)
    print(y_CN_test)
    print(predict_CN)
    print("accuracy on CN test data: ", cn.score(X_CN_test, y_CN_test))
    print("precision on CN test data: ", precision_score(y_CN_test, predict_CN))
    print("Pearson score on CN test data: ", cal_pearson(y_CN_test, predict_CN), '\n')
    pd.DataFrame(gridcn.cv_results_).T.to_csv('./grid/grid_CN.csv')


# MCI
"""
best parameters:
gamma 0.1
random_state 1
coef0 100
degree 2
class_weight {0: 1, 1: 2}
kernel poly
accuracy on MCI test data:  0.7283950617283951
precision on MCI test data:  0.6666666666666666
Pearson score on MCI test data:  0.41309219466158226 
"""


def mci_with_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/cla_MCI_extra_Data.csv')
    y_MCI = MCI.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_MCI = scaler.fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, y_MCI_train, y_MCI_test = train_test_split(X_MCI, y_MCI.values, test_size=0.25,
                                                                        random_state=1)
    print("test set:", Counter(y_MCI_test))
    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_MCI_train))
    X_MCI_train, y_MCI_train = sm.fit_resample(X_MCI_train, y_MCI_train)
    print("train set after oversampling:", Counter(y_MCI_train))

    svc_MCI = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    grid = GridSearchCV(svc_MCI, param_grid=params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    grid = grid.fit(X_MCI_train, y_MCI_train)
    mci = grid.best_estimator_
    print('best Pearson score on train set:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])

    predict = mci.predict(X_MCI_test)
    print(y_MCI_test)
    print(predict)
    print("accuracy on MCI test data: ", mci.score(X_MCI_test, y_MCI_test))
    print("precision on MCI test data: ", precision_score(y_MCI_test, predict))
    print("Pearson score on MCI test data: ", cal_pearson(y_MCI_test, predict), '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_MCI.csv')


# AD
"""
best parameters:
gamma 1
random_state 1
coef0 100
degree 2
class_weight balanced
kernel poly
accuracy on AD test data:  0.76
precision on AD test data:  0.8571428571428571
Pearson score on AD test data:  0.5103545438827407 
"""


def ad_with_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD_extra_data.csv')
    y_AD = AD.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_AD = scaler.fit_transform(AD.values)
    X_AD_train, X_AD_test, y_AD_train, y_AD_test = train_test_split(X_AD, y_AD.values, test_size=0.25, random_state=1)
    print("test set:", Counter(y_AD_test))

    # oversampling
    sm = SMOTE()
    print("train set:", Counter(y_AD_train))
    X_AD_train, y_AD_train = sm.fit_resample(X_AD_train, y_AD_train)
    print("train set after oversampling:", Counter(y_AD_train))

    svc_AD = SVC()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 1.5, 2, 3],
              'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]}
    scoring = make_scorer(cal_pearson)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=0)
    gridad = GridSearchCV(svc_AD, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    gridad.fit(X_AD_train, y_AD_train)
    ad = gridad.best_estimator_
    print('best Pearson score on train set:', gridad.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])

    predict_AD = ad.predict(X_AD_test)
    print(y_AD_test)
    print(predict_AD)
    print("accuracy on AD test data: ", ad.score(X_AD_test, y_AD_test))
    print("precision on AD test data: ", precision_score(y_AD_test, predict_AD))
    print("Pearson score on AD test data: ", cal_pearson(y_AD_test, predict_AD), '\n')
    pd.DataFrame(gridad.cv_results_).to_csv('./grid/grid_AD.csv')


if __name__ == "__main__":
    print('#### without ADNI-MEM ADNI-EF data ####')
    cn_without_extra_data()
    mci_without_extra_data()
    ad_without_extra_data()
    print('#### with ADNI-MEM ADNI-EF data ####')
    cn_with_extra_data()
    mci_with_extra_data()
    ad_with_extra_data()
