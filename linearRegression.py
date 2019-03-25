import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import MinMaxScaler

import dataprocess
from svm_crossvalidate import cal_pearson


def declined(x, y):
    print(x)
    print(y)
    declined = []
    for i in range(0, len(x)):
        declined.append(int((y[i] - x[i]) <= -3))
    return np.array(declined)


def lr_CN():
    print('*** CN regression ***')
    CN = pd.read_csv('./data/reg_CN.csv')
    y_CN = CN.pop('deltaMMSE')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)

    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_CN, y_CN)
    cn = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', cn.get_params())


def lr_MCI():
    print('*** MCI regression ***')
    MCI = pd.read_csv('./data/reg_MCI.csv')
    y_MCI = MCI.pop('deltaMMSE')
    X_MCI = MinMaxScaler().fit_transform(MCI.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', mci.get_params())


def lr_AD():
    print('*** AD regression ***')
    AD = pd.read_csv('./data/reg_AD.csv')
    y_AD = AD.pop('deltaMMSE')
    X_AD = MinMaxScaler().fit_transform(AD.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_AD, y_AD)
    ad = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', ad.get_params())


def lr_CN_extra_data():
    print('*** CN regression ***')
    CN = pd.read_csv('./data/reg_CN_extra_data.csv')
    y_CN = CN.pop('deltaMMSE')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_CN, y_CN)
    cn = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', cn.get_params())


def lr_MCI_extra_data():
    print('*** MCI regression ***')
    MCI = pd.read_csv('./data/reg_MCI_extra_data.csv')
    y_MCI = MCI.pop('deltaMMSE')
    X_MCI = MinMaxScaler().fit_transform(MCI.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', mci.get_params())


def lr_AD_extra_data():
    print('*** AD regression ***')
    AD = pd.read_csv('./data/reg_AD_extra_data.csv')
    y_AD = AD.pop('deltaMMSE')
    X_AD = MinMaxScaler().fit_transform(AD.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(lr, {}, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_AD, y_AD)
    ad = grid.best_estimator_
    print('best score:', grid.best_score_)
    # print('best params:', ad.get_params())


def lr(data, group):
    print('*** %s linear regression ***' % group)
    data.dropna(axis=0, how='any', inplace=True)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = make_scorer(cal_pearson)
    results = cross_validate(lr, X, y, scoring=scoring, cv=cv, return_train_score=False)
    # print('scores:', results['test_score'])
    print('mean score:', results['test_score'].mean(), '\n')


if __name__ == "__main__":
    # print('#### without ADNI-MEM ADNI-EF data ####')
    # lr_CN()
    # lr_MCI()
    # lr_AD()
    # print('#### with ADNI-MEM ADNI-EF data ####')
    # lr_CN_extra_data()
    # lr_MCI_extra_data()
    # lr_AD_extra_data()

    data = pd.read_csv('./data_genetic/data_all_features.csv')
    # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(CN_o, 'CN with original data')
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(MCI_o, 'MCI with original data')
    AD = data[data.DX_bl == 3].copy()
    AD_o = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(AD_o, 'AD with original data')
    all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(all_o, 'overall with original data')

    # with genetic features
    CN_genetic = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(CN_genetic, 'CN with genetic features')

    MCI_genetic = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(MCI_genetic, 'MCI with genetic features')

    AD_genetic = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(AD_genetic, 'AD with genetic features')

    overall_genetic = data.drop(columns=['RID', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    lr(overall_genetic, 'overall with genetic features')

    # with ADNI features
    CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    lr(CN_ADNI, 'CN with ADNI features')
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    lr(MCI_ADNI, 'MCI with ADNI features')
    AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    lr(AD_ADNI, 'AD with ADNI features')
    overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    lr(overall_ADNI, 'overall with ADNI features')
