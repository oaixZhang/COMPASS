import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


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


def svr(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    # print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    svr = SVR()
    params = [{'kernel': ['poly'], 'degree': [2], 'gamma': [0.1, 1, 10], 'C': [0.1, 1, 10, 100],
               'coef0': [0, 0.1, 1, 10, 100, 0.01]}, ]
    # {'kernel': ['rbf'], 'gamma': [100, 0.1, 1, 10], 'C': [0.1, 0.01, 1, 10, 100, 1000]}]
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


if __name__ == "__main__":
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    data.pop('APOE3')
    # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    svr(CN_o, 'CN with original data')
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    svr(MCI_o, 'MCI with original data')
    AD = data[data.DX_bl == 3].copy()
    AD_o = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    svr(AD_o, 'AD with original data')
    # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(all_o, 'overall with original data')

    # with ADNI features
    CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    svr(CN_ADNI, 'CN with ADNI features')
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    svr(MCI_ADNI, 'MCI with ADNI features')
    AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    svr(AD_ADNI, 'AD with ADNI features')
    # overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # svr(overall_ADNI, 'overall with ADNI features')
