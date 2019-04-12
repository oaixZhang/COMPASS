import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import MinMaxScaler

from svr import cal_pearson


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
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    # data.pop('APOE3')
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

    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # lr(CN_ADNI, 'CN with ADNI features')
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # lr(MCI_ADNI, 'MCI with ADNI features')
    # AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # lr(AD_ADNI, 'AD with ADNI features')
    # overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # lr(overall_ADNI, 'overall with ADNI features')
