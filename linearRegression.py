import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from svr import cal_pearson


def lr(data, group):
    print('*** %s linear regression ***' % group)
    # data.dropna(axis=0, how='any', inplace=True)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    lr = LinearRegression()
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = make_scorer(cal_pearson)
    results = cross_validate(lr, X, y, scoring=scoring, cv=cv, return_train_score=False)
    mscore = results['test_score'].mean()
    print('mean score:', mscore, '\n')
    return mscore


def lr_with_old_data():
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    data.pop('APOE3')
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


# 422 samples
def lr_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    # data = df.iloc[:, 0:12].copy()
    # # original data
    # CN = data[data.DX_bl == 1].copy()
    # CN_o = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(CN_o, 'CN with basic data')
    # MCI = data[data.DX_bl == 2].copy()
    # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(MCI_o, 'MCI with basic data')
    # AD = data[data.DX_bl == 3].copy()
    # AD_o = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(AD_o, 'AD with basic data')
    # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(all_o, 'overall with original data')
    #
    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(CN_ADNI, 'CN with ADNI features')
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(MCI_ADNI, 'MCI with ADNI features')
    # AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(AD_ADNI, 'AD with ADNI features')
    # # overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # # svr(overall_ADNI, 'overall with ADNI features')

    # # with imaging data
    # CN_imaging = df[df.DX_bl == 1].copy()
    # CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(CN_img, 'CN with imaging data')
    # MCI_imaging = df[df.DX_bl == 2].copy()
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(MCI_img, 'MCI with imaging data')
    # AD_imaging = df[df.DX_bl == 3].copy()
    # AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # lr(AD_img, 'AD with imaging data')
    # # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # svr(all_o, 'overall with original data')
    #
    # # with imaging and ADNI features
    # CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(CN_img, 'CN with imaging and ADNI')
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(MCI_img, 'MCI with imaging and ADNI')
    # AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # lr(AD_img, 'AD with imaging and ADNI')
    # # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # svr(all_o, 'overall with original data')

    # basic clinical features + imaging feature
    data = df.iloc[:, 0:12].copy()
    imaging = df.iloc[:, 12:].copy()  # imaging data
    # original data
    MCI = data[data.DX_bl == 1].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # arg_maxs = [986, 886, 836, 2006, 2004, 211, 1861, 1854, 86]
    arg_maxs = [1707, 1657, 1553, 1128, 1353, 730, 655, 1378, 1278]
    # arg_maxs = [2055, 981, 986, 843, 2068, 218, 5, 893, 2061]
    for i in range(9):
        MCI_o['img_feature{}'.format(i)] = imaging.iloc[:, arg_maxs[i]]
        lr(MCI_o.copy(), 'MCI with imaging data')


'''
E:\Miniconda3\python.exe E:/PycharmProjects/COMPASS/linearRegression.py
*** MCI with imaging data linear regression ***
mean score: 0.4094733945324768 

*** MCI with imaging data linear regression ***
mean score: 0.4034395455069453 

*** MCI with imaging data linear regression ***
mean score: 0.39786758411823264 

*** MCI with imaging data linear regression ***
mean score: 0.42798320542985147 

*** MCI with imaging data linear regression ***
mean score: 0.4442587149416045 

*** MCI with imaging data linear regression ***
mean score: 0.4513993032220508 



with ADNI
E:\Miniconda3\python.exe E:/PycharmProjects/COMPASS/linearRegression.py
*** MCI with imaging data linear regression ***
mean score: 0.5760988311373493 

*** MCI with imaging data linear regression ***
mean score: 0.5763838036481591 

*** MCI with imaging data linear regression ***
mean score: 0.571192419496677 

*** MCI with imaging data linear regression ***
mean score: 0.5850854903207536 

*** MCI with imaging data linear regression ***
mean score: 0.5945797965211572 

*** MCI with imaging data linear regression ***
mean score: 0.590723002637133

'''


def select_features():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    # basic clinical features + one imaging feature
    data = df.iloc[:, 0:12].copy()  # original data
    imaging = df.iloc[:, 12:].copy()  # imaging data
    # pearsons = np.zeros(imaging.shape[1])
    # for i in range(imaging.shape[1]):
    #     data['imaging'] = imaging.iloc[:, i]
    #     # CN = data[data.DX_bl == 1]
    #     # CN_o = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    #     # MCI = data[data.DX_bl == 2]
    #     # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    #     AD = data[data.DX_bl == 3]
    #     AD_o = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    #     # pearsonscore = lr(CN_o, 'CN with {}th imaging feature'.format(i))
    #     # pearsonscore = lr(MCI_o, 'MCI with {}th imaging feature'.format(i))
    #     pearsonscore = lr(AD_o, 'AD with {}th imaging feature'.format(i))
    #     pearsons[i] = pearsonscore
    # np.save('./logs/ad_imaging.npy', pearsons)
    pcn = np.load('./logs/cn_imaging.npy')
    pmci = np.load('./logs/mci_imaging.npy')
    pad = np.load('./logs/ad_imaging.npy')

    arg_maxs = np.argsort(pcn)[:-21:-1]
    print(arg_maxs)  # [1707 1657 1553 1128 1353  730  655 1378 1278 1978  625   73 1428 1578 1807  123 1758 1832 1328 1953]
    arg_maxs = np.argsort(pmci)[:-21:-1]
    print(arg_maxs)  # [ 986  886  836 2006 2004  211 1861 1854   86 2011 1904 2030 1856 1914 36 1905 1911 2080 1872 1903]
    arg_maxs = np.argsort(pad)[:-21:-1]
    print(arg_maxs)  # [2055  981  986  843 2068  218    5  893 2061 831  993 1018  886  211  830   81 1005   86  880 881]


def select_features4ROC():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    # basic clinical features + one imaging feature
    data = df.iloc[:, 0:12].copy()  # original data
    imaging = df.iloc[:, 12:].copy()  # imaging data
    aucs = np.zeros(imaging.shape[1])
    for i in range(imaging.shape[1]):
        data['imaging'] = imaging.iloc[:, i]
        MCI = data[data.DX_bl == 2]
        MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
        y = MCI_o.pop('DECLINED').values
        X = MinMaxScaler().fit_transform(MCI_o.values)
        lr = LogisticRegression(solver='lbfgs')
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=9)
        scoring = make_scorer(roc_auc_score, needs_proba=True)
        results = cross_validate(lr, X, y, scoring=scoring, cv=cv, return_train_score=False)
        mscore = results['test_score'].mean()
        print('mean score:', mscore, '\n')
        aucs[i] = mscore
    np.save('./logs/mci_imaging_auroc.npy', aucs)
    aucmci = np.load('./logs/mci_imaging_auroc.npy')

    arg_maxs = np.argsort(aucmci)[:-10:-1]
    print(arg_maxs)  # [ 886  986  836 1903 1011 1904 2004 1854   86]


if __name__ == "__main__":
    # select_features4ROC()
    # lr_with_imaging_data()
    select_features()
