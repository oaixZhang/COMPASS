import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from svr import cal_pearson


def lr(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    lr = LinearRegression()
    # lr = SVR(kernel='poly',degree=3,gamma='auto',coef0=100,C=1)
    cv = RepeatedKFold(n_splits=10, n_repeats=100)
    scoring = make_scorer(cal_pearson)
    results = cross_validate(lr, X, y, scoring=scoring, cv=cv, return_train_score=False)
    mscore = results['test_score'].mean()
    print('mean score:', mscore, '\n')
    return mscore


# 422 samples
def lr_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(CN_o, 'CN with basic data')
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(MCI_o, 'MCI with basic data')
    AD = data[data.DX_bl == 3].copy()
    AD_o = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(AD_o, 'AD with basic data')

    # with ADNI features
    CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    lr(CN_ADNI, 'CN with ADNI features')
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    lr(MCI_ADNI, 'MCI with ADNI features')
    AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    lr(AD_ADNI, 'AD with ADNI features')

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    CN_img = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_CN = [1707, 1553, 730, 73, 1758]
    for i in range(len(arg_max_CN)):
        CN_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_CN[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        # print(MCI_o)
        print(imaging.iloc[:, arg_max_CN[i]].name)
    lr(CN_img.copy(), 'CN with imaging data , i={}'.format(i))

    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_MCI = [986, 2006, 1904, 2080]
    for i in range(len(arg_max_MCI)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_MCI[i]]
        print(imaging.iloc[:, arg_max_MCI[i]].name)
    lr(MCI_img.copy(), 'MCI with imaging data , i={}'.format(i))

    AD_img = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_AD = [2055, 981, 218, 5]
    for i in range(len(arg_max_AD)):
        AD_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_AD[i]]
        print(imaging.iloc[:, arg_max_AD[i]].name)
    lr(AD_img.copy(), 'AD with imaging data , i={}'.format(i))


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
    print(arg_maxs)
    arg_maxs = np.argsort(pmci)[:-21:-1]
    print(arg_maxs)
    arg_maxs = np.argsort(pad)[:-21:-1]
    print(arg_maxs)


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
    print(arg_maxs)


if __name__ == "__main__":
    # select_features4ROC()
    lr_with_imaging_data()
    # select_features()
