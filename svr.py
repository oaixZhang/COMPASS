import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.decomposition import PCA


def cal_pearson(x, y):
    r_row, p_value = pearsonr(x, y)
    return r_row


def svr(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    svr = SVR()
    params = [{'kernel': ['poly'], 'degree': [1], 'gamma': [1], 'C': [0.1, 1, 10, 100],
               'coef0': [0, 0.1, 1, 10, 100]}, ]
    # params = {'kernel': ['rbf'], 'gamma': [100, 0.1, 1, 10], 'C': [0.1, 0.01, 1, 10, 100, 1000]}
    cv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


def svr_with_pca(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = data.values
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    X_basic = X[:, :-2150].copy()
    X_img = X[:, -2150:].copy()
    X_img = PCA(4).fit_transform(X_img)
    X = np.hstack((X_basic, X_img))
    X = MinMaxScaler().fit_transform(X)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    svr = SVR()
    params = [{'kernel': ['poly'], 'degree': [1], 'gamma': [1], 'C': [0.1, 1, 10, 100],
               'coef0': [0, 0.1, 1, 10, 100]}, ]
    # params = {'kernel': ['rbf'], 'gamma': [100, 0.1, 1, 10], 'C': [0.1, 0.01, 1, 10, 100, 1000]}
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


def svr_with_clinical_data():
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


# 422 samples
def svr_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # # original data
    CN = data[data.DX_bl == 1].copy()
    # CN_o = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(CN_o, 'CN with basic data')
    MCI = data[data.DX_bl == 2].copy()
    # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(MCI_o, 'MCI with basic data')
    AD = data[data.DX_bl == 3].copy()
    # AD_o = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(AD_o, 'AD with basic data')
    # # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # svr(all_o, 'overall with original data')
    # #
    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(CN_ADNI, 'CN with ADNI features')
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(MCI_ADNI, 'MCI with ADNI features')
    # AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(AD_ADNI, 'AD with ADNI features')
    # # overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # # svr(overall_ADNI, 'overall with ADNI features')

    # # with imaging data
    # CN_imaging = df[df.DX_bl == 1].copy()
    # CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr_with_pca(CN_img, 'CN with imaging data')
    # MCI_imaging = df[df.DX_bl == 2].copy()
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr_with_pca(MCI_img, 'MCI with imaging data')
    # AD_imaging = df[df.DX_bl == 3].copy()
    # AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr_with_pca(AD_img, 'AD with imaging data')
    # # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # svr(all_o, 'overall with original data')
    #
    # # with imaging and ADNI features
    # CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr_with_pca(CN_img, 'CN with imaging and ADNI')
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr_with_pca(MCI_img, 'MCI with imaging and ADNI')
    # AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr_with_pca(AD_img, 'AD with imaging and ADNI')
    # # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # svr(all_o, 'overall with original data')

    # basic clinical features + imaging feature
    # data = df.iloc[:, 0:12].copy()
    imaging = df.iloc[:, 12:].copy()  # imaging data
    # original data
    # MCI = data[data.DX_bl == 1].copy()
    CN_img = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # arg_maxs = [1707, 1657, 1553, 1128, 1353, 730, 655, 1378, 1278]
    arg_max_CN = [1707, 1553, 730, 73, 1758]
    # arg_maxs = [986, 886, 836, 2006, 2004, 211, 1861, 1854, 86]
    # arg_maxs = [986, 2006, 1904, 2080]
    # arg_maxs = [2055, 981, 986, 843, 2068, 218, 5, 893, 2061]
    # arg_maxs = [2055, 981, 218, 5, 211, 830, 81, 1005, 86, 880]
    for i in range(len(arg_max_CN)):
        CN_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_CN[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        # print(MCI_o)
        print(imaging.iloc[:, arg_max_CN[i]].name)
    svr(CN_img.copy(), 'CN with imaging data , i={}'.format(i))

    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_MCI = [986, 2006, 1904, 2080]
    for i in range(len(arg_max_MCI)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_MCI[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        # print(MCI_o)
        print(imaging.iloc[:, arg_max_MCI[i]].name)
    svr(MCI_img.copy(), 'MCI with imaging data , i={}'.format(i))

    AD_img = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_AD = [2055, 981, 218, 5]
    for i in range(len(arg_max_AD)):
        AD_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_AD[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        # print(MCI_o)
        print(imaging.iloc[:, arg_max_AD[i]].name)
    svr(AD_img.copy(), 'AD with imaging data , i={}'.format(i))

    '''
    *** MCI with imaging data svr regression ***  i=4  drop similar features
    X.shape:  (135, 13) y.shape:  (135,)
    best Pearson score: 0.6821452555434724
    best parameters:  {'C': 100, 'coef0': 10, 'degree': 1, 'gamma': 1, 'kernel': 'poly'}
    FreeSurfer.convexity..mean.2011     0.6112787016848616
    geodesic.depth..skew.2006           0.6261463486011841
    FreeSurfer.convexity..skew.1008     0.6509310175612549
    mean.curvature..MAD.1034            0.6735983115952879
    FreeSurfer.convexity..skew.2012     0.6821452555434724

    *** MCI with imaging data svr regression ***  i=3  drop similar features
    X.shape:  (196, 12) y.shape:  (196,)
    best Pearson score: 0.6237654659135236
    best parameters:  {'C': 100, 'coef0': 0.1, 'degree': 1, 'gamma': 1, 'kernel': 'poly'}
    FreeSurfer.thickness..25..1015      0.5603951297011807
    FreeSurfer.thickness..25..2009      0.5867884424007108
    FreeSurfer.thickness..median.2007   0.5984695826505688
    Volume.2008                         0.6237654659135236

    *** MCI with imaging data svr regression ***  i=3  drop similar features
    X.shape:  (91, 12) y.shape:  (91,)
    best Pearson score: 0.7024158312734484
    best parameters:  {'C': 100, 'coef0': 0.1, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
    Volume.1008                         0.6890056980824936
    FreeSurfer.thickness..25..1009      0.6911095054771033
    mean.curvature..75..1025            0.7014605688168164
    area.1008                           0.7024158312734484
    '''


# 422 samples
def img_pca():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    # basic clinical features + imaging feature
    data = df.iloc[:, 0:12].copy()
    imaging = df.iloc[:, 12:].copy()  # imaging data
    # original data
    MCI = data[data.DX_bl == 3].copy()
    MCI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    y = MCI.pop('deltaMMSE').values

    # PCA
    # arg_max = [1707, 1657, 1553, 1128, 1353, 730, 655, 1378, 1278, 1978, 625, 73, 1428, 1578, 1807, 123, 1758, 1832,
    #            1328, 1953]
    # arg_max = [986, 886, 836, 2006, 2004, 211, 1861, 1854, 86, 2011, 1904, 2030, 1856, 1914, 36, 1905, 1911, 2080, 1872,
    #            1903]
    arg_max = [2055, 981, 986, 843, 2068, 218, 5, 893, 2061, 831, 993, 1018, 886, 211, 830, 81, 1005, 86, 880, 881]
    X_img = imaging.iloc[MCI.index, arg_max].values
    X_img = PCA(4).fit_transform(X_img)
    X = np.hstack((MCI.values, X_img))
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)

    X = MinMaxScaler().fit_transform(X)
    svr = SVR()
    params = [{'kernel': ['poly'], 'degree': [1], 'gamma': [1], 'C': [0.1, 1, 10, 100],
               'coef0': [0, 0.1, 1, 10, 100]}, ]
    cv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


'''
CN pca =4
X.shape:  (135, 12) y.shape:  (135,)
best Pearson score: 0.6118877875444588
best parameters:  {'C': 10, 'coef0': 100, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
pca = 5  
X.shape:  (135, 13) y.shape:  (135,)
best Pearson score: 0.6022417297152556
best parameters:  {'C': 10, 'coef0': 0.1, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
MCI pca=3
X.shape:  (196, 11) y.shape:  (196,)
best Pearson score: 0.6266372261407923
best parameters:  {'C': 10, 'coef0': 100, 'degree': 1, 'gamma': 1, 'kernel': 'poly'}
pca=4 
X.shape:  (196, 12) y.shape:  (196,)
best Pearson score: 0.6190105601540308
best parameters:  {'C': 10, 'coef0': 0, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
AD pca=3
X.shape:  (91, 11) y.shape:  (91,)
best Pearson score: 0.6947878616442724
best parameters:  {'C': 10, 'coef0': 100, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
pca = 4
X.shape:  (91, 12) y.shape:  (91,)
best Pearson score: 0.6878011877966744
best parameters:  {'C': 10, 'coef0': 1, 'degree': 1, 'gamma': 1, 'kernel': 'poly'} 
'''

if __name__ == "__main__":
    # svr_with_clinical_data()
    svr_with_imaging_data()

    # PCA
    # df = pd.read_csv('./data_genetic/imaging_data.csv')
    # MCI = df[df.DX_bl == 3].copy()
    # MCI = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF','DECLINED'])
    # svr_with_pca(MCI,'MCI with PCA')

    # img_pca()
