import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from svr import cal_pearson
from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold, KFold, RepeatedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, ExtraTreesClassifier, RandomForestClassifier
from gcforest import GCForestRegressor, CascadeForestRegressor, GCForest, CascadeForest
from sklearn.decomposition import PCA


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 100, "max_depth": 10,
         "objective": "binary:logistic", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 100, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression", 'solver': 'liblinear'})
    config["cascade"] = ca_config
    return config


def regression(group):
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    data.pop('APOE3')
    print('***************** group=', group, '*********************')
    X = data[data.DX_bl == group].copy()
    X = X.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    y = X.pop('deltaMMSE').values
    X = X.values
    cv = RepeatedKFold(10, 5)
    scores = []
    print('without ADNI FEATURE')
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        regressor = GCForestRegressor(estimators_config={
            'mgs': [{'estimator_class': ExtraTreesRegressor,
                     'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }
                     }, {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }}],
            'cascade': [{'estimator_class': ExtraTreesRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': ExtraTreesRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }}]})
        regressor.fit(X_train, y_train)
        # X_train = cf.transform(X_train)
        # svc = SVC(kernel='poly', degree=1, C=1, coef0=100, class_weight={0: 1, 1: 2}, gamma=1, probability=True)
        # svc.fit(X_train, y_train)
        # X_test = cf.transform(X_test)
        pred = regressor.predict(X_test)
        pearson = cal_pearson(y_test, pred)
        print(pearson)
        scores.append(pearson)
    scores = np.asarray(scores)
    print(scores)
    print("************ pearson: ", scores.mean(), '\n')

    # with ADNI feature
    print('with ADNI FEATURE')
    X = data[data.DX_bl == group].copy()
    X = X.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    y = X.pop('deltaMMSE').values
    # X = MinMaxScaler().fit_transform(X.values)
    X = X.values
    scores = []
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        regressor = GCForestRegressor(estimators_config={
            'mgs': [{'estimator_class': ExtraTreesRegressor,
                     'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }
                     }, {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }}],
            'cascade': [{'estimator_class': ExtraTreesRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': ExtraTreesRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestRegressor,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }}]})
        regressor.fit(X_train, y_train)
        # X_train = cf.transform(X_train)
        # svc = SVC(kernel='poly', degree=1, C=1, coef0=100, class_weight={0: 1, 1: 2}, gamma=1, probability=True)
        # svc.fit(X_train, y_train)
        # X_test = cf.transform(X_test)
        pred = regressor.predict(X_test)
        pearson = cal_pearson(y_test, pred)
        print(pearson)
        scores.append(pearson)
    scores = np.asarray(scores)
    print(scores)
    print("************ pearson: ", scores.mean(), '\n')



def dfregressor(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = data.values
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    cv = RepeatedKFold(5, 1)
    scores = []
    print('without ADNI FEATURE')
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        regressor = CascadeForestRegressor(
            estimators_config=[{'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }}])
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_test)
        pearson = cal_pearson(y_test, pred)
        print(pearson)
        scores.append(pearson)
    scores = np.asarray(scores)
    print(scores)
    print("************ pearson: ", scores.mean(), '\n')


def reg_with_pca(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    X_basic = X[:, :-2150].copy()
    X_img = X[:, -2150:].copy()
    X_img = PCA(0.9).fit_transform(X_img)
    X = np.hstack((X_basic, X_img))
    X = MinMaxScaler().fit_transform(X)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    cv = RepeatedKFold(5, 1)
    scores = []
    print('without ADNI FEATURE')
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        regressor = CascadeForestRegressor(
            estimators_config=[{'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }}])
        regressor.fit(X_train, y_train)
        pred = regressor.predict(X_test)
        pearson = cal_pearson(y_test, pred)
        print(pearson)
        scores.append(pearson)
    scores = np.asarray(scores)
    print(scores)
    print("************ pearson: ", scores.mean(), '\n')


# 422 samples
def reg_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')

    # data = df.iloc[:, 0:12].copy()
    # # original data
    # CN = data[data.DX_bl == 1].copy()
    # CN_o = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # dfregressor(CN_o, 'CN with basic data')
    # MCI = data[data.DX_bl == 2].copy()
    # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # dfregressor(MCI_o, 'MCI with basic data')
    # AD = data[data.DX_bl == 3].copy()
    # AD_o = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # # dfregressor(AD_o, 'AD with basic data')

    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(CN_ADNI, 'CN with ADNI features')
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(MCI_ADNI, 'MCI with ADNI features')
    # AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(AD_ADNI, 'AD with ADNI features')

    # with imaging data
    CN_imaging = df[df.DX_bl == 1].copy()
    CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    reg_with_pca(CN_img, 'CN with imaging data')
    MCI_imaging = df[df.DX_bl == 2].copy()
    MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    reg_with_pca(MCI_img, 'MCI with imaging data')
    AD_imaging = df[df.DX_bl == 3].copy()
    AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    reg_with_pca(AD_img, 'AD with imaging data')

    # # with imaging and ADNI features
    # CN_img = CN_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(CN_img, 'CN with imaging and ADNI')
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(MCI_img, 'MCI with imaging and ADNI')
    # AD_img = AD_imaging.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # dfregressor(AD_img, 'AD with imaging and ADNI')


def roc_MCI():
    files = ['clf_MCI', 'clf_MCI_extra_data']
    titles = ['MCI_Without_ADNI_feature', 'MCI_With_ADNI_feature']
    classifiers = [
        # SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=100, probability=True),
        # SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=1, probability=True)
    ]
    index = 0
    mean_tprs = []
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
        plt.subplot(1, 2, index + 1)
        # clf = classifiers[index]
        for train, test in cv.split(X, y):
            # clf = GCForest(config)
            # clf.fit_transform(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

            clf = LogisticRegression(solver='liblinear')
            clf.fit(X[train], y[train])
            probas_ = clf.predict_proba(X[test])

            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.3f)' % mean_auc, lw=2, alpha=.8)
        mean_tprs.append(mean_tpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        plt.title(titles[index], fontsize=6)
        plt.legend(loc='lower right', fontsize=6)
        index += 1
    plt.show()


if __name__ == '__main__':
    # roc_whole()
    # roc_df()
    # test()
    # roc_MCI()
    reg_with_imaging_data()
