import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import roc_curve, auc, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split, \
    RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from gcforest import GCForest, CascadeForest


def roccurve():
    files = ['clf_MCI', 'clf_MCI_extra_data']
    classifiers = [
        SVC(kernel='poly', C=1, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=100, probability=True),
        SVC(kernel='poly', C=1, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=10, probability=True)]
    index = 0
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        svc = classifiers[index]
        cv = StratifiedKFold(n_splits=10)

        # plt.subplot(1, 2, index + 1)
        for train, test in cv.split(X, y):
            probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC=%0.3f)' % mean_auc, lw=2, alpha=.8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of MCI')
        plt.legend(loc='lower right', fontsize=10)
        index += 1
    plt.show()


def gridsearch(data, group):
    print('*** %s svm classification***' % group)
    y = data.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    clf = SVC(kernel='poly', degree=1, gamma=1, probability=True)
    # cv = StratifiedKFold(n_splits=10, random_state=9)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)
    params = {
        'C': [0.1, 1, 10],
        'coef0': [0, 0.1, 1, 10, 100],
        'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 1}, 'balanced']
    }
    scoring = make_scorer(roc_auc_score, needs_proba=True)
    grid = GridSearchCV(clf, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best AUROC score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    # pd.DataFrame(grid.cv_results_).to_csv('./clf%s.csv' % group, index=0)


def df_svc_grid(X, y):
    # print('*** %s svm classification***' % group)
    # X = MinMaxScaler().fit_transform(data.values)
    clf = SVC(kernel='poly', degree=1, class_weight={0: 1, 1: 2}, gamma=1, probability=True)
    cv = StratifiedKFold(n_splits=5, random_state=9)
    params = {
        'C': [0.1, 1, 10],
        'coef0': [0, 0.1, 1, 10, 100],
    }
    scoring = make_scorer(roc_auc_score, needs_proba=True)
    # scores = cross_val_score(mlp, scoring=scoring, X=X, y=y, cv=cv)
    grid = GridSearchCV(clf, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best AUROC score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    return grid.best_estimator_
    # pd.DataFrame(grid.cv_results_).to_csv('./clf%s.csv' % group, index=0)


def cv(data, group):
    print('*** %s svm classification***' % group)
    y = data.pop('DECLINED').values
    # X = MinMaxScaler().fit_transform(data.values)
    X = data.values
    cv = StratifiedKFold(n_splits=10)
    scores = []
    for train_index, test_index in cv.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        cf = GCForest(estimators_config={
            'mgs': [{'estimator_class': ExtraTreesClassifier,
                     'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }
                     }, {'estimator_class': RandomForestClassifier,
                         'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }}],
            'cascade': [{'estimator_class': ExtraTreesClassifier,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': ExtraTreesClassifier,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestClassifier,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                              'n_jobs': -1, }},
                        {'estimator_class': RandomForestClassifier,
                         'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 'sqrt',
                                              'n_jobs': -1, }}]})
        cf.fit(X_train, y_train)
        X_train = cf.transform(X_train)
        # svc = SVC(kernel='poly', degree=1, C=1, coef0=100, class_weight={0: 1, 1: 2}, gamma=1, probability=True)
        # svc.fit(X_train, y_train)
        svc = df_svc_grid(X_train, y_train)
        X_test = cf.transform(X_test)
        pred = svc.predict_proba(X_test)
        auroc = roc_auc_score(y_test, pred[:, 1])
        print(auroc)
        scores.append(auroc)
    scores = np.asarray(scores)
    print(scores)
    print("************ auroc: ", scores.mean(), '\n')


"""
without ADNI
[0.546875   0.3828125  0.8515625  1.         1.         0.625
 0.69166667 0.7047619  0.86666667 0.35238095]
************ auroc:  0.702172619047619

[0.4140625  0.3828125  0.8828125  1.         1.         0.60833333
 0.625      0.6952381  0.73333333 0.44761905]
************ auroc:  0.678921130952381

[0.3984375  0.359375   0.875      1.         1.         0.74166667
 0.675      0.68571429 0.81904762 0.45714286]
************ auroc:  0.7011383928571429 

with ADNI
[0.6640625  0.6640625  0.9296875  0.86666667 0.79166667 0.925
 0.79166667 0.4952381  0.8        0.76190476]
************ auroc:  0.7689955357142857

[0.734375   0.6953125  0.921875   0.84166667 0.78333333 0.86666667
 0.825      0.46666667 0.82857143 0.72380952]
************ auroc: 0.7687276785714285
 
[0.734375   0.65625    0.9296875  0.825      0.775      0.925
 0.83333333 0.45714286 0.79047619 0.76190476]
************ auroc:  0.7688169642857143 
"""


def roc():
    files = ['clf_MCI', 'clf_MCI_extra_data']
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = data.values
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cv = StratifiedKFold(n_splits=10)

        for train, test in cv.split(X, y):
            df = GCForest(estimators_config={
                'mgs': [{'estimator_class': ExtraTreesClassifier,
                         'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }
                         }, {'estimator_class': RandomForestClassifier,
                             'estimator_params': {'n_estimators': 30, 'min_samples_split': 0.1, 'n_jobs': -1, }}],
                'cascade': [{'estimator_class': ExtraTreesClassifier,
                             'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                                  'n_jobs': -1, }},
                            {'estimator_class': ExtraTreesClassifier,
                             'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1,
                                                  'max_features': 'sqrt', 'n_jobs': -1, }},
                            {'estimator_class': RandomForestClassifier,
                             'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1, 'max_features': 1,
                                                  'n_jobs': -1, }},
                            {'estimator_class': RandomForestClassifier,
                             'estimator_params': {'n_estimators': 1000, 'min_samples_split': 0.1,
                                                  'max_features': 'sqrt', 'n_jobs': -1, }}]})
            df.fit(X[train], y[train])
            probas_ = df.predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=r'Mean ROC (AUC=%0.3f)' % mean_auc, lw=2, alpha=.8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of MCI')
        plt.legend(loc='lower right', fontsize=10)
    plt.show()


def roc_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    gridsearch(MCI_o, 'MCI with basic data')
    """
    *** MCI with basic data svm classification***
    X.shape:  (196, 6) y.shape:  (196,)
    best AUROC score: 0.6162591575091576
    best parameters:  {'C': 1, 'class_weight': 'balanced', 'coef0': 1} 
    """
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    gridsearch(MCI_ADNI, 'MCI with ADNI features')
    '''
    *** MCI with ADNI features svm classification***
    X.shape:  (196, 8) y.shape:  (196,)
    best AUROC score: 0.786098901098901
    best parameters:  {'C': 10, 'class_weight': {0: 1, 1: 2}, 'coef0': 10} 
    '''

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    # arg_maxs = [986, 886, 836, 2006, 2004, 211, 1861, 1854, 86]
    arg_maxs = [986, 2006, 1904, 2080]
    # arg_maxs = [886, 986, 836, 1903, 1011, 1904, 2004, 1854, 86]
    for i in range(len(arg_maxs)):
        MCI_o['img_feature{}'.format(i)] = imaging.iloc[:, arg_maxs[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        gridsearch(MCI_o.copy(), 'MCI with selected imaging features')
    '''
    *** MCI with selected imaging features svm classification***
    X.shape:  (196, 12) y.shape:  (196,)
    best AUROC score: 0.8124198717948719
    best parameters:  {'C': 10, 'class_weight': {0: 1, 1: 1}, 'coef0': 10}
    '''

    # # with imaging data
    # MCI_imaging = df[df.DX_bl == 2].copy()
    # MCI_img = MCI_imaging.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    # gridsearch(MCI_img, 'MCI with imaging data')
    """
    *** MCI with imaging data svm classification***
    best AUROC score: 0.5271520146520147
    best parameters:  {'C': 1, 'coef0': 0} 
    """


if __name__ == "__main__":
    # roccurve()
    # clfmci = pd.read_csv('./data_genetic/clf_MCI.csv')
    # gridsearch(clfmci, 'MCI with original data')
    # clfmci = pd.read_csv('./data_genetic/clf_MCI_extra_data.csv')
    # gridsearch(clfmci, 'MCI with ADNI data')
    # clfmci = pd.read_csv('./data_genetic/clf_MCI.csv')
    # cv(clfmci, 'MCI with original data')
    # clfmci = pd.read_csv('./data_genetic/clf_MCI_extra_data.csv')
    # cv(clfmci, 'MCI with ADNI data')
    # roc()  # deep forest for mci roc
    roc_with_imaging_data()
    # clfmci = pd.read_csv('./data_genetic/clf_MCI.csv')
    # cv(clfmci, 'MCI with original data')
    # clfmci = pd.read_csv('./data_genetic/clf_MCI_extra_data.csv')
    # cv(clfmci, 'MCI with ADNI data')
