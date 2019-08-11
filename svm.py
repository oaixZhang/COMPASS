import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, make_scorer, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, \
    RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from gcforest import GCForest


def roccurve():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    arg_max = [986, 2006, 1904, 2080]
    for i in range(len(arg_max)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max[i]]
    data = [MCI_o, MCI_ADNI, MCI_img]
    labels = ['Basic clinical features', 'With ADNI features', 'With ADNI and MRI features']
    classifiers = [
        SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=1, probability=True),
        SVC(kernel='poly', C=10, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=100, probability=True),
        SVC(kernel='poly', C=10, class_weight={0: 1, 1: 1}, gamma=1, degree=1, coef0=0.1, probability=True)]
    index = 0
    for d in data:
        y = d.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(d.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        svc = classifiers[index]
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=9)

        for train, test in cv.split(X, y):
            probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label=r'%s (AUC=%0.3f)' % (labels[index], mean_auc), lw=2, alpha=.8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right', fontsize=10)
        index += 1
    plt.show()


def roccurve_comparison():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    arg_max = [986, 2006, 1904, 2080]
    for i in range(len(arg_max)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max[i]]
    data = [MCI_ADNI, MCI_img]

    labels = ['iCOMPASS', 'XGBoost', 'Logistic Regression', 'Random Forest', 'Gaussian']
    plotauc = [0.785, 0.704, 0.754, 0.680, 0.739]
    # plotauc=[0.818,0.768,0.782,0.754,0.783]
    classifiers = [
        # SVC(kernel='poly', C=10, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=100, probability=True),
        SVC(kernel='rbf', C=10, class_weight={0: 1, 1: 2}, gamma=0.1, probability=True),
        # SVC(kernel='poly', C=10, class_weight={0: 1, 1: 1}, gamma='auto', degree=3, coef0=100, probability=True),
        # LinearSVC(class_weight={0: 2, 1: 1}),
        XGBClassifier(10),
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(n_estimators=10),
        GaussianProcessClassifier(random_state=0)]
    index = 0
    fig = plt.figure()
    for classifier in classifiers:
        d = data[0].copy()
        y = d.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(d.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)

        for train, test in cv.split(X, y):
            if hasattr(classifier, 'decision_function'):
                probas_ = classifier.fit(X[train], y[train]).decision_function(X[test])
                fpr, tpr, thresholds = roc_curve(y[test], probas_)
            else:
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        # plt.plot(mean_fpr, mean_tpr, label=r'%s (AUC=%0.3f)' % (labels[index], mean_auc), lw=2, alpha=.8)
        print(mean_auc)
        plt.plot(mean_fpr, mean_tpr, label=r'%s (AUC=%0.3f)' % (labels[index], plotauc[index]), lw=2, alpha=.8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right', fontsize=10)
        index += 1
    plt.show()
    fig.savefig('./roc_comparison1.eps', dpi=500)


def pre_rec_curve_comparison():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    arg_max = [986, 2006, 1904, 2080]
    for i in range(len(arg_max)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max[i]]
    data = [MCI_ADNI, MCI_img]

    labels = ['COM_MRI', 'COMPASS', 'Logistic Regression', 'Random Forest', 'Gaussian']

    classifiers = [
        # SVC(kernel='poly', C=10, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=100, probability=True),
        SVC(kernel='rbf', C=10, class_weight={0: 1, 1: 2}, gamma=0.1, probability=True),
        # SVC(kernel='poly', C=10, class_weight={0: 1, 1: 1}, gamma='auto', degree=3, coef0=100, probability=True),
        # LinearSVC(class_weight={0: 2, 1: 1}),
        XGBClassifier(10),
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(n_estimators=100),
        GaussianProcessClassifier(random_state=0)]
    index = 0
    for classifier in classifiers:
        d = data[1].copy()
        y = d.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(d.values)
        aucs = []
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)
        for train, test in cv.split(X, y):
            if hasattr(classifier, 'decision_function'):
                probas_ = classifier.fit(X[train], y[train]).decision_function(X[test])
                pre_rec_auc = average_precision_score(y[test], probas_)
            else:
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                pre_rec_auc = average_precision_score(y[test], probas_[:, 1])
            aucs.append(pre_rec_auc)
        mean_auc = np.mean(aucs)
        print(labels[index], mean_auc)
        index += 1


def pre_rec_curve():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    arg_max = [986, 2006, 1904, 2080]
    for i in range(len(arg_max)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max[i]]
    # data = [MCI_o, MCI_ADNI, MCI_img]
    data = [MCI_ADNI, MCI_img]

    # labels = ['Basic clinical features', 'With ADNI features', 'With ADNI and MRI features']
    labels = ['With ADNI features', 'With ADNI and MRI features']
    classifiers = [
        # SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=1, probability=True),
        SVC(kernel='poly', C=10, class_weight={0: 1, 1: 2}, gamma=1, degree=1, coef0=100, probability=True),
        # SVC(kernel='poly', C=10, class_weight={0: 1, 1: 1}, gamma=1, degree=1, coef0=0.1, probability=True),
        SVC(kernel='rbf', C=10, class_weight={0: 1, 1: 2}, gamma=0.1, probability=True)]

    index = 0
    for d in data:
        y = d.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(d.values)
        aucs = []
        svc = classifiers[index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=18)
        yscore = svc.fit(X_train, y_train).decision_function(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, yscore)
        auc = average_precision_score(y_test, yscore)
        print(auc)
        # plt.plot(recall, precision, label=r'%s (AUC=%0.3f)' % (labels[index], mean_auc[index]), lw=2, alpha=.8)
        plt.step(recall, precision, label=r'%s (AUC=%0.3f)' % (labels[index], mean_auc[index]), lw=2, alpha=.5)
        print(recall)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower right', fontsize=10)
        index += 1
    plt.show()


def gridsearch(data, group):
    print('*** %s svm classification***' % group)
    y = data.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    # clf = SVC(kernel='poly', degree=1, gamma=1, probability=True)
    clf = SVC(probability=True)
    # cv = StratifiedKFold(n_splits=10, random_state=9)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)
    params = [{'kernel': ['poly'], 'degree': [3], 'gamma': [0.1, 'auto'], 'C': [0.1, 1, 10],
               'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 1}, 'balanced'],
               'coef0': [0, 0.1, 1, 10, 100]}]
    # params = {'kernel': ['rbf'], 'gamma': [0.1, 1, 'auto'], 'C': [0.1, 100, 1, 10],
    #           'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 1}, 'balanced']}
    scoring = make_scorer(roc_auc_score, needs_proba=True)
    grid = GridSearchCV(clf, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best AUROC score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    # pd.DataFrame(grid.cv_results_).to_csv('./clf%s.csv' % group, index=0)


def spearman(data, group):
    print('*** %s svm classification***' % group)
    y = data.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    clf = SVC(kernel='poly', degree=1, gamma=1, probability=True)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=100, random_state=9)
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


def roc_with_imaging_data():
    df = pd.read_csv('./data_genetic/imaging_data.csv')
    data = df.iloc[:, 0:12].copy()
    # original data
    MCI = data[data.DX_bl == 2].copy()
    MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    gridsearch(MCI_o, 'MCI with basic data')
    # with ADNI features
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    gridsearch(MCI_ADNI, 'MCI with ADNI features')

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'deltaMMSE'])
    arg_max = [986, 2006, 1904, 2080]
    for i in range(len(arg_max)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
    gridsearch(MCI_img.copy(), 'MCI with selected imaging features')


if __name__ == "__main__":
    # roccurve()
    roccurve_comparison()
    # pre_rec_curve()
    # pre_rec_curve_comparison()
