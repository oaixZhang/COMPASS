import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import interp
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, make_scorer, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_validate, cross_val_score, \
    RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from CascadeForest import CascadeForest
from GCForest import gcForest
from gcforest.gcforest import GCForest
from svr import cal_pearson


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


def randomForest(data, group):
    print('*** %s random forest regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    # print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    rf = RandomForestRegressor()
    params = {'n_estimators': [10, 50, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    # scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(rf, params, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


def crossvalidate(X, y, estimator, scoring=make_scorer(cal_pearson)):
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    results = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    # print('scores:', results['test_score'])
    print('mean score:', results.mean(), '\n')


def regression(group):
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    print('group=', group)
    X = data[data.DX_bl == group].copy()
    X = X.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    y = X.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(X.values)

    rf = RandomForestRegressor(n_estimators=101, min_samples_split=0.1, max_features=1, oob_score=True)
    crossvalidate(X, y, rf)

    cf = CascadeForest()
    crossvalidate(X, y, cf)

    print('with ADNI FEATURE')
    X = data[data.DX_bl == group].copy()
    X = X.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    y = X.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(X.values)

    rf = RandomForestRegressor(n_estimators=101, min_samples_split=0.1, max_features=1, oob_score=True)
    crossvalidate(X, y, rf)

    cf = CascadeForest()
    crossvalidate(X, y, cf)

    # ef = ExtraTreesRegressor(n_estimators=101, min_samples_split=0.1, max_features=1, oob_score=True)
    # crossvalidate(X, y, rf)

    # lr = LinearRegression()
    # crossvalidate(X, y, lr)

    # cf.fit(X_train, y_train)
    # cf_pred = cf.predict(X_test)
    # cf_pred = np.mean(cf_pred, axis=0)
    # print('cf pearson:', cal_pearson(y_test, cf_pred))


def cv4clf(X, y, estimator):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = ['accuracy_score', 'precision_score']
    results = cross_validate(estimator, X, y, scoring=scoring, cv=cv, return_train_score=False)
    print('mean score:', results, '\n')


def clf():
    data = pd.read_csv('./data_genetic/data_all_features.csv')

    # original data
    MCI = data[data.DX_bl == 2].copy()
    MCI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'deltaMMSE'])
    y = MCI.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(MCI.values)

    gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
    # cv4clf(X, y, gcf)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
    gcf.fit(X_train, y_train)
    pred = gcf.predict(X_test)
    print('y_test', y_test)
    print('y_pred', pred)
    print('accuracy:', accuracy_score(y_test, pred))
    print('precision:', precision_score(y_test, pred))
    print('f1 score:', f1_score(y_test, pred))


def roc_df():
    files = ['clf_CN', 'clf_MCI', 'clf_AD', 'clf_CN_extra_data', 'clf_MCI_extra_data', 'clf_AD_extra_Data']
    titles = ['CN_Without_ADNI_feature', 'MCI_Without_ADNI_feature', 'AD_Without_ADNI_feature', 'CN_With_ADNI_feature',
              'MCI_With_ADNI_feature', 'AD_With_ADNI_feature']
    index = 0
    mean_tprs = []
    config = get_toy_config()
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        if index % 3 == 0:
            cv = StratifiedKFold(n_splits=4, random_state=0)
        else:
            cv = StratifiedKFold(n_splits=5, random_state=0)
        plt.subplot(2, 3, index + 1)
        for train, test in cv.split(X, y):
            clf = gcForest(shape_1X=4, window=2, tolerance=0.0)
            clf.fit(X[train], y[train])
            probas_ = clf.predict(X[test])

            # clf = GCForest(config)
            # clf.fit_transform(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

            # clf = LogisticRegression(solver='liblinear')
            # clf.fit(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

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
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.2f)' % mean_auc, lw=2, alpha=.8)
        mean_tprs.append(mean_tpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        plt.title(titles[index], fontsize=6)
        plt.legend(loc='lower right', fontsize=6)
        index += 1

    plt.show()

    mean_tprs = np.array(mean_tprs)
    weighted_tpr = mean_tprs[0] * 152 / 489 + mean_tprs[1] * 230 / 489 + mean_tprs[2] * 107 / 489
    weighted_tpr_with_extra = mean_tprs[3] * 152 / 489 + mean_tprs[4] * 230 / 489 + mean_tprs[5] * 107 / 489
    weighted_auc = auc(mean_fpr, weighted_tpr)
    plt.plot(mean_fpr, weighted_tpr, color='b', label=r'Without ADNI features (AUC=%0.2f)' % weighted_auc, lw=2,
             alpha=.8)
    weighted_auc_with_extra = auc(mean_fpr, weighted_tpr_with_extra)
    plt.plot(mean_fpr, weighted_tpr_with_extra, color='r',
             label=r'With ADNI features (AUC=%0.2f)' % weighted_auc_with_extra,
             lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=.5)
    plt.title('ROC Curve')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()


def roc_MCI():
    files = ['clf_MCI', 'clf_MCI_extra_data']
    titles = ['MCI_Without_ADNI_feature', 'MCI_With_ADNI_feature']
    classifiers = [
        gcForest(shape_1X=6, window=3, tolerance=0.0),
        gcForest(shape_1X=8, window=3, tolerance=0.0)
        # SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=100, probability=True),
        # SVC(kernel='poly', C=1, class_weight='balanced', gamma=1, degree=1, coef0=1, probability=True)
    ]
    index = 0
    mean_tprs = []
    config = get_toy_config()
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
            # clf = classifiers[index]
            # clf.fit(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

            # gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
            # gcf.cascade_forest(X[train], y[train])
            # probas_ = gcf.cascade_forest(X[test])
            # probas_ = np.mean(probas_, axis=0)

            clf = GCForest(config)
            clf.fit_transform(X[train], y[train])
            probas_ = clf.predict_proba(X[test])

            # clf = LogisticRegression(solver='liblinear')
            # clf.fit(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

            # clf = SVC(probability=True,gamma=0.1)
            # clf.fit(X[train], y[train])
            # probas_ = clf.predict_proba(X[test])

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
    # regression(1)
    # regression(2)
    # regression(3)
    # clf()
    # roc_whole()
    # roc_df()
    # test()
    roc_MCI()
