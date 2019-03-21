import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

import svm


def crossvalidate(file, params, splits):
    print('*** {} ***'.format(file))
    data = pd.read_csv('./data_genetic/{}.csv'.format(file))
    y = data.pop('DECLINED').values
    X = data.values
    pipe = Pipeline([
        ('norm', MinMaxScaler(feature_range=(0, 1))),
        ('clf', SVC(kernel='poly', degree=1, coef0=100, gamma=1)
         )])
    # params = {'clf__C': [0.1, 1, 10, 100],
    #           'clf__class_weight': [{0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 12}, 'balanced']}
    cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=10, random_state=9)
    scoring = make_scorer(svm.cal_pearson)
    # scoring = make_scorer(roc_auc_score)
    grid = GridSearchCV(pipe, params, scoring=scoring, iid=False, cv=cv, return_train_score=False)

    grid.fit(X, y)

    print('best Pearson score:', grid.best_score_)
    print('best params:', grid.best_params_)

    pd.DataFrame(grid.cv_results_).to_csv('./grid_new/{}.csv'.format(file), index=0)


def svc_crossvalidate(file, params, cv):
    print('*** {} ***'.format(file))
    data = pd.read_csv('./data_genetic/{}.csv'.format(file))
    y = data.pop('DECLINED').values
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)

    scoring = make_scorer(svm.cal_pearson)
    grid = GridSearchCV(SVC(kernel='poly', degree=1, coef0=100, random_state=0), params, scoring=scoring, cv=cv,
                        return_train_score=False, iid=True)
    grid.fit(X, y)
    result = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./grid/{}.csv'.format(file), index=0)

    return result

#
# def cn_without_extra_data():
#     cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=9)
#     params = [
#         {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10], 'C': [1, 10, 100, 1000],
#          'class_weight': ['balanced', {0: 1, 1: 8}, {0: 1, 1: 10}]}]
#     svc_crossvalidate('clf_CN', params, cv)
#
#
# def mci_without_extra_data():
#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=9)
#     params = [{'kernel': ['poly'], 'degree': [1, 2], 'C': [1, 10, 100, 1000],
#                'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'gamma': [0.01, 0.1, 1, 10],
#                'coef0': [0, 0.1, 1, 10, 100]},
#               {'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10],
#                'class_weight': ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}], 'C': [1, 10, 100, 1000]}]
#     svc_crossvalidate('clf_MCI', params, cv)


if __name__ == '__main__':
    # df = pd.read_csv('./data_genetic/data_genetic.csv')
    # CN = df[df.DX_bl == 1].copy()
    # CN['DECLINED'] = [int(delta <= -3) for delta in CN['deltaMMSE']]
    # CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'deltaMMSE'], inplace=True)
    crossvalidate('clf_CN',
                  {'clf__C': [0.1, 1, 10, 100],
                   'clf__class_weight': [{0: 1, 1: 8}, {0: 1, 1: 10}, {0: 1, 1: 12}, 'balanced']}, 3)
    # MCI = df[df.DX_bl == 2].copy()
    # MCI['DECLINED'] = [int(delta <= -3) for delta in MCI['deltaMMSE']]
    # MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'deltaMMSE'], inplace=True)
    crossvalidate('clf_MCI',
                  {'clf__C': [0.1, 1, 10, 100],
                   'clf__class_weight': [{0: 1, 1: 1}, {0: 1, 1: 2}, {0: 1, 1: 3}, 'balanced']}, 10)

    # AD = df[df.DX_bl == 3].copy()
    # AD['DECLINED'] = [int(delta <= -3) for delta in AD['deltaMMSE']]
    # AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'deltaMMSE'], inplace=True)
    crossvalidate('clf_AD',
                  {'clf__C': [0.1, 1, 10, 100],
                   'clf__class_weight': [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 1, 1: 2}, 'balanced']}, 10)

    # cn_without_extra_data()
    # mci_without_extra_data()
