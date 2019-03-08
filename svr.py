import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from svm_crossvalidate import cal_pearson
import svm_crossvalidate, svm_oversampling


def svr_CN():
    print('*** CN svr regression ***')
    CN = pd.read_csv('./data/reg_CN.csv')
    y_CN = CN.pop('deltaMMSE')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [100, 0.1, 1, 10]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    grid.fit(X_CN, y_CN)
    cn = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_CN_svr_o.csv', index=0)


def svr_MCI():
    print('*** MCI svr regression ***')
    MCI = pd.read_csv('./data/reg_MCI.csv')
    y_MCI = MCI.pop('deltaMMSE')
    X_MCI = MinMaxScaler().fit_transform(MCI.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [100, 0.1, 1, 10]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_MCI_svr_o.csv', index=0)


def svr_AD():
    print('*** AD svr regression ***')
    AD = pd.read_csv('./data/reg_AD.csv')
    y_AD = AD.pop('deltaMMSE')
    X_AD = MinMaxScaler().fit_transform(AD.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_AD, y_AD)
    ad = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_AD_svr_o.csv', index=0)


def svr_CN_extra_data():
    print('*** CN regression ***')
    CN = pd.read_csv('./data/reg_CN_extra_data.csv')
    y_CN = CN.pop('deltaMMSE')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)

    grid.fit(X_CN, y_CN)
    cn = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ')
    for key in params:
        print(key, cn.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_CN_svr.csv', index=0)


def svr_MCI_extra_data():
    print('*** MCI regression ***')
    MCI = pd.read_csv('./data/reg_MCI_extra_data.csv')
    y_MCI = MCI.pop('deltaMMSE')
    X_MCI = MinMaxScaler().fit_transform(MCI.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_MCI, y_MCI)
    mci = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_MCI_svr.csv', index=0)


def svr_AD_extra_data():
    print('*** AD regression ***')
    AD = pd.read_csv('./data/reg_AD_extra_data.csv')
    y_AD = AD.pop('deltaMMSE')
    X_AD = MinMaxScaler().fit_transform(AD.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2, 3], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X_AD, y_AD)
    ad = grid.best_estimator_
    print('best Pearson score:', grid.best_score_)
    print('best parameters:')
    for key in params:
        print(key, ad.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_AD_svr.csv', index=0)


def run():
    print('#### without ADNI-MEM ADNI-EF data ####')
    svr_CN()
    svr_MCI()
    svr_AD()
    print('#### with ADNI-MEM ADNI-EF data ####')
    svr_CN_extra_data()
    svr_MCI_extra_data()
    svr_AD_extra_data()


def SMOregPolyKernel(file, E):
    print('*** {} regression poly kernel ***'.format(file))
    data = pd.read_csv('./data/{}.csv'.format(file))
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    svr = SVR(kernel='poly', degree=E)
    params = {'coef0': [0, 0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X, y)
    mci = grid.best_estimator_
    print('best Pearson score {} SMOregE{}:'.format(file, E), grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_{}_SMOregE{}.csv'.format(file, E), index=0)


def SMOregRBFKernel(file):
    print('*** {} regression RBF kernel ***'.format(file))
    data = pd.read_csv('./data/{}.csv'.format(file))
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    svr = SVR(kernel='rbf')
    params = {'gamma': [0.01, 0.1, 1, 10, 100]}
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=0)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=True)
    grid.fit(X, y)
    mci = grid.best_estimator_
    print('best Pearson score {} :'.format(file), grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_{}_RBF.csv'.format(file), index=0)


if __name__ == "__main__":
    # print('#### SMOreg ####')
    for i in range(2, 4):
        SMOregPolyKernel('reg_CN', i)
        SMOregPolyKernel('reg_MCI', i)
        SMOregPolyKernel('reg_AD', i)
        SMOregPolyKernel('reg_CN_extra_data', i)
        SMOregPolyKernel('reg_MCI_extra_data', i)
        SMOregPolyKernel('reg_AD_extra_data', i)
    print('#### SVR RBF kernel ####')
    SMOregRBFKernel('reg_CN')
    SMOregRBFKernel('reg_MCI')
    SMOregRBFKernel('reg_AD')
    SMOregRBFKernel('reg_CN_extra_data')
    SMOregRBFKernel('reg_MCI_extra_data')
    SMOregRBFKernel('reg_AD_extra_data')
