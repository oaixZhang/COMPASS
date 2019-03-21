import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from svm_crossvalidate import cal_pearson


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
    result = grid.best_estimator_
    print('best Pearson score {} :'.format(file), grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, result.get_params()[key])
    pd.DataFrame(grid.cv_results_).to_csv('./grid/grid_{}_RBF.csv'.format(file), index=0)


def svr(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    svr = SVR()
    params = {'coef0': [0, 0.1, 1, 10, 100, 1000], 'degree': [1,2], 'kernel': ['poly'],
              'gamma': [100, 0.1, 1, 10, 0.01, 1000], 'C': [0.1, 1, 10, 100, 0.01, 1000]}
    # params = {'kernel': ['rbf'], 'gamma': [100, 0.1, 1, 10], 'C': [0.1, 0.01, 1, 10, 100,1000]}
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


if __name__ == "__main__":
    # print('#### SMOreg ####')
    # for i in range(2, 4):
    #     SMOregPolyKernel('reg_CN', i)
    #     SMOregPolyKernel('reg_MCI', i)
    #     SMOregPolyKernel('reg_AD', i)
    #     SMOregPolyKernel('reg_CN_extra_data', i)
    #     SMOregPolyKernel('reg_MCI_extra_data', i)
    #     SMOregPolyKernel('reg_AD_extra_data', i)
    # print('#### SVR RBF kernel ####')
    # SMOregRBFKernel('reg_CN')
    # SMOregRBFKernel('reg_MCI')
    # SMOregRBFKernel('reg_AD')
    # SMOregRBFKernel('reg_CN_extra_data')
    # SMOregRBFKernel('reg_MCI_extra_data')
    # SMOregRBFKernel('reg_AD_extra_data')
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    # original data
    # CN = data[data.DX_bl == 1].copy()
    # CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(CN_o, 'CN with original data')
    #
    # MCI = data[data.DX_bl == 2].copy()
    # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(MCI_o, 'MCI with original data')

    AD = data[data.DX_bl == 3].copy()
    AD_o = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    print(AD_o)
    svr(AD_o, 'AD with original data')

    # all_o = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(all_o, 'overall with original data')
    #
    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # svr(CN_ADNI, 'CN with ADNI features')
    #
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # svr(MCI_ADNI, 'MCI with ADNI features')

    AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    print(AD_ADNI)
    svr(AD_ADNI, 'AD with ADNI features')

    # overall_ADNI = data.drop(columns=['RID', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # svr(overall_ADNI, 'overall with ADNI features')

    # # with genetic features
    # CN_genetic = CN.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(CN_genetic, 'CN with genetic features')
    #
    # MCI_genetic = MCI.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(MCI_genetic, 'MCI with genetic features')
    #
    # AD_genetic = AD.drop(columns=['RID', 'DX_bl', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(AD_genetic, 'AD with genetic features')
    #
    # overall_genetic = data.drop(columns=['RID', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # svr(overall_genetic, 'overall with genetic features')
    #
    # # plus genetic features
    # CN = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(CN, 'CN with all features')
    #
    # MCI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(MCI, 'MCI with all features')
    #
    # AD = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    # svr(AD, 'AD with all features')
    #
    # overall = data.drop(columns=['RID', 'DECLINED'])
    # svr(overall, 'overall with all features')
