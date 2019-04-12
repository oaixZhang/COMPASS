import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from svr import cal_pearson
from svm import calculate_auroc


def regmlp(data, group):
    print('*** %s MLP regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    mlp = MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive', max_iter=2000, tol=0.001,
                        learning_rate_init=0.01)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=9)
    params = {
        'hidden_layer_sizes': [(20,), (50,), (20, 6), (50, 6), (20, 10), (50, 10), (50, 10, 6), (20, 10, 6), (20, 8),
                               (50, 8)],
        'batch_size': [5, 10, 20, 50],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'random_state': [0, 9, 1, 6],
        'momentum': [0.9, 0.8, 0.7]
    }
    scoring = make_scorer(cal_pearson)
    # scores = cross_val_score(mlp, scoring=scoring, X=X, y=y, cv=cv)
    grid = GridSearchCV(mlp, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./mlp%s.csv' % group, index=0)


def clfmlp(data, group):
    print('*** %s MLP classification***' % group)
    y = data.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(data.values)
    mlp = MLPClassifier(activation='tanh', solver='sgd', learning_rate='adaptive', max_iter=2000, tol=0.001,
                        learning_rate_init=0.01)
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=9)
    params = {
        'hidden_layer_sizes': [(20,), (50,), (20, 6), (50, 6), (20, 10), (50, 10), (50, 10, 6), (20, 10, 6)],
        'batch_size': [10, 20],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'random_state': [0, 9, 1, 6],
        # 'momentum': [0.9, 0.8, 0.7]
    }
    scoring = make_scorer(calculate_auroc)
    # scores = cross_val_score(mlp, scoring=scoring, X=X, y=y, cv=cv)
    grid = GridSearchCV(mlp, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best AUROC score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./mlpclf%s.csv' % group, index=0)




if __name__ == '__main__':
    data = pd.read_csv('./data_genetic/data_all_features.csv')
    data.pop('APOE3')
    # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # regmlp(CN_o, 'CN with original data')
    # MCI = data[data.DX_bl == 2].copy()
    # MCI_o = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # regmlp(MCI_o, 'MCI with original data')
    # AD = data[data.DX_bl == 3].copy()
    # AD_o = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # regmlp(AD_o, 'AD with original data')
    #
    # # with ADNI features
    # CN_ADNI = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # regmlp(CN_ADNI, 'CN with ADNI features')
    #
    # MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # regmlp(MCI_ADNI, 'MCI with ADNI features')
    #
    # AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'DECLINED'])
    # regmlp(AD_ADNI, 'AD with ADNI features')
    clfmci = pd.read_csv('./data_genetic/clf_MCI.csv')
    clfmlp(clfmci, 'MCI with original data')
    clfmci = pd.read_csv('./data_genetic/clf_MCI_extra_data.csv')
    clfmlp(clfmci, 'MCI with ADNI data')
