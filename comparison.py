import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from svr import cal_pearson


def regmlp(data, group):
    print('*** %s mlp regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    mlp = MLPRegressor(activation='relu', solver='sgd', max_iter=2000, learning_rate_init=0.01)
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=9)
    params = {
        'hidden_layer_sizes': [(50,), (20, 10), (50, 10), (100,)],
        'batch_size': [10, 'auto'], }
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(mlp, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


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
    }
    scoring = make_scorer(cal_pearson)
    # scores = cross_val_score(mlp, scoring=scoring, X=X, y=y, cv=cv)
    grid = GridSearchCV(mlp, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)
    grid.fit(X, y)
    print('best AUROC score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')
    pd.DataFrame(grid.cv_results_).to_csv('./mlpclf%s.csv' % group, index=0)


def gridsearch(data, group):
    print('*** %s svr regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = data.values
    # X = MinMaxScaler().fit_transform(data.values)
    print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    svr = RandomForestRegressor()
    params = {'n_estimators': [50, 100], 'min_samples_split': [0.1, 2], 'max_features': ["auto", 'sqrt']}
    cv = RepeatedKFold(n_splits=10, n_repeats=100, random_state=9)
    scoring = make_scorer(cal_pearson)
    grid = GridSearchCV(svr, params, scoring=scoring, cv=cv, return_train_score=False, iid=False)

    grid.fit(X, y)
    print('best Pearson score:', grid.best_score_)
    print('best parameters: ', grid.best_params_, '\n')


# 422 samples
def with_imaging_data():
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
    regmlp(CN_ADNI, 'CN with ADNI features')
    MCI_ADNI = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    regmlp(MCI_ADNI, 'MCI with ADNI features')
    AD_ADNI = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    regmlp(AD_ADNI, 'AD with ADNI features')

    # basic clinical features + imaging feature
    imaging = df.iloc[:, 12:].copy()  # imaging data
    CN_img = CN.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_CN = [1707, 1553, 730, 73, 1758]
    for i in range(len(arg_max_CN)):
        CN_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_CN[i]]
        # MCI_o = pd.concat([MCI_o, imaging.iloc[:, arg_maxs[i]]], axis=1, join='inner')
        # print(MCI_o)
        print(imaging.iloc[:, arg_max_CN[i]].name)
    regmlp(CN_img.copy(), 'CN with imaging data , i={}'.format(i))

    MCI_img = MCI.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_MCI = [986, 2006, 1904, 2080]
    for i in range(len(arg_max_MCI)):
        MCI_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_MCI[i]]
        print(imaging.iloc[:, arg_max_MCI[i]].name)
    regmlp(MCI_img.copy(), 'MCI with imaging data , i={}'.format(i))

    AD_img = AD.drop(columns=['RID', 'DX_bl', 'DECLINED'])
    arg_max_AD = [2055, 981, 218, 5]
    for i in range(len(arg_max_AD)):
        AD_img['img_feature{}'.format(i)] = imaging.iloc[:, arg_max_AD[i]]
        print(imaging.iloc[:, arg_max_AD[i]].name)
    regmlp(AD_img.copy(), 'AD with imaging data , i={}'.format(i))


if __name__ == '__main__':
    with_imaging_data()
