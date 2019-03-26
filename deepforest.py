import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV, cross_validate, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer
from svm import cal_pearson
from CascadeForest import CascadeForest


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


def crossvalidate(X, y, estimator):
    estimator = estimator
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=9)
    scoring = make_scorer(cal_pearson)
    results = cross_val_score(estimator, X, y, scoring=scoring, cv=cv)
    # print('scores:', results['test_score'])
    print('mean score:', results.mean(), '\n')


if __name__ == '__main__':
    data = pd.read_csv('./data_genetic/data_all_features.csv')

    # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    # randomForest(CN_o, 'original CN')
    y = CN_o.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(CN_o.values)
    # print('X.shape: ', X.shape, 'y.shape: ', y.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

    rf = RandomForestRegressor(n_estimators=101, min_samples_split=0.1, max_features=1, oob_score=True)
    crossvalidate(X, y, rf)

    ef = ExtraTreesRegressor(n_estimators=101, min_samples_split=0.1, max_features=1, oob_score=True)
    crossvalidate(X, y, rf)

    lr = LinearRegression()
    crossvalidate(X, y, lr)

    # cf = CascadeForest()
    # crossvalidate(X, y, cf)

    # cf.fit(X_train, y_train)
    # cf_pred = cf.predict(X_test)
    # cf_pred = np.mean(cf_pred, axis=0)
    # print('cf pearson:', cal_pearson(y_test, cf_pred))
