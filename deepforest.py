import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from gcforest.gcforest import GCForest


def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 10
    ca_config["estimators"] = []
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 10, "max_depth": 5,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 10, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "LogisticRegression"})
    config["cascade"] = ca_config
    return config


def gcf(data, group):
    print('*** %s linear regression ***' % group)
    y = data.pop('deltaMMSE').values
    X = MinMaxScaler().fit_transform(data.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    config = get_toy_config()
    gcf = GCForest(config)
    # cv = RepeatedKFold(n_splits=10, n_repeats=1, random_state=9)
    # scoring = make_scorer(cal_pearson)
    # results = cross_validate(gcf, X, y, scoring=scoring, cv=cv, return_train_score=False)
    # print('mean score:', results['test_score'].mean())
    gcf.fit_transform(X_train, y_train)
    y_pred = gcf.predict(X_test)
    # print('score: ', cal_pearson(y_test, y_pred))
    print('result: ', y_pred)


if __name__ == '__main__':
    data = pd.read_csv('./data_genetic/data_all_features.csv')

    # original data
    CN = data[data.DX_bl == 1].copy()
    CN_o = CN.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    gcf(CN_o, 'CN with original data')
