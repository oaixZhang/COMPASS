import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def cal_pearson(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    # sum x_list、y_list
    sum_xy = np.sum(np.multiply(x, y))
    # x_list square、y_list square
    sum_x2 = sum([pow(i, 2) for i in x])
    sum_y2 = sum([pow(j, 2) for j in y])
    molecular = sum_xy - (float(sum_x) * float(sum_y) / n)
    # calculate
    denominator = np.sqrt((sum_x2 - float(sum_x ** 2) / n) * (sum_y2 - float(sum_y ** 2) / n))
    if denominator != 0:
        return molecular / denominator
    else:
        return 0


def rbf_kernel(X, y, cv):
    best_params = {}
    best_score = 0
    for weight in ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 1, 1: 3}]:
        for gamma in [0.01, 0.1, 1, 10, 100, 1000]:
            scores = []
            for train_index, test_index in cv.split(X, y):
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                y_test = y[test_index]
                # oversampling
                X_train, y_train = RandomOverSampler(sampling_strategy=1).fit_resample(X_train, y_train)

                svc = SVC(kernel='rbf', gamma=gamma, class_weight=weight)

                svc.fit(X_train, y_train)

                predict = svc.predict(X_test)
                scores.append(cal_pearson(y_test, predict))
            curscore = np.array(scores).mean()
            if curscore > best_score:
                best_score = curscore
                best_params = {'gamma': gamma, 'class_weight': weight}
    return best_score, best_params


def poly_kernel(X, y, cv):
    best_score = 0
    best_params = {}
    # poly
    for d in [1, 2]:
        for c in [0.01, 0.1, 1, 10, 100]:
            for weight in ['balanced', {0: 1, 1: 1}, {0: 1, 1: 2}, {0: 2, 1: 1}]:
                for gamma in [0.1, 0.01, 1, 10]:
                    scores = []
                    for train_index, test_index in cv.split(X, y):
                        X_train = X[train_index]
                        X_test = X[test_index]
                        y_train = y[train_index]
                        y_test = y[test_index]
                        # oversampling

                        X_train, y_train = RandomOverSampler().fit_resample(X_train, y_train)

                        svc = SVC(kernel='poly', degree=d, gamma=gamma, coef0=c, class_weight=weight)

                        svc.fit(X_train, y_train)

                        predict = svc.predict(X_test)
                        scores.append(cal_pearson(y_test, predict))
                    curscore = np.array(scores).mean()
                    if curscore > best_score:
                        best_score = curscore
                        best_params = {'d': d, 'c': c, 'class_weight': weight, 'gamma': gamma}
    return best_score, best_params


"""
without ADNI-MEM ADNI-EF
CN

"""


def cn_without_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN.csv')
    y = CN.pop('DECLINED').values
    X = MinMaxScaler().fit_transform(CN.values)

    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=0)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


"""
MCI
"""


def mci_without_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI.csv')
    y = MCI.pop('DECLINED').values
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(MCI.values)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


"""
AD
"""


def ad_without_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD.csv')
    y = AD.pop('DECLINED')
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(AD.values)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


"""
with ADNI-MEM ADNI-EF data
CN
"""


def cn_with_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN_extra_data.csv')
    y = CN.pop('DECLINED')
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(CN.values)
    rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=1)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


"""
MCI
"""


def mci_with_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI_extra_Data.csv')
    y = MCI.pop('DECLINED')
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(MCI.values)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


"""
AD
"""


def ad_with_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD_extra_data.csv')
    y = AD.pop('DECLINED')
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(AD.values)
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

    poly_score, poly_params = poly_kernel(X, y, rskf)
    print("best score:", poly_score)
    print("best params:", poly_params)
    # rbf
    best_score, best_params = rbf_kernel(X, y, rskf)
    print("best score:", best_score)
    print("best params:", best_params)


def run():
    print('#### without ADNI-MEM ADNI-EF data ####')
    cn_without_extra_data()
    mci_without_extra_data()
    ad_without_extra_data()
    print('#### with ADNI-MEM ADNI-EF data ####')
    cn_with_extra_data()
    mci_with_extra_data()
    ad_with_extra_data()


if __name__ == "__main__":
    run()
