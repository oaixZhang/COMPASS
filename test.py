import sys

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import mlogger
import random
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score


# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
#
# x_train, x_test, y_train, y_test = ts(x, y, test_size=0.3)
#
# clf = svm.SVC()
# clf.fit(x_train,y_train)
# score_rbf = clf.score(x_test,y_test)
# print("The score of rbf is : %f"%score_rbf)

# x, y = datasets.make_moons()
# print(x.shape)
# print(y)
# plt.scatter(x[y==0,0],x[y==0,1])
# plt.scatter(x[y==1,0],x[y==1,1])
# plt.show()

# svc=svm.SVC()
# svc.fit(x,y)
# score = svc.score(x,y)
# print("score: ",score)
# np.cov()

# x, y = datasets.load_digits(return_X_y=True)
#
# digits = datasets.load_digits()
# print(x.shape)
# print(digits.data.shape)
# plt.imshow(digits.images[0])  # doctest: +SKIP
# print(x[0])
# print(digits.images[0])
# plt.show()  # doctest: +SKIP

def testGCForest():
    # data = pd.read_csv('./data_genetic/data_all_features.csv')
    # MCI = data[data.DX_bl == 2].copy()
    # X = MCI.drop(columns=['RID', 'DX_bl', 'TOMM40_A1', 'TOMM40_A2', 'ADNI_MEM', 'ADNI_EF', 'DECLINED'])
    X = pd.read_csv('./data/clf_MCI.csv')
    y = X.pop('DECLINED').values
    X = X.values
    skf = StratifiedKFold(10, random_state=0)
    g = []
    s = []
    for traini, testi in skf.split(X, y):
        X_train, X_test, Y_train, Y_test = X[traini], X[testi], y[traini], y[testi]
        print(X_test.tolist())
        # gcf = gcForest(tolerance=0.0, min_samples_cascade=20)
        gcf.cascade_forest(X_train, Y_train)
        pred_proba = gcf.cascade_forest(X_test)
        print(np.array(pred_proba).shape)
        print(pred_proba)
        svc = SVC(C=100, kernel='poly', degree=2, gamma=0.01, class_weight={0: 1, 1: 2}, coef0=10)
        svc.fit(X_train, Y_train)
        s.append(svc.score(X_test, Y_test))
        # print(np.array(g).ravel())
        # print(np.array(s).mean())


if __name__ == '__main__':
    # data = [
    #     [11.53, 11.69, 11.70, 11.51, 871365.0, 1],
    #     [11.64, 11.63, 11.72, 11.57, 722764.0, 1],
    #     [11.59, 11.48, 11.59, 11.41, 461808.0, 1],
    #     [11.39, 11.19, 11.40, 11.15, 1074465.0, 1]]
    # df = DataFrame(data, index=["2017-10-18", "2017-10-19", "2017-10-20", "2017-10-23"],
    #                columns=["open", "close", "high", "low", "volume", "code"])
    # print(df)
    # print(df.values)
    # str = '2,3,2,3,2'
    # a = [1, 2, 2]
    # print(str.count('2'))
    # print(Counter(str))
    # sys.stdout = mlogger.Logger('./grid/test.grid')

    # print('grid test')
    # a = np.array([1, 2, 3])
    # print(np.square(a))
    #
    # rkf = RepeatedKFold(n_splits=10, n_repeats=2)
    # x = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).reshape(10, 2)
    # y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    # for train, test in rkf.split(x):
    #     print('train:', train, "test:", test)

    # rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=1, random_state=0)
    # for train_index, test_index in rskf.split([0, 1, 2, 3]):
    #     print('train', train_index)
    #     print('test', test_index)
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b=a[0]*1+a[1]*2
    # print(b)

    # random.seed(0)
    # for i in range(5):
    #     print(random.randint(1, 100))
    # testGCForest()

    # print(np.array(a).shape)
    # print(np.mean(a, axis=0))
    # print(np.mean(a))

    # swap_pred = np.swapaxes(a, 0, 1)
    # print(swap_pred.shape)
    # print(swap_pred)
    # add_feat = swap_pred.reshape([41, -1])
    # print(add_feat.shape)
    # print(add_feat)
    # feat_arr = np.concatenate([add_feat, xtest], axis=1)
    # print(feat_arr)

    # a = np.random.rand(3, 5)
    # print(a.shape)
    # print(a)
    # s = np.hstack([a] + [[[1], [1], [1]]])
    # print(s)
    # b = np.random.rand(3, 1)
    # print(b)
    # c = np.hstack((a, b))
    # print(c)

    # a = np.array([[1], [2], [3]])
    a = [1, 2, 3]
    b = np.array([[[1,1], [2,2], [3,3]],[[1,1], [2,2], [3,3]]])
    print(np.hstack(b))
