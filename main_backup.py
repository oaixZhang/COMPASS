import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from scipy.stats import pearsonr


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


def linear_reg():
    CN = pd.read_csv('./data/CN_with_MEM_EF.csv')
    Y_CN = CN.pop('DECLINED')
    AD = pd.read_csv('./data/AD_with_MEM_EF.csv')
    Y_AD = AD.pop('DECLINED')
    MCI = pd.read_csv('./data/MCI_with_MEM_EF.csv')
    Y_MCI = MCI.pop('DECLINED')

    scaler = MinMaxScaler(feature_range=(0, 1))
    # AD
    X_AD = scaler.fit_transform(AD.values)
    X_AD_train, X_AD_test, Y_AD_train, Y_AD_test = train_test_split(X_AD, Y_AD.values)
    lr_AD = LinearRegression()
    lr_AD.fit(X_AD_train, Y_AD_train)
    predict_AD = lr_AD.predict(X_AD_test)
    print(Y_AD_test)
    print(predict_AD)
    print("accuracy on AD: ", lr_AD.score(X_AD_test, Y_AD_test))
    print("precision on AD: ", precision_score(Y_AD_test, predict_AD))
    print("Pearson score on AD: ", cal_pearson(Y_AD_test, predict_AD))

    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp = []
    for train_index, test_index in rkf.split(X_AD, Y_AD.values):
        train_X, train_y = X_AD[train_index], Y_AD.values[train_index]
        test_X, test_y = X_AD[test_index], Y_AD.values[test_index]
        lr_AD.fit(train_X, train_y)
        predict_AD = lr_AD.predict(test_X)
        temp.append(cal_pearson(test_y, predict_AD))
    print("Pearson score on AD 10 fold cross validate: ", np.array(temp).mean())


# def sigmoidNN(csvPath):
#     data = pd.read_csv(csvPath)
#     y = data.pop('DECLINED').values
#     print(y)
#
#     X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
#
#     nn = keras.Sequential([
#         keras.layers.Dense(1, input_dim=6, activation='sigmoid'),
#         keras.layers.Dropout(0.2)
#     ])
#     nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     nn.fit(X_train, y_train, epochs=5)
#     predict = nn.predict(X_test)
#     predict = predict.flatten()
#     print("y_test: ", y_test)
#     print("predict: ", predict)
#     predict = [int(a >= 0.5) for a in predict]
#     print("predict: ", predict)
#     print("Pearson score on sigmoid: ", pearsonr(y_test, predict)[0])
#
#     return nn

def my_kernel(X, Y):
    return np.dot(X, Y.T) + 100


# without ADNI-MEM ADNI-EF
def model_without_extra_data():
    CN = pd.read_csv('./data/data_CN.csv')
    Y_CN = CN.pop('DECLINED')
    AD = pd.read_csv('./data/data_AD.csv')
    Y_AD = AD.pop('DECLINED')
    MCI = pd.read_csv('./data/data_MCI.csv')
    Y_MCI = MCI.pop('DECLINED')

    # MCI
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_MCI = scaler.fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, Y_MCI_train, Y_MCI_test = train_test_split(X_MCI, Y_MCI.values)
    # svc_MCI = SVC(kernel='rbf', gamma='scale')
    svc_MCI = SVC(kernel='poly', coef0=100, degree=1, gamma=1, class_weight={0: 1, 1: 3})
    msvc_MCI = SVC(kernel=my_kernel, class_weight={0: 1, 1: 3})
    svc_MCI.fit(X_MCI_train, Y_MCI_train)
    msvc_MCI.fit(X_MCI_train, Y_MCI_train)
    predict = svc_MCI.predict(X_MCI_test)
    mpredict = msvc_MCI.predict(X_MCI_test)
    print(Y_MCI_test)
    print(predict)
    print(mpredict)
    print("accuracy on MCI: ", svc_MCI.score(X_MCI_test, Y_MCI_test))
    print("precision on MCI: ", precision_score(Y_MCI_test, predict))
    # print("Pearson score on MCI: ", (Y_MCI_test, predict)[0])
    print("Pearson score on MCI: ", cal_pearson(Y_MCI_test, predict))
    print("Pearson score on MCI: ", cal_pearson(Y_MCI_test, mpredict))
    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp_MCI = []
    mtemp_MCI = []
    for train_index, test_index in rkf.split(X_MCI, Y_MCI.values):
        train_X, train_y = X_MCI[train_index], Y_MCI.values[train_index]
        test_X, test_y = X_MCI[test_index], Y_MCI.values[test_index]
        svc_MCI.fit(train_X, train_y)
        msvc_MCI.fit(train_X, train_y)
        predict_MCI = svc_MCI.predict(test_X)
        mpredict_MCI = msvc_MCI.predict(test_X)
        temp_MCI.append(cal_pearson(test_y, predict_MCI))
        mtemp_MCI.append(cal_pearson(test_y, mpredict_MCI))
    print("Pearson score on MCI 10 fold cross validate: ", np.array(temp_MCI).mean())
    print("Pearson score on mMCI 10 fold cross validate: ", np.array(mtemp_MCI).mean())

    # CN
    X_CN = MinMaxScaler().fit_transform(CN.values)
    X_CN_train, X_CN_test, Y_CN_train, Y_CN_test = train_test_split(X_CN, Y_CN.values)
    svc_CN = SVC(kernel='poly', coef0=100, degree=1, gamma=1, class_weight={0: 1, 1: 10})
    # msvc_CN = SVC(kernel=my_kernel, class_weight={0: 1, 1: 10})
    msvc_CN = SVC(kernel=my_kernel, class_weight='balanced')
    svc_CN.fit(X_CN_train, Y_CN_train)
    msvc_CN.fit(X_CN_train, Y_CN_train)
    predict_CN = svc_CN.predict(X_CN_test)
    mpredict_CN = msvc_CN.predict(X_CN_test)
    print(Y_CN_test)
    print(predict_CN)
    print(mpredict_CN)
    print("accuracy on CN: ", svc_CN.score(X_CN_test, Y_CN_test))
    print("precision on CN: ", precision_score(Y_CN_test, predict_CN))
    # print("Pearson score on CN: ", pearsonr(Y_CN_test, predict_CN)[0])
    print("Pearson score on CN: ", cal_pearson(Y_CN_test, predict_CN))
    print("Pearson score on mCN: ", cal_pearson(Y_CN_test, mpredict_CN))
    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp_CN = []
    mtemp_CN = []
    for train_index, test_index in rkf.split(X_CN, Y_CN.values):
        train_X, train_y = X_CN[train_index], Y_CN.values[train_index]
        test_X, test_y = X_CN[test_index], Y_CN.values[test_index]
        svc_CN.fit(train_X, train_y)
        msvc_CN.fit(train_X, train_y)
        predict_CN = svc_CN.predict(test_X)
        mpredict_CN = msvc_CN.predict(test_X)
        temp_CN.append(cal_pearson(test_y, predict_CN))
        mtemp_CN.append(cal_pearson(test_y, mpredict_CN))
    print("Pearson score on CN 10 fold cross validate: ", np.array(temp_CN).mean())
    print("Pearson score on mCN 10 fold cross validate: ", np.array(mtemp_CN).mean())

    # AD
    X_AD = scaler.fit_transform(AD.values)
    X_AD_train, X_AD_test, Y_AD_train, Y_AD_test = train_test_split(X_AD, Y_AD.values)
    svc_AD = SVC(kernel='poly', coef0=100, degree=1, gamma=1, class_weight={0: 2, 1: 1})
    msvc_AD = SVC(kernel=my_kernel, class_weight={0: 2, 1: 1})
    svc_AD.fit(X_AD_train, Y_AD_train)
    msvc_AD.fit(X_AD_train, Y_AD_train)
    predict_AD = svc_AD.predict(X_AD_test)
    mpredict_AD = msvc_AD.predict(X_AD_test)
    print(Y_AD_test)
    print(predict_AD)
    print(mpredict_AD)
    print("accuracy on AD: ", svc_AD.score(X_AD_test, Y_AD_test))
    print("precision on AD: ", precision_score(Y_AD_test, predict_AD))
    # print("Pearson score on CN: ", pearsonr(Y_CN_test, predict_CN)[0])
    print("Pearson score on AD: ", cal_pearson(Y_AD_test, predict_AD))
    print("Pearson score on mAD: ", cal_pearson(Y_AD_test, mpredict_AD))

    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp = []
    mtemp = []
    for train_index, test_index in rkf.split(X_AD, Y_AD.values):
        train_X, train_y = X_AD[train_index], Y_AD.values[train_index]
        test_X, test_y = X_AD[test_index], Y_AD.values[test_index]
        svc_AD.fit(train_X, train_y)
        msvc_AD.fit(train_X, train_y)
        predict_AD = svc_AD.predict(test_X)
        mpredict_AD = msvc_AD.predict(test_X)
        temp.append(cal_pearson(test_y, predict_AD))
        mtemp.append(cal_pearson(test_y, mpredict_AD))
    print("Pearson score on AD 10 fold cross validate: ", np.array(temp).mean())
    print("Pearson score on mAD 10 fold cross validate: ", np.array(mtemp).mean())


# with ADNI-MEM ADNI-EF data
def model_with_extra_data():
    AD = pd.read_csv('./data/clf_AD_extra_data.csv')
    Y_AD = AD.pop('DECLINED')
    MCI = pd.read_csv('./data/clf_MCI_extra_data.csv')
    Y_MCI = MCI.pop('DECLINED')

    # MCI
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_MCI = scaler.fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, Y_MCI_train, Y_MCI_test = train_test_split(X_MCI, Y_MCI.values, random_state=1)
    svc_MCI = SVC(class_weight={0: 1, 1: 3}, random_state=1)
    params = {'coef0': [0, 0.1, 1, 10, 100], 'degree': [1, 2], 'kernel': ['poly', 'rbf'],
              'gamma': [0.1, 0.5, 1, 1.5, 2]}
    scoring = make_scorer(cal_pearson)
    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    grid = GridSearchCV(svc_MCI, param_grid=params, scoring=scoring, cv=rkf, return_train_score=False)
    grid = grid.fit(X_MCI_train, Y_MCI_train)
    mci = grid.best_estimator_
    print('best score:', grid.best_score_)
    print('best parameters:')
    for key in params.keys():
        print(key, mci.get_params()[key])
    predict = mci.predict(X_MCI_test)
    print(Y_MCI_test)
    print(predict)
    print("accuracy on MCI test data: ", mci.score(X_MCI_test, Y_MCI_test))
    print("precision on MCI test data: ", precision_score(Y_MCI_test, predict))
    print("Pearson score on MCI test data: ", cal_pearson(Y_MCI_test, predict))

    pd.DataFrame(grid.cv_results_).T.to_csv('./grid/grid_MCI.csv')

    # temp_MCI = []
    # # mtemp_MCI = []
    # for train_index, test_index in rkf.split(X_MCI, Y_MCI.values):
    #     train_X, train_y = X_MCI[train_index], Y_MCI.values[train_index]
    #     test_X, test_y = X_MCI[test_index], Y_MCI.values[test_index]
    #     svc_MCI.fit(train_X, train_y)
    #     # msvc_MCI.fit(train_X, train_y)
    #     predict_MCI = svc_MCI.predict(test_X)
    #     # mpredict_MCI = msvc_MCI.predict(test_X)
    #     temp_MCI.append(cal_pearson(test_y, predict_MCI))
    #     # mtemp_MCI.append(cal_pearson(test_y, mpredict_MCI))
    # print("Pearson score on MCI 10 fold cross validate: ", np.array(temp_MCI).mean())
    # # print("Pearson score on mMCI 10 fold cross validate: ", np.array(mtemp_MCI).mean())

    # CN
    CN = pd.read_csv('./data/CN_with_MEM_EF.csv')
    Y_CN = CN.pop('DECLINED')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    X_CN_train, X_CN_test, Y_CN_train, Y_CN_test = train_test_split(X_CN, Y_CN.values, random_state=1)
    svc_CN = SVC(kernel='poly', coef0=10, degree=1, gamma=1, class_weight={0: 1, 1: 10})
    msvc_CN = SVC(kernel=my_kernel, class_weight={0: 1, 1: 10})
    svc_CN.fit(X_CN_train, Y_CN_train)
    msvc_CN.fit(X_CN_train, Y_CN_train)
    predict_CN = svc_CN.predict(X_CN_test)
    mpredict_CN = msvc_CN.predict(X_CN_test)
    print(Y_CN_test)
    print(predict_CN)
    print(mpredict_CN)
    print("accuracy on CN: ", svc_CN.score(X_CN_test, Y_CN_test))
    print("precision on CN: ", precision_score(Y_CN_test, predict_CN))
    # print("Pearson score on CN: ", pearsonr(Y_CN_test, predict_CN)[0])
    print("Pearson score on CN: ", cal_pearson(Y_CN_test, predict_CN))
    print("Pearson score on mCN: ", cal_pearson(Y_CN_test, mpredict_CN))
    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp_CN = []
    mtemp_CN = []
    for train_index, test_index in rkf.split(X_CN, Y_CN.values):
        train_X, train_y = X_CN[train_index], Y_CN.values[train_index]
        test_X, test_y = X_CN[test_index], Y_CN.values[test_index]
        svc_CN.fit(train_X, train_y)
        msvc_CN.fit(train_X, train_y)
        predict_CN = svc_CN.predict(test_X)
        mpredict_CN = msvc_CN.predict(test_X)
        temp_CN.append(cal_pearson(test_y, predict_CN))
        mtemp_CN.append(cal_pearson(test_y, mpredict_CN))
    print("Pearson score on CN 10 fold cross validate: ", np.array(temp_CN).mean())
    print("Pearson score on mCN 10 fold cross validate: ", np.array(mtemp_CN).mean())

    # AD
    X_AD = scaler.fit_transform(AD.values)
    X_AD_train, X_AD_test, Y_AD_train, Y_AD_test = train_test_split(X_AD, Y_AD.values)
    svc_AD = SVC(kernel='poly', coef0=100, degree=1, gamma=1, class_weight={0: 2, 1: 1})
    msvc_AD = SVC(kernel=my_kernel, class_weight={0: 2, 1: 1})
    svc_AD.fit(X_AD_train, Y_AD_train)
    msvc_AD.fit(X_AD_train, Y_AD_train)
    predict_AD = svc_AD.predict(X_AD_test)
    mpredict_AD = msvc_AD.predict(X_AD_test)
    print(Y_AD_test)
    print(predict_AD)
    print(mpredict_AD)
    print("accuracy on AD: ", svc_AD.score(X_AD_test, Y_AD_test))
    print("precision on AD: ", precision_score(Y_AD_test, predict_AD))
    # print("Pearson score on CN: ", pearsonr(Y_CN_test, predict_CN)[0])
    print("Pearson score on AD: ", cal_pearson(Y_AD_test, predict_AD))
    print("Pearson score on mAD: ", cal_pearson(Y_AD_test, mpredict_AD))

    # 10 fold cross validate
    rkf = RepeatedKFold(n_splits=10, n_repeats=10)
    temp = []
    mtemp = []
    for train_index, test_index in rkf.split(X_AD, Y_AD.values):
        train_X, train_y = X_AD[train_index], Y_AD.values[train_index]
        test_X, test_y = X_AD[test_index], Y_AD.values[test_index]
        svc_AD.fit(train_X, train_y)
        msvc_AD.fit(train_X, train_y)
        predict_AD = svc_AD.predict(test_X)
        mpredict_AD = msvc_AD.predict(test_X)
        temp.append(cal_pearson(test_y, predict_AD))
        mtemp.append(cal_pearson(test_y, mpredict_AD))
    print("Pearson score on AD 10 fold cross validate: ", np.array(temp).mean())
    print("Pearson score on mAD 10 fold cross validate: ", np.array(mtemp).mean())


if __name__ == "__main__":
    # print('#### without ADNI-MEM ADNI-EF data ####')
    # model_without_extra_data()
    print('#### with ADNI-MEM ADNI-EF data ####')
    model_with_extra_data()

    # linear_reg()
    # calPearsonUsingLR('./data/data_CN.csv')

    # sigmoid
    # model = sigmoidNN('./data/data_CN.csv')
    # model = sigmoidNN('./data/data_AD.csv')
    # model = sigmoidNN('./data/data_MCI.csv')

    # svmOnTotalSet()
    # calPearsonUsingLR('./data/data_CN.csv')
    # calPearsonUsingLR('./data/data_AD.csv')
    # calPearsonUsingLR('./data/data_MCI.csv')
