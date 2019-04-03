from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy import interp
import dataprocess


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


# without ADNI-MEM ADNI-EF
# CN
"""
best parameters: 
gamma 1
coef0 100
degree 2
class_weight {0: 1, 1: 10}
kernel poly
accuracy on CN test data:  0.7959183673469388
precision on CN test data:  0.0
Pearson score on CN test data:  -0.1113692092779409  
"""


def cn_without_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN.csv')
    y_CN = CN.pop('DECLINED')
    X_CN = MinMaxScaler().fit_transform(CN.values)
    X_CN_train, X_CN_test, y_CN_train, y_CN_test = train_test_split(X_CN, y_CN.values, test_size=0.25, random_state=92)

    print("test set:", Counter(y_CN_test))

    svc = SVC(C=1, kernel='poly', degree=2, gamma=1, coef0=100, class_weight={0: 1, 1: 10})
    svc.fit(X_CN_train, y_CN_train)
    predict_CN = svc.predict(X_CN_test)

    print(y_CN_test)
    print(predict_CN)
    print("accuracy on CN test data: ", svc.score(X_CN_test, y_CN_test))
    print("precision on CN test data: ", precision_score(y_CN_test, predict_CN))
    print("Pearson score on CN test data: ", cal_pearson(y_CN_test, predict_CN), '\n')


#  MCI
"""
best parameters:
gamma 1.5
random_state 1
coef0 1
degree 2
class_weight balanced
kernel poly
accuracy on MCI test data:  0.5802469135802469
precision on MCI test data:  0.46511627906976744
Pearson score on MCI test data:  0.18033915132256592 
"""


def mci_without_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI.csv')
    y_MCI = MCI.pop('DECLINED')
    X_MCI = MinMaxScaler(feature_range=(0, 1)).fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, y_MCI_train, y_MCI_test = train_test_split(X_MCI, y_MCI.values, test_size=0.25,
                                                                        random_state=6)
    print("test set:", Counter(y_MCI_test))

    svc = SVC(kernel='rbf', C=1, class_weight='balanced', gamma=1)
    svc.fit(X_MCI_train, y_MCI_train)
    predict = svc.predict(X_MCI_test)

    print(y_MCI_test)
    print(predict)
    print("accuracy on MCI test data: ", svc.score(X_MCI_test, y_MCI_test))
    print("precision on MCI test data: ", precision_score(y_MCI_test, predict))
    print("Pearson score on MCI test data: ", cal_pearson(y_MCI_test, predict), '\n')


# AD
"""
best parameters:
degree 2
gamma 0.1
coef0 100
kernel poly
class_weight balanced
random_state 1
accuracy on AD test data:  0.56
precision on AD test data:  0.7777777777777778
Pearson score on AD test data:  0.21527777777777782
"""


def ad_without_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD.csv')
    y_AD = AD.pop('DECLINED')
    X_AD = MinMaxScaler(feature_range=(0, 1)).fit_transform(AD.values)
    X_AD_train, X_AD_test, y_AD_train, y_AD_test = train_test_split(X_AD, y_AD.values, test_size=0.25, random_state=4)
    print("test set:", Counter(y_AD_test))

    svc = SVC(C=1, coef0=100, kernel='poly', degree=2, gamma=0.1, class_weight={0: 2, 1: 1})
    svc.fit(X_AD_train, y_AD_train)
    predict_AD = svc.predict(X_AD_test)

    print(y_AD_test)
    print(predict_AD)
    print("accuracy on AD test data: ", svc.score(X_AD_test, y_AD_test))
    print("precision on AD test data: ", precision_score(y_AD_test, predict_AD))
    print("Pearson score on AD test data: ", cal_pearson(y_AD_test, predict_AD), '\n')


# with ADNI-MEM ADNI-EF data
# CN
"""
best parameters: 
gamma 2
random_state 1
coef0 0
degree 1
class_weight balanced
kernel rbf
accuracy on CN test data:  0.7755102040816326
precision on CN test data:  0.0
Pearson score on CN test data:  -0.12592155012732867 
"""


def cn_with_extra_data():
    print('*** CN ***')
    CN = pd.read_csv('./data/clf_CN_extra_data.csv')
    y_CN = CN.pop('DECLINED')
    X_CN = MinMaxScaler(feature_range=(0, 1)).fit_transform(CN.values)
    X_CN_train, X_CN_test, y_CN_train, y_CN_test = train_test_split(X_CN, y_CN.values, test_size=0.25,
                                                                    random_state=76)
    print("test set:", Counter(y_CN_test))
    # oversampling

    svc = SVC(C=1, kernel='poly', degree=2, gamma=1, coef0=1, class_weight={0: 1, 1: 8})
    svc.fit(X_CN_train, y_CN_train)
    predict_CN = svc.predict(X_CN_test)

    print(y_CN_test)
    print(predict_CN)
    print("accuracy on CN test data: ", svc.score(X_CN_test, y_CN_test))
    print("precision on CN test data: ", precision_score(y_CN_test, predict_CN))
    print("Pearson score on CN test data: ", cal_pearson(y_CN_test, predict_CN), '\n')


# MCI
"""
best parameters:
gamma 0.1
random_state 1
coef0 100
degree 2
class_weight {0: 1, 1: 2}
kernel poly
accuracy on MCI test data:  0.7283950617283951
precision on MCI test data:  0.6666666666666666
Pearson score on MCI test data:  0.41309219466158226 
"""


def mci_with_extra_data():
    print('*** MCI ***')
    MCI = pd.read_csv('./data/clf_MCI_extra_Data.csv')
    y_MCI = MCI.pop('DECLINED')
    X_MCI = MinMaxScaler(feature_range=(0, 1)).fit_transform(MCI.values)
    X_MCI_train, X_MCI_test, y_MCI_train, y_MCI_test = train_test_split(X_MCI, y_MCI.values, test_size=0.25,
                                                                        random_state=6)
    print("test set:", Counter(y_MCI_test))

    svc = SVC(kernel='poly', C=1, class_weight={0: 1, 1: 2}, gamma=0.1, degree=2, coef0=100)
    svc.fit(X_MCI_train, y_MCI_train)
    predict = svc.predict(X_MCI_test)

    print(y_MCI_test)
    print(predict)
    print("accuracy on MCI test data: ", svc.score(X_MCI_test, y_MCI_test))
    print("precision on MCI test data: ", precision_score(y_MCI_test, predict))
    print("Pearson score on MCI test data: ", cal_pearson(y_MCI_test, predict), '\n')


# AD
"""
best parameters:
gamma 0.1
coef0 100
degree 2
class_weight {0: 2, 1: 1}
kernel poly

random=5:
accuracy on AD test data:  0.88
precision on AD test data:  1.0
Pearson score on AD test data:  0.773905989951895 
"""


def ad_with_extra_data():
    print('*** AD ***')
    AD = pd.read_csv('./data/clf_AD_extra_data.csv')
    y_AD = AD.pop('DECLINED')
    X_AD = MinMaxScaler(feature_range=(0, 1)).fit_transform(AD.values)
    X_AD_train, X_AD_test, y_AD_train, y_AD_test = train_test_split(X_AD, y_AD.values, test_size=0.25, random_state=5)
    print("test set:", Counter(y_AD_test))

    svc = SVC(C=1, coef0=100, kernel='poly', degree=2, gamma=0.1, class_weight={0: 2, 1: 1})
    svc.fit(X_AD_train, y_AD_train)
    predict_AD = svc.predict(X_AD_test)

    print(y_AD_test)
    print(predict_AD)
    print("accuracy on AD test data: ", svc.score(X_AD_test, y_AD_test))
    print("precision on AD test data: ", precision_score(y_AD_test, predict_AD))
    print("Pearson score on AD test data: ", cal_pearson(y_AD_test, predict_AD), '\n')


def roc(svc, file, cv):
    data = pd.read_csv('./data/%s.csv' % file)
    y = data.pop('DECLINED').values
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    svc = svc
    i = 0
    for train, test in cv.split(X, y):
        probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_auc, lw=2, alpha=.8)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(file)
    plt.legend(loc="lower right")
    plt.show()


def roccurve():
    files = ['clf_CN', 'clf_MCI', 'clf_AD', 'clf_CN_extra_data', 'clf_MCI_extra_data', 'clf_AD_extra_Data']
    classifiers = [
        SVC(C=0.1, kernel='poly', degree=1, gamma=1, coef0=100, class_weight={0: 1, 1: 12}, probability=True),
        SVC(kernel='poly', C=0.1, class_weight={0: 1, 1: 1}, gamma=1, coef0=100, probability=True),
        SVC(C=0.1, coef0=100, kernel='poly', degree=2, gamma=1, class_weight={0: 1, 1: 1}, probability=True),
        SVC(C=1, kernel='poly', degree=2, gamma=1, coef0=1, class_weight={0: 1, 1: 8}, probability=True),
        SVC(kernel='poly', C=1, class_weight={0: 1, 1: 2}, gamma=0.1, degree=2, coef0=100, probability=True),
        SVC(C=1, coef0=100, kernel='poly', degree=2, gamma=0.1, class_weight={0: 2, 1: 1}, probability=True)]
    index = 0
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        svc = classifiers[index]
        if index % 3 == 0:
            cv = StratifiedKFold(n_splits=4, random_state=0)
        else:
            cv = StratifiedKFold(n_splits=10, random_state=0)
        plt.subplot(2, 3, index + 1)
        # j = 0
        for train, test in cv.split(X, y):
            probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3)
            # j += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        # std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.2f)' % mean_auc, lw=2, alpha=.8)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(file)
        plt.legend(loc='lower right', fontsize=5)
        index += 1
    plt.show()


def roc_whole():
    files = ['clf_CN', 'clf_MCI', 'clf_AD', 'clf_CN_extra_data', 'clf_MCI_extra_data', 'clf_AD_extra_Data']
    classifiers = [SVC(C=0.1, kernel='poly', degree=2, gamma=10, coef0=1, class_weight={0: 1, 1: 8}, probability=True),
                   SVC(kernel='rbf', C=1, class_weight='balanced', gamma=1, probability=True),
                   SVC(C=1, coef0=100, kernel='poly', degree=2, gamma=0.1, class_weight={0: 2, 1: 1}, probability=True),
                   SVC(C=1000, kernel='poly', degree=1, gamma=0.01, coef0=100, class_weight={0: 1, 1: 8},
                       probability=True),
                   SVC(kernel='poly', C=1, class_weight={0: 1, 1: 2}, gamma=0.1, degree=2, coef0=100, probability=True),
                   SVC(C=1, coef0=100, kernel='poly', degree=2, gamma=0.1, class_weight={0: 2, 1: 1}, probability=True)]
    titles = ['CN_Without_ADNI_feature', 'MCI_Without_ADNI_feature', 'AD_Without_ADNI_feature', 'CN_With_ADNI_feature',
              'MCI_With_ADNI_feature',
              'AD_With_ADNI_feature']
    index = 0
    mean_tprs = []
    for file in files:
        data = pd.read_csv('./data_genetic/%s.csv' % file)
        y = data.pop('DECLINED').values
        X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        svc = classifiers[index]
        if index % 3 == 0:
            cv = StratifiedKFold(n_splits=4, random_state=0)
        else:
            cv = StratifiedKFold(n_splits=10, random_state=0)
        plt.subplot(2, 3, index + 1)
        for train, test in cv.split(X, y):
            probas_ = svc.fit(X[train], y[train]).predict_proba(X[test])
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.2f)' % mean_auc, lw=2, alpha=.8)
        mean_tprs.append(mean_tpr)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        plt.title(titles[index], fontsize=6)
        plt.legend(loc='lower right', fontsize=6)
        index += 1

    plt.show()

    mean_tprs = np.array(mean_tprs)
    weighted_tpr = mean_tprs[0] * 241 / 767 + mean_tprs[1] * 403 / 767 + mean_tprs[2] * 123 / 767
    weighted_tpr_with_extra = mean_tprs[3] * 241 / 767 + mean_tprs[4] * 403 / 767 + mean_tprs[5] * 123 / 767
    weighted_auc = auc(mean_fpr, weighted_tpr)
    plt.plot(mean_fpr, weighted_tpr, color='b', label=r'Without ADNI features (AUC=%0.2f)' % weighted_auc, lw=2,
             alpha=.8)
    weighted_auc_with_extra = auc(mean_fpr, weighted_tpr_with_extra)
    plt.plot(mean_fpr, weighted_tpr_with_extra, color='r',
             label=r'With ADNI features (AUC=%0.2f)' % weighted_auc_with_extra,
             lw=2, alpha=.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, alpha=.5)
    plt.title('ROC Curve')
    plt.legend(loc='lower right', fontsize=12)
    plt.show()


if __name__ == "__main__":
    # print('#### without ADNI-MEM ADNI-EF data ####')
    # cn_without_extra_data()
    # mci_without_extra_data()
    # ad_without_extra_data()
    # print('#### with ADNI-MEM ADNI-EF data ####')
    # cn_with_extra_data()
    # mci_with_extra_data()
    # ad_with_extra_data()
    # roc(SVC(C=0.1, kernel='poly', degree=3, gamma=1, coef0=0.1, class_weight={0: 1, 1: 8}, probability=True),
    #     'clf_CN', StratifiedKFold(n_splits=4, random_state=0))
    # roc(SVC(C=0.1, kernel='poly', degree=2, gamma=10, coef0=0.1, class_weight={0: 1, 1: 8}, probability=True),
    #     'clf_MCI', StratifiedKFold(n_splits=4, random_state=0))
    # roc(SVC(C=0.1, kernel='poly', degree=2, gamma=10, coef0=0.1, class_weight={0: 1, 1: 8}, probability=True),
    #     'clf_AD', StratifiedKFold(n_splits=4, random_state=0))
    # roccurve()
    roc_whole()
