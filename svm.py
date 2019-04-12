import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def calculate_auroc(ytrue, probas):
    fpr, tpr, thresholds = roc_curve(ytrue, probas)
    roc_auc = auc(fpr, tpr)
    return roc_auc


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
