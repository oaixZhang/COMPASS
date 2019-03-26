import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from svm import cal_pearson
from sklearn.base import BaseEstimator


class CascadeForest(BaseEstimator):
    def __init__(self, cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, max_cascade_layer=np.inf,
                 min_samples_cascade=0.05, tolerance=0.0, n_jobs=1):
        setattr(self, 'n_layer', 0)
        setattr(self, '_n_samples', 0)
        setattr(self, 'n_cascadeRF', int(n_cascadeRF))
        setattr(self, 'cascade_test_size', cascade_test_size)
        setattr(self, 'n_cascadeRFtree', int(n_cascadeRFtree))
        setattr(self, 'max_cascade_layer', max_cascade_layer)
        setattr(self, 'min_samples_cascade', min_samples_cascade)
        setattr(self, 'tolerance', tolerance)
        setattr(self, 'n_jobs', n_jobs)

    def fit(self, X, y=None):
        setattr(self, 'n_layer', 0)
        test_size = getattr(self, 'cascade_test_size')
        max_layers = getattr(self, 'max_cascade_layer')
        tol = getattr(self, 'tolerance')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        self.n_layer += 1
        pred = self._cascade_layer(X_train, y_train)
        score = self._cascade_evaluation(X_test, y_test)
        concated_feature = self._concat(X_train, pred)

        self.n_layer += 1
        layered_pred = self._cascade_layer(concated_feature, y_train)
        layered_score = self._cascade_evaluation(X_test, y_test)

        while layered_score > (score + tol) and self.n_layer <= max_layers:
            score = layered_score
            pred = layered_pred
            concated_feature = self._concat(X_train, pred)
            self.n_layer += 1
            layered_pred = self._cascade_layer(concated_feature, y_train)
            layered_score = self._cascade_evaluation(X_test, y_test)

        if layered_score < score:
            n_cascadeRF = getattr(self, 'n_cascadeRF')
            for irf in range(n_cascadeRF):
                delattr(self, '_casrf{}_{}'.format(self.n_layer, irf))
                delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
            self.n_layer -= 1

        return self

    def _predict(self, X):
        layer = 1
        pred = self._cascade_layer(X, layer=layer)
        while layer < getattr(self, 'n_layer'):
            layer += 1
            concated_feature = self._concat(X, pred)
            pred = self._cascade_layer(concated_feature, layer=layer)
        return pred

    def predict(self, X):
        layer = 1
        pred = self._cascade_layer(X, layer=layer)
        while layer < getattr(self, 'n_layer'):
            layer += 1
            concated_feature = self._concat(X, pred)
            pred = self._cascade_layer(concated_feature, layer=layer)
        pred = np.mean(pred, axis=0)
        return pred

    def _cascade_layer(self, X, y=None, layer=0):
        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_samples_cascade')

        n_jobs = getattr(self, 'n_jobs')
        rf = RandomForestRegressor(n_estimators=n_tree, max_features='sqrt',
                                   min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
        crf = RandomForestRegressor(n_estimators=n_tree, max_features=1,
                                    min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
        lr = LinearRegression()
        svr = SVR()
        pred = []
        if y is not None:
            # print('Adding Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                rf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casrf{}_{}'.format(self.n_layer, irf), rf)
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), crf)
                pred.append(rf.oob_prediction_)
                pred.append(crf.oob_prediction_)
        elif y is None:
            for irf in range(n_cascadeRF):
                rf = getattr(self, '_casrf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                pred.append(rf.predict(X))
                pred.append(crf.predict(X))

        return pred

    def _cascade_evaluation(self, X_test, y_test):
        pred = np.mean(self._predict(X_test), axis=0)
        pearson_score = cal_pearson(y_test, pred)
        # print('Layer validation pearson socre = {}'.format(pearson_score))

        return pearson_score

    def _concat(self, X, pred):
        swap_pred = np.swapaxes(pred, 0, 1)
        reshaped = swap_pred.reshape([np.shape(X)[0], -1])
        concated = np.concatenate([reshaped, X], axis=1)

        return concated
