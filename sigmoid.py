import pandas as pd

import keras
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def sigmoidNN(csvPath):
    data = pd.read_csv(csvPath)
    y = data.pop('DECLINED').values
    print(y)

    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(data.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    nn = keras.Sequential([
        keras.layers.Dense(1, input_dim=6, activation='sigmoid'),
        keras.layers.Dropout(0.2)
    ])
    nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn.fit(X_train, y_train, epochs=5)
    predict = nn.predict(X_test)
    predict = predict.flatten()
    print("y_test: ", y_test)
    print("predict: ", predict)
    predict = [int(a >= 0.5) for a in predict]
    print("predict: ", predict)
    print("Pearson score on sigmoid: ", pearsonr(y_test, predict)[0])

    return nn
