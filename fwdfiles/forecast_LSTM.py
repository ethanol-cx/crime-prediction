import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import os
from fwdfiles.general_functions import savePredictions


def plotPrediction(df, trainPredict, testPredict, look_back, scaler):
                # shift train predictions for plotting
    trainPredictPlot = np.empty_like(df)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(
        trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(df)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back):, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


def create_X_y(data, look_back):
    X = []
    y = []
    for i in range(len(data)-look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)


def load_LSTM_model(look_back, batch_size):
    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(
        batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def forecast_LSTM(clusters, realCrimes, periodsAhead_list, gridshape, ignoreFirst, threshold, maxDist):
    np.random.seed(23)
    scaler = MinMaxScaler(feature_range=(0, 1))
    cluster_size = len(clusters.Cluster.values)
    cluster_cntr = -1
    periodsAhead_cntr = -1
    test_size = len(realCrimes) // 3
    forecasted_data = np.zeros(
        (len(periodsAhead_list), cluster_size, test_size))
    look_back = 3
    batch_size = 1
    model = load_LSTM_model(look_back, batch_size)
    test_predict_index = None
    for c in clusters.Cluster.values:
        print("Predicting cluster {} with threshold {} using LSTM".format(
            c, threshold))
        cluster_cntr += 1
        df = realCrimes['C{}_Crimes'.format(c)]
        df = scaler.fit_transform(np.array(df).reshape(-1, 1))
        X, y = create_X_y(df, look_back)

        # train test split
        y_train = y[:-test_size]
        y_test = y[-test_size:]
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        file_path = "parameters/{}/{}_parameters_grid({},{})_cluster({})_ignore({})_threshold({})_dist({}).h5".format(
            'LSTM', 'LSTM', *gridshape, c, ignoreFirst, threshold, maxDist)
        if Path(file_path).is_file():
            print("Loading existing weights ...")
            model.load_weights(file_path)
        else:
            for i in range(80):
                model.fit(X_train, y_train, epochs=1,
                          batch_size=batch_size, verbose=2, shuffle=False)
                model.reset_states()
            # save weights
            os.makedirs(os.path.abspath("parameters/"), exist_ok=True)
            os.makedirs(os.path.abspath(
                "parameters/{}".format('LSTM')), exist_ok=True)
            model.save_weights(file_path)
        trainPredict = model.predict(X_train, batch_size=batch_size)

        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        y_train = scaler.inverse_transform(y_train)
        y_test = scaler.inverse_transform(y_test)

        # for each predict horizon - `periodsAhead`, we perform recursive multi-step timeseries prediction with different timesteps ahead
        # Note: the the `start` and the `end` defines the window and splits the observation and the "y_test"
        for periodsAhead in periodsAhead_list:
            print('LSTM, threshold: {}, cluster: {}, periodsAhead: {}'.format(
                threshold, c, periodsAhead))
            periodsAhead_cntr += 1
            testPredict = np.zeros(test_size)

            # start from the first row of the features
            for i in range(test_size):
                X_test_i = X_test[i].reshape(1, -1, 1)
                for _ in range(periodsAhead):
                    pred = model.predict(X_test_i, batch_size=1)
                    X_test_i = np.append(
                        X_test_i[:, 1:, :][0], pred[-1]).reshape(1, -1, 1)
                testPredict[i] = pred[-1]  # it contains one data: `pred_i`

            testPredict = scaler.inverse_transform(testPredict.reshape(-1, 1))

            # Could insert plotPrediction() here to see the prediction results while running

            # flatten the testPredict to store
            testPredict = testPredict.flatten()

            # apply the constraint that all predictions should be non-negative
            testPredict = [x if x >= 0 else 0 for x in testPredict]

            # store the prediction to the corresponding column `periodsAhead_cntr` and `cluster_cntr`
            forecasted_data[periodsAhead_cntr][cluster_cntr] = testPredict

        # reset the periodsAhead_cntr
        periodsAhead_cntr = -1

    # store the prediction
    for i in range(len(periodsAhead_list)):
        periodsAhead = periodsAhead_list[i]
        forecasts = pd.DataFrame(data=forecasted_data[i].T, columns=['C{}_Forecast'.format(c)
                                                                     for c in clusters.Cluster.values])
        savePredictions(clusters, realCrimes, forecasts, 'LSTM',
                        gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
