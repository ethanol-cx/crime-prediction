import pandas as pd
import numpy as np
from pyramid.arima import auto_arima
import sklearn.model_selection as ms
import statsmodels.api as sm
from fwdfiles.general_functions import savePredictions, saveParameters, getIfParametersExists
# Compute predictions using Seasonal Moving Average Model


def forecast_ARIMA(method, clusters, realCrimes, periodsAhead_list, gridshape, ignoreFirst, threshold, maxDist, orders=[], seasonal_orders=[]):
    assert method in ['MA', 'AR', 'ARIMA']
    print("Starting Predictions_{}".format(method))
    cluster_size = len(clusters.Cluster.values)
    cluster_cntr = -1
    periodsAhead_cntr = -1
    test_size = len(realCrimes['C1_Crimes']) // 3
    forecasted_data = np.zeros(
        (len(periodsAhead_list), cluster_size, test_size))
    for c in clusters.Cluster.values:
        print("Predicting cluster {} with threshold {} using {}".format(
            c, threshold, method))
        cluster_cntr += 1
        df = realCrimes['C{}_Crimes'.format(c)]
        # train test split
        train = df[:-test_size]
        if train.sum() < 2:
            continue

        test = df[-test_size:]
        pred_model = None

        # if the parameters exist
        params = getIfParametersExists(
            method, gridshape, c, ignoreFirst, threshold, maxDist)
        if params:
            print("Loading existing parameters for ...")
            print(params)
            pred_model = sm.tsa.statespace.SARIMAX(endog=df, order=params[0], seasonal_order=params[1],
                                                   enforce_stationarity=False, enforce_invertibility=False, hamilton_representation=False)
        else:
            # apply appropriate ranges for the grid search based on the `method`
            p_max = 0
            q_max = 0
            P_max = 0
            Q_max = 0
            p_start = 0
            q_start = 0
            P_start = 0
            Q_start = 0
            D_max = 1
            d_max = 2
            if method == 'MA' or method == 'ARIMA':
                q_start = 0
                Q_start = 0
                q_max = 5
                Q_max = 1
            if method == 'AR' or method == 'ARIMA':
                p_start = 0
                P_start = 0
                p_max = 5
                P_max = 1

            # get the optimal combination of the hyperparameters through stepwise search given that we had only the training data
            stepwise_model = auto_arima(train, start_p=p_start, max_p=p_max, start_q=q_start, max_q=q_max, m=52, start_P=P_start, max_P=P_max,
                                        start_Q=Q_start, max_Q=Q_max, seasonal=True, trace=True, error_action='ignore', suppress_warnings=True,
                                        max_d=d_max, max_D=D_max, disp=0, max_order=10, maxiter=50)

            saveParameters(stepwise_model.order, stepwise_model.seasonal_order,
                           method, gridshape, c, ignoreFirst, threshold, maxDist)

            if stepwise_model.seasonal_order:
                pred_model = sm.tsa.statespace.SARIMAX(
                    endog=df, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order, enforce_stationarity=False, enforce_invertibility=False, hamilton_representation=False)
            else:
                pred_model = sm.tsa.statespace.SARIMAX(
                    endog=df, order=stepwise_model.order, enforce_stationarity=False, enforce_invertibility=False, hamilton_representation=False)
        coef_results = pred_model.fit(disp=0)

        # for each predict horizon - `periodsAhead`, we perform rolling time series prediction with different window sizes
        # Note: the the `start` and the `end` defines the window and splits the observation and the "y_test"
        for periodsAhead in periodsAhead_list:
            print(method, threshold, c, periodsAhead)
            periodsAhead_cntr += 1
            predictions = np.zeros(test_size)
            for i in range(test_size):
                pred = coef_results.get_prediction(
                    start=i+len(train)-periodsAhead, end=i+len(train), dynamic=True)
                predictions[i] = pred.predicted_mean.values[-1]
                # history.append(pd.Series(test[i]), ignore_index=True)

            # apply the assumption that all predictions should be non-negative
            predictions = [x if x >= 0 else 0 for x in predictions]

            # store the prediction to the corresponding column `periodsAhead_cntr` and `cluster_cntr`
            forecasted_data[periodsAhead_cntr][cluster_cntr] = predictions

        # reset the periodsAhead_cntr
        periodsAhead_cntr = -1

    # store the prediction
    for i in range(len(periodsAhead_list)):
        periodsAhead = periodsAhead_list[i]
        forecasts = pd.DataFrame(data=forecasted_data[i].T, columns=['C{}_Forecast'.format(c)
                                                                     for c in clusters.Cluster.values])
        forecasts.index = df[-test_size:].index
        savePredictions(clusters, realCrimes, forecasts, method,
                        gridshape, ignoreFirst, periodsAhead, threshold, maxDist)
