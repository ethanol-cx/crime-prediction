import pandas as pd
import numpy as np
from pyramid.arima import auto_arima
import sklearn.model_selection as ms
import statsmodels.api as sm


# following lines compute the forecast, called from ScriptForecast
def init_predictions_Harmonic():
    print("Starting Harmonic")
# Compute predictions


def predictions_Harmonic(clusters, realCrimes, periodsAhead=52, orders=[(0, 0)], seasonal_orders=[(0, 0)]):
    cluster_size = len(clusters.Cluster.values)
    test_size = len(realCrimes['C1_Crimes']) // 3
    forecast_data = np.zeros(
        (cluster_size, test_size))
    columnCntr = 0
    for c in clusters.Cluster.values:
        df = realCrimes['C{}_Crimes'.format(c)]
        # all_zero_flag = 1
        # for x in df:
        #     if x != 0:
        #         all_zero_flag = 0
        #         break
        # if all_zero_flag == 1:
        #     forecast_data[columnCntr] = np.zeros(test_size)
        #     columnCntr += 1
        #     continue
        train = df[:-test_size]
        test = df[-test_size:]
        # if (periodsAhead == 1) or len(orders) == 1:
        #     stepwise_model = auto_arima(train, start_p=0, start_q=0,
        #                                 max_q=5, max_p=0, m=52,
        #                                 start_P=0, max_P=0, start_Q=0, max_Q=5, seasonal=True,
        #                                 trace=True,
        #                                 error_action='ignore',
        #                                 suppress_warnings=True,
        #                                 stepwise=True, disp=0)
        #     print(stepwise_model.order)
        #     print(stepwise_model.seasonal_order)
        #     orders.append(stepwise_model.order)
        #     seasonal_orders.append(stepwise_model.seasonal_order)
        # pred_model = sm.tsa.statespace.SARIMAX(
        #     df, order=orders[columnCntr+1], seasonal_order=seasonal_orders[columnCntr+1], enforce_stationarity=False, enforce_invertibility=False)
        pred_model = sm.tsa.statespace.SARIMAX(
            df, order=(3, 1, 1), seasonal_order=(0, 0, 0, 52), enforce_stationarity=False, enforce_invertibility=False)
        coef_results = pred_model.fit(disp=0)
        # The test set starts at 2016-01-03
        predictions = np.zeros(test_size)
        for i in range(0, test_size - 1):
            temp_pred = coef_results.get_prediction(
                start=i + len(train) - periodsAhead, end=i+len(train), dynamic=0)
            t_predictions = temp_pred.predicted_mean
            predictions[i] = t_predictions[len(t_predictions) - 1]
        predictions = [x if x > 0 else 0 for x in predictions]
        forecast_data[columnCntr] = np.array(predictions)

        columnCntr += 1
    forecasts = pd.DataFrame(forecast_data.T, columns=['C{}_Forecast'.format(c)
                                                       for c in clusters.Cluster.values])
    df = realCrimes['C1_Crimes']
    forecasts.index = df[-test_size:].index
    return forecasts, orders, seasonal_orders
