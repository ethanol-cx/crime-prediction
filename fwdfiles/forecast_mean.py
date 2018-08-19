import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from pyramid.arima import auto_arima
import sklearn.model_selection as ms
import statsmodels.api as sm

# def fMM(series, h=5, window=3):
#     # add h future periods
#     indices = [series.index[-1] +
#                np.timedelta64(7, 'D')*t for t in range(-window+1, h+1)]
#     forecasts = pd.Series(data=None, index=indices, name='Crimes')
#     forecasts.update(series)

#     # compute forecasts recursively
#     for t in range(forecasts.shape[0]-h, forecasts.shape[0]):
#         forecasts.iloc[t] = forecasts.iloc[t-window:t].mean()
#     return forecasts[-h:]

# # perform cross validation for one cluster


# def tsCV(ts, periodsAhead=5, window=3):
#     forecasts = pd.Series(data=None, index=ts.index, name='Crimes')

#     for t in range(periodsAhead+window-1, ts.shape[0]):
#         forecasts.iloc[t] = fMM(ts.iloc[t-window-periodsAhead+1: t -
#                                         periodsAhead+1], h=periodsAhead, window=window).iloc[-1]
#     return forecasts


# # compute all clusters of realCrimes
# window_global = None


# def sequentialComputeForecasts(realCrimes, periodsAhead=1):
#     ret = list()
#     for c in realCrimes.columns:
#         ret.append(
#             tsCV(realCrimes[c], periodsAhead=periodsAhead, window=window_global))
#     return ret


# following lines compute the forecast, called from ScriptForecast
def init_predictions_Mean():
    print("Starting Predictions_MM")
#     pandas2ri.activate()
#     robjects.r('''
#         library(forecast)
#         library(foreach)
#         library(parallel)
#         library(doParallel)
#         library(R.utils)
#         library(zoo)

#         fAR <- function(y, h) {
#             model <- Arima(y, order=c(0,3,0))
#             fc <- forecast(model, h=h)
#         }

#         computeClusterErrors <- function(weeks, initialYear, initialWeek, periodsAhead, forecastfunction) {
#             ts <- msts(weeks, start=c(initialYear, initialWeek), seasonal.periods = c(4,52))
#             tsCV(ts, forecastfunction, h=periodsAhead)
#         }

#         parallelComputeCVErrors <- function(realCrimes, initialYear, initialWeek, periodsAhead=1) {

#             cl <- makeCluster( min(detectCores()-1, length(realCrimes)) )
#             registerDoParallel(cl)
#             ret <- foreach(cluster=1:length(realCrimes),
#                 .combine='cbind',
#                 .packages=c('forecast', 'R.utils'),
#                 .export=c('computeClusterErrors', 'fAR')) %dopar% {
#                 cat(paste('computing cluster', cluster,'...\n'))
#                 computeClusterErrors(realCrimes[cluster], initialYear, initialWeek, periodsAhead, fAR)
#             }
#             stopCluster(cl)
#             ret
#         }
#     ''')

# Compute predictions


def predictions_Mean(clusters, realCrimes, periodsAhead=52, orders=[(0,0)], seasonal_orders=[(0,0)]):
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
        if (periodsAhead == 1) or len(orders) == 1:
            stepwise_model = auto_arima(train, start_p=0, start_q=0,
                                        max_q=5, max_p=0, m=52,
                                        start_P=0, max_P=0, start_Q=0, max_Q=5, seasonal=True,
                                        trace=True,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True, disp=0)
            print(stepwise_model.order)
            print(stepwise_model.seasonal_order)
            if stepwise_model.order == None:
                orders.append((0,0,0))
            else:
                orders.append(stepwise_model.order)
            if stepwise_model.seasonal_order == None:
                seasonal_orders.append((0,0,0,52))
            else:
                seasonal_orders.append(stepwise_model.seasonal_order)
        pred_model = sm.tsa.statespace.SARIMAX(
            df, order=orders[columnCntr+1], seasonal_order=seasonal_orders[columnCntr+1], enforce_stationarity=False, enforce_invertibility=False)

        # print("data frame")
        # print(df)
        coef_results = pred_model.fit(disp=0)
        # The test set starts at 2016-01-03
        # print(test.index)
        predictions = np.zeros(test_size)
        for i in range(0, test_size - 1):
            temp_pred = coef_results.get_prediction(
                start=pd.to_datetime(df.index[i+ len(train) - periodsAhead]), end=pd.to_datetime(test.index[i]), dynamic=0)
            t_predictions = temp_pred.predicted_mean
            predictions[i] = t_predictions[len(t_predictions) - 1]
        # pred_ci = pred.conf_int()
        # predictions = pred.predicted_mean
        predictions = [x if x > 0 else 0 for x in predictions]
        forecast_data[columnCntr] = np.array(predictions)

        columnCntr += 1
    forecasts = pd.DataFrame(forecast_data.T, columns=['C{}_Forecast'.format(c)
                                      for c in clusters.Cluster.values])
    df = realCrimes['C1_Crimes']
    forecasts.index = df[-test_size:].index
    print(forecasts)
    return forecasts, orders, seasonal_orders




    # # find initial date
    # compute all predictions in parallel
    # r_parallelComputeCVErrors = robjects.r('parallelComputeCVErrors')
    # errors = np.array(r_parallelComputeCVErrors(realCrimes, initialYear, initialWeek,
    #                                             periodsAhead=periodsAhead))
    # print(errors.shape)
    # print(periodsAhead)
    # if periodsAhead != 1:
    #     errors = np.array([errors[:, (c) * periodsAhead + periodsAhead - 1]
    #                        for c in range(0, len(clusters.Cluster.values))]).T
    # print(np.array(errors).shape)
    # # create DataFrame with forecasts
    # forecasts = pd.DataFrame(
    #     errors, columns=['C{}_Forecast'.format(c) for c in clusters.Cluster.values])
    # forecasts.index = realCrimes.index
    # # forecast = real_values - errors
    # for c in clusters.Cluster.values:
    #     forecasts['C{}_Forecast'.format(c)] = [a - b for a,
    #                                            b in zip(realCrimes['C{}_Crimes'.format(c)], forecasts['C{}_Forecast'.format(c)])]
    # return forecasts

# # following lines compute the forecast, called from ScriptForecast
# def init_predictions_Mean(window=4):
#     global window_global
#     window_global = window

# Compute predictions


# def predictions_Mean(clusters, realCrimes, periodsAhead=52):
#     # compute all predictions in parallel
#     forecasts = sequentialComputeForecasts(
#         realCrimes, periodsAhead=periodsAhead)

#     # create DataFrame with forecasts
#     forecasts = pd.DataFrame(np.transpose(forecasts), columns=[
#                              'C{}_Forecast'.format(c) for c in clusters.Cluster.values])
#     forecasts.index = realCrimes.index
#     return forecasts
