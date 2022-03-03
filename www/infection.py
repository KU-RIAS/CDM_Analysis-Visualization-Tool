import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
from sklearn.metrics import r2_score


import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def data_split(data, organism='staphylococcus aureus', date='2018-01-01'):
    '''
    Train/Test data split
    '''
    if not any((isinstance(x, datetime) for x in data.index)):
        data.index = pd.to_datetime(data.index)

    data = data[data['organism'] == organism].drop('organism', axis=1)  # 해당 organism 추출
    train_data = data[data.index < date]  # 2010년 1월 - 2017년 12월 (7년)
    test_data = data[data.index >= date]  # 2018년 1월 - 2018년 12월 (1년) 
    return train_data, test_data

def get_decomposition(data, target='infection_risk', visualize=False):
    '''
    시계열을 분해하여 Trend와 Seasonality가 존재하는지 여부 확인
    '''
    decompostion = sm.tsa.seasonal_decompose(data[target], model='additive',period=12)
    observed = decompostion.observed
    trend = decompostion.trend
    seasonal = decompostion.seasonal
    resid = decompostion.resid

    if visualize:
        fig, ax = plt.subplots(4, 1, figsize=(12, 12))
        ax[0].plot(observed)
        ax[1].plot(trend)
        ax[2].plot(seasonal)
        ax[3].plot(resid)

        ax[0].set_ylabel('observed')
        ax[1].set_ylabel('trend')
        ax[2].set_ylabel('seasonal')
        ax[3].set_ylabel('resid')
        plt.show()

    return observed, trend, seasonal, resid

def acf_pacf_graph(data):
    '''
    data에 대한 ACF, PACF 그래프
    '''
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle('ACF & PACF')
    sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=30, ax=ax[0])
    sm.graphics.tsa.plot_pacf(data.values.squeeze(), lags=30, ax=ax[1]); # 그래프 2번 안나오게 ;
    
def differencing_data(data, target='infection_risk'):
    '''
    raw data에 대한 1차 차분
    '''
    diff_data = data[target].diff()
    diff_data = diff_data.dropna()
    return diff_data

def arima_train(data, order=(1,1,0)):
    '''
    arima model train
    '''
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    aic, bic, mse, mae = model_fit.aic, model_fit.bic, model_fit.mse, model_fit.mae
    return model_fit
    
def search_best_arima(data, model_type='arima'):
    '''
    Grid-search로 최적의 Hyper-parameter tuning
    model_type: arima or sarima
    '''
    
    assert model_type == 'arima' or model_type == 'sarima', 'Only support arima or sarima models'
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Parameter seach
    p = range(0,3)
    d = range(1,2)
    q = range(0,3)
    pdq = list(itertools.product(p, d, q))
    
    if model_type =='sarima':
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    scores=[]
    params=[]
    for i in pdq:
        if model_type =='sarima':
            for j in seasonal_pdq:
                model = SARIMAX(data, order=(i), seasonal_order=(j))
                model_fit = model.fit()
                scores.append(round(model_fit.mae,2))
                params.append((i, j))
        else:
            model = ARIMA(data, order=(i))
            model_fit = model.fit()
            print(f'ARIMA: {i} >> AIC : {round(model_fit.mae,2)}')
            scores.append(round(model_fit.mae,2))

    # Re-training with best paramters
    if model_type =='sarima':
        optimal = [(params[i], j) for i,j in enumerate(scores) if j==min(scores)]
        best_model = SARIMAX(data, order=optimal[0][0][0], seasonal_order=optimal[0][0][1])
        best_model_fit = best_model.fit()
    else:
        optimal = [(pdq[i], j) for i,j in enumerate(scores) if j==min(scores)]
        best_model = ARIMA(data, order=optimal[0][0])
        best_model_fit = best_model.fit()

    return best_model_fit


def arima_predict(model, test_data, train_data=None, visualize=False, extension=12):
    '''
    학습된 모델로 test 기간 예측 수행
    extension: 예측할 기간 (Test set 외 이후 기간 예측, 단위:월)
    '''
    result = model.get_forecast(len(test_data)+extension)
    result = result.summary_frame(alpha=0.05)    # 신뢰수준 95%

    # Predict, Upper, Lower bound values (95% confidence-level)
    predicted_value = result['mean']
    predicted_ub = result['mean_ci_upper']
    predicted_lb = result['mean_ci_lower']
    predict_index = list(test_data.index)
    for i in range(0, extension-1):
        prev_month = predict_index[-1].month
        iter_date = predict_index[-1]
        while prev_month == iter_date.month:
            iter_date += pd.Timedelta(1, unit='d')
        predict_index.append(iter_date)
    
    if visualize:
        fig, ax = plt.subplots(figsize=(15,10))
        ax.plot(pd.concat([train_data, test_data]), label='Actual value')
        ax.plot(predict_index, predicted_value, label='Prediction')
        ax.vlines(pd.to_datetime('2018-01-01'), 0, 100, linestyle='--', color='r', label='Start of Forecast')
        ax.fill_between(predict_index, predicted_lb, predicted_ub, color='k', alpha=0.1, label='0.95 Prediction Interval')
        ax.legend(loc='upper left')
        plt.suptitle(f'ARIMA Prediction Results')
        plt.show()
        
    return predicted_value, predicted_ub, predicted_lb, predict_index

