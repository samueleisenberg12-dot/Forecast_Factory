# Imports
# ========================================

# API Imports
from fastapi import FastAPI, File, UploadFile

# General Library Imports

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.indexes.datetimes import DatetimeIndex

# Data Science Imports

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Machine Learning Imports
from sklearn.linear_model import LinearRegression
from tensorflow.keras.layers import LSTM
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform

# ========================================
# Function Definitions 
# ========================================

# Data Preprocessing
# ========================================

def clean_df(df: pd, demand_column_str: str, day_column_str: str):
    # Extract only date/time column and demand value column
    df_new = df[[demand_column_str, day_column_str]]
    # Ensure demand column is numeric values
    df_new[demand_column_str] = pd.to_numeric(df_new[demand_column_str], errors='coerce')

    # Ensure day column is datetime
    df_new[day_column_str] = pd.to_datetime(df_new[day_column_str])

    # Sum same day data
    df_new = df_new.groupby(day_column_str, as_index=False)[demand_column_str].sum()
 
    # Verify that data is exactly one day apart
    df_new = df_new.sort_values(by=day_column_str)
    deltas = df_new[day_column_str].diff().dropna()
    is_daily = (deltas == pd.Timedelta(days=1)).all()
    is_daily=True

    # Set day to index
    df_new.set_index(day_column_str, inplace=True)

    return df_new, is_daily

def split_data(pt,pv, raw_data):

    assert(pt>=0)
    assert(pv>=0)
    assert(pt+pv<=1)

    segment_train = int(len(raw_data)*pt)
    segment_validate = int(len(raw_data)*pv)
    
    data_train = raw_data.iloc[:segment_train,:].copy()
    data_validate = raw_data.iloc[segment_train:(segment_train+segment_validate),:].copy()
    data_test = raw_data.iloc[segment_train+segment_validate:,:].copy()
    
    return data_train, data_validate, data_test

def separate_and_normalize(data_train, data_validate, data_test):

    scaler = StandardScaler()

    scaler.fit(data_train)

    data_train['scaled'] = scaler.transform(data_train)
    data_validate['scaled'] = scaler.transform(data_validate)
    data_test['scaled'] = scaler.transform(data_test)

    return data_train, data_validate, data_test, scaler

def organize_data(d,F=6,T=12):

    N = d.shape[0]
    Np = N - F - T + 1
    X = np.empty((Np,T))
    Y = np.empty(Np)
    t = np.empty(Np,dtype=DatetimeIndex)

    # print(d)
    # Fill in X, Y, and t
    for i in range(Np):
        X[i, :] = d.iloc[i : i+T].transpose()
        Y[i] = d.iloc[i + T - 1 + F] 
        t[i] = d.index[i + T]

    return X, Y, t

def append_day(X,t):
    t = np.array([i.day for i in t])
    t = t.reshape(t.shape[0], 1)
    return np.hstack((X, t))

# Model Training and Evaluations
# ========================================

def assess(x_current,y,yhat):
    y = np.array(y).flatten()
    yhat = np.array(yhat).flatten()
    x_current = np.array(x_current).flatten()
    
    mae_m = sum(abs(y-yhat))/len(y)
    mae_b = sum(abs(y-x_current))/len(y)
    perf = 1 - (mae_m/mae_b)

    return perf

def linreg_eval(Xtrain, Xtrain_day, ytrain, Xvalid, Xvalid_day, yvalid):
    model = LinearRegression().fit(Xtrain, ytrain)
    day_model = LinearRegression().fit(Xtrain_day, ytrain)

    linreg_perf = assess(Xvalid[:,-1], yvalid, model.predict(Xvalid))
    linreg_day_perf = assess(Xvalid_day[:,-2], yvalid, day_model.predict(Xvalid_day))

    models = np.array([model, day_model])
    performance = np.array([linreg_perf, linreg_day_perf])

    idx = np.argmax(performance)
    best_model = models[idx]
    best_model_r2 = performance[idx]
    return best_model, best_model_r2

def lstm_eval(Xtrain_day, ytrain, Xvalid_day, Xvalid, yvalid):
    ki = GlorotUniform(seed=2434) # This is needed for reproducibility of the results

    random.seed(2434)  # Do not change this. It is needed to ensure repeatability.

    model_lstm = Sequential([
        LSTM(16, input_shape=(Xtrain_day.shape[1],1), return_sequences=True, kernel_initializer=ki),
        LSTM(16, return_sequences=True, kernel_initializer=ki),
        LSTM(8, kernel_initializer=ki),
        Dense(1, kernel_initializer=ki)
    ])

    model_lstm.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
    history_lstm = model_lstm.fit(x=Xtrain_day, y=ytrain, epochs=10, validation_data=(Xvalid_day,yvalid))
    lstm_perf = assess(Xvalid[:,-1],yvalid,model_lstm.predict(Xvalid_day))
    return model_lstm, lstm_perf

def best_perf(linreg_perf, lstm_perf, SRNN_perf=-1):
    dict = {
        "Linear Regression Performance: ": linreg_perf,
        "LSTM Performance: ": lstm_perf,
        "SRNN Performance: ": SRNN_perf
    }

    best_name = max(dict, key=dict.get)
    best_value = dict[best_name]

    return best_name, best_value

def forecast(model, df, demand_column_str, days, scaler, T=12):

    # Get the last T values as a plain Python list of floats
    last_sequence = df[demand_column_str].values[-T:].astype(float).tolist()
    last_date = df.index[-1]

    forecast_dict = {}

    for i in range(1, days + 1):
        # Make sure the input is a 2D numpy array of shape (1, T)
        X_input = np.array(last_sequence[-T:], dtype=float).reshape(1, T)

        # Predict next value
        y_pred = model.predict(X_input)[0]

        # Determine the date for this prediction
        next_date = last_date + pd.Timedelta(days=i)

        # Store in dictionary
        forecast_dict[next_date.strftime("%Y-%m-%d")] = float(y_pred)

        # Append prediction to sequence for next iteration
        last_sequence.append(float(y_pred))

    return forecast_dict

