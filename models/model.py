import os
import pandas as pd
import numpy as np
import xgboost as xgb
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import Split_the_dataset

def dummy_Mean(X_train,X_Verify,X_test,y_train):#均值回归模型
    dummy_mean= DummyRegressor(strategy='mean')
    dummy_mean.fit(X_train, y_train)
    y_pred_train = dummy_mean.predict(X_train)
    y_pred_Verify = dummy_mean.predict(X_Verify)
    y_pred_test= dummy_mean.predict(X_test)
    return y_pred_train, y_pred_test, y_pred_Verify

def lr_Model(X_train,X_Verify,X_test,y_train):# 线性回归模型
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_train = lr_model.predict(X_train)
    y_pred_Verify = lr_model.predict(X_Verify)
    y_pred_test = lr_model.predict(X_test)
    return y_pred_train, y_pred_test, y_pred_Verify

def rf_mMdel(X_train,X_Verify,X_test,y_train):  # 随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_train = rf_model.predict(X_train)
    y_pred_Verify = rf_model.predict(X_Verify)
    y_pred_test = rf_model.predict(X_test)
    return y_pred_train, y_pred_test, y_pred_Verify

def XGBoost_Model(X_train,X_Verify,X_test,y_train):#XGBoost模型
    XGBoost_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                                     colsample_bytree=0.8, random_state=42, n_jobs=-1)
    XGBoost_model.fit(X_train, y_train, eval_set=[(X_Verify, y_Verify)], verbose=100)
    y_pred_train = XGBoost_model.predict(X_train)
    y_pred_Verify = XGBoost_model.predict(X_Verify)
    y_pred_test = XGBoost_model.predict(X_test)
    return y_pred_train, y_pred_test, y_pred_Verify

def gb_Model(X_train,X_Verify,X_test,y_train):
    gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                         random_state=42)
    gb_model.fit(X_train, y_train)
    y_pred_train = gb_model.predict(X_train)
    y_pred_Verify = gb_model.predict(X_Verify)
    y_pred_test = gb_model.predict(X_test)
    return y_pred_train, y_pred_test, y_pred_Verify