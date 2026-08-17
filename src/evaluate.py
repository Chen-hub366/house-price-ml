import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import Split_the_dataset
def evaluate_model(y_true, y_pr):
    mae = mean_absolute_error(y_true, y_pr)
    rmse = np.sqrt(mean_squared_error(y_true, y_pr))
    r2= r2_score(y_true, y_pr)
    print(f"平均误差: {mae:.2f} 均方差: {rmse:.2f}决定系数: {r2:.4f}")
def shap_text(XGBoost_model):#shap图
    explainer = shap.TreeExplainer(XGBoost_model)
    shap_values_test = explainer.shap_values(X_test)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
    plt.xlabel('Mean |SHAP value|')
    plt.show()
