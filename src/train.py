import os
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import shap
import warnings
import matplotlib.pyplot as plt
from evaluate import evaluate_model
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
#数据集切分
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
filepath=os.path.join(project_root, 'data/processed/cleaned_data.csv')
X_train,X_Verify,X_test,y_train,y_Verify,y_test=Split_the_dataset(filepath)
print("数据集切分完成")
#对偏态的变量进行对数变换
y_train_log= np.log1p(y_train)
y_Verify_log= np.log1p(y_Verify)
y_test_log= np.log1p(y_test)
X_train['square'] = np.log1p(X_train['square'])
X_Verify['square'] = np.log1p(X_Verify['square'])
X_test['square'] = np.log1p(X_test['square'])
#类别变量编码
abel_cols = ['district', 'buildingType', 'renovationCondition', 'buildingStructure', 'elevator', 'fiveYearsProperty',
             'subway', 'floor_level']
ohe = OneHotEncoder(sparse_output=False)
ohe_df = pd.DataFrame(ohe.fit_transform(X_train[abel_cols]),  index=X_train.index)
X_train = pd.concat([X_train.drop(abel_cols, axis=1), ohe_df], axis=1)
ohe_verify_df = pd.DataFrame(ohe.transform(X_Verify[abel_cols]),  index=X_Verify.index)
X_Verify = pd.concat([X_Verify.drop(abel_cols, axis=1), ohe_verify_df], axis=1)
ohe_test_df = pd.DataFrame(ohe.transform(X_test[abel_cols]),  index=X_test.index)
X_test = pd.concat([X_test.drop(abel_cols, axis=1), ohe_test_df], axis=1)
#数字变量标准化
num_cols= ['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'ladderRatio', 'floor_sum',
           'house_age', 'total_rooms', 'room_density', 'communityAverage','followers', 'Lng', 'Lat']
stdsc = StandardScaler()
X_train[num_cols]= stdsc.fit_transform(X_train[num_cols])
X_Verify[num_cols]= stdsc.transform(X_Verify[num_cols])
X_test[num_cols]= stdsc.transform(X_test[num_cols])

X_train.columns = X_train.columns.astype(str)#保证列名是字符串
X_Verify.columns = X_Verify.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# dummy_mean= DummyRegressor(strategy='mean')#均值回归模型
# dummy_mean.fit(X_train, y_train)
# y_pred_train = dummy_mean.predict(X_train)
# y_pred_Verify = dummy_mean.predict(X_Verify)
# y_pred_test= dummy_mean.predict(X_test)

# lr_model = LinearRegression()#线性回归模型
# lr_model.fit(X_train, y_train)
# y_pred_train = lr_model.predict(X_train)
# y_pred_Verify = lr_model.predict(X_Verify)
# y_pred_test= lr_model.predict(X_test)

# #随机森林回归模型
# rf_model = RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)
# rf_model.fit(X_train, y_train)
# y_pred_train= rf_model.predict(X_train)
# y_pred_Verify= rf_model.predict(X_Verify)
# y_pred_test= rf_model.predict(X_test)

# XGBoost模型
XGBoost_model = xgb.XGBRegressor(n_estimators=1000,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1)
XGBoost_model.fit( X_train, y_train,eval_set=[(X_Verify, y_Verify)], verbose=100)
y_pred_train= XGBoost_model.predict(X_train)
y_pred_Verify = XGBoost_model.predict(X_Verify)
y_pred_test = XGBoost_model.predict(X_test)

# Gradient Boosting模型
# gb_model = GradientBoostingRegressor(n_estimators=500,learning_rate=0.05,max_depth=6,subsample=0.8,random_state=42)
# gb_model.fit(X_train, y_train)
# y_pred_train= gb_model.predict(X_train)
# y_pred_Verify= gb_model.predict(X_Verify)
# y_pred_test= gb_model.predict(X_test)

#评估函数输出结果
evaluate_model(y_train, y_pred_train)
evaluate_model(y_Verify, y_pred_Verify)
evaluate_model(y_test, y_pred_test)
#shap图
# explainer = shap.TreeExplainer(XGBoost_model)
# shap_values_test = explainer.shap_values(X_test)
# plt.figure(figsize=(10, 8))
# shap.summary_plot(shap_values_test, X_test, plot_type="bar", show=False)
# plt.xlabel('Mean |SHAP value|')
# plt.show()
#Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_Verify, y_Verify)],
        verbose=0
    )
    y_pred_Verify = model.predict(X_Verify)
    mae = mean_absolute_error(y_Verify, y_pred_Verify)
    return mae
warnings.filterwarnings('ignore')
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"最佳验证集 MAE: {study.best_value:.2f}")
print(f"最佳参数组合: {study.best_params}")
best_params = study.best_params
best_params['random_state'] = 42
best_params['n_jobs'] = -1
final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)
y_test_pred = final_model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
print("\n--- 最终测试集表现 ---")
print(f"R²: {test_r2:.4f}")
print(f"MAE: {test_mae:.2f}")