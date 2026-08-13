import os
import pandas as pd
import numpy as np
from evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
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
num_cols= ['square', 'livingRoom', 'drawingRoom', 'kitchen', 'bathRoom', 'ladderRatio', 'floor_sum', 'trade_year',
           'trade_month', 'house_age', 'total_rooms', 'room_density', 'followers', 'communityAverage', 'Lng', 'Lat']
stdsc = StandardScaler()
X_train[num_cols]= stdsc.fit_transform(X_train[num_cols])
X_Verify[num_cols]= stdsc.transform(X_Verify[num_cols])
X_test[num_cols]= stdsc.transform(X_test[num_cols])

X_train.columns = X_train.columns.astype(str)#保证列名是字符串
X_Verify.columns = X_Verify.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

dummy_mean= DummyRegressor(strategy='mean')#均值回归模型
dummy_mean.fit(X_train, y_train)
y_pr_train = dummy_mean.predict(X_train)
y_pr_Verify = dummy_mean.predict(X_Verify)
y_pr_test= dummy_mean.predict(X_test)

evaluate_model(y_train, y_pr_train)#测试
evaluate_model(y_Verify, y_pr_Verify)
evaluate_model(y_test, y_pr_test)

