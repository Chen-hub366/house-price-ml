import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocess import Split_the_dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
filepath=os.path.join(project_root, 'data/processed/cleaned_data.csv')
X_train,X_Verify,X_test,y_train,y_Verify,y_test=Split_the_dataset(filepath)
print("数据集切分完成")

y_train_log= np.log1p(y_train)
y_Verify_log= np.log1p(y_Verify)
y_test_log= np.log1p(y_test)
X_train['square'] = np.log1p(X_train['square'])
X_Verify['square'] = np.log1p(X_Verify['square'])
X_test['square'] = np.log1p(X_test['square'])

abel_cols = ['district', 'buildingType', 'renovationCondition', 'buildingStructure', 'elevator', 'fiveYearsProperty', 'subway', 'floor_level']
