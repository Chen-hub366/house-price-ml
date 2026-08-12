import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def clean_columns (df):#清理无用的列
    columns_del = ['url', 'id', 'Cid', 'DOM', 'price']
    df.drop(columns_del, axis=1,inplace=True,errors='ignore')
    return df

def clean_duplicate(df):#清理重复值
    df.drop_duplicates(inplace=True)
    df.index = range(df_clean.shape[0])
    df.index
    return df

def clean_missing(df):#清理缺失值
    missing = df.isnull().mean()
    missing_del = missing[missing > 0.25].index
    df.drop(columns=missing_del, inplace=True)
    buildingType_mode = df['buildingType'].mode()[0]
    df['buildingType'] = df['buildingType'].fillna(buildingType_mode)
    elevator_mode = df['elevator'].mode()[0]
    df['elevator'] = df['elevator'].fillna(elevator_mode)
    df['fiveYearsProperty'] = df['fiveYearsProperty'].fillna(0)
    district_median = df.groupby('district')['communityAverage'].median()
    missing_Average = df['communityAverage'].isnull()
    df.loc[missing_Average, 'communityAverage'] = df.loc[missing_Average, 'district'].map(district_median)
    subway_mode = df['subway'].mode()[0]
    df['subway'] = df['subway'].fillna(subway_mode)
    return df

def num_cols(df):#转换数据类型，解析特征
    float_cols =['Lng','Lat','totalPrice', 'square','communityAverage']
    for col in float_cols:
        df[col]=pd.to_numeric(df[col], errors='coerce')
    int_cols =['livingRoom', 'drawingRoom', 'kitchen','bathRoom', 'elevator','subway','fiveYearsProperty']
    for cols in int_cols:
        df[cols] =pd.to_numeric(df[cols], errors='coerce').fillna(0).astype(int)
    df['constructionTime'] = pd.to_numeric(df['constructionTime'], errors='coerce')
    df['floor_level'] = df['floor'].str[0].map({'底': 0, '低': 1, '中': 2, '高': 3, '顶': 4})
    df['floor_sum'] = pd.to_numeric(df['floor'].str[1:], errors='coerce')
    df['floor_level'] = df['floor_level'].fillna(-1).astype(int)
    df['floor_sum'] = df['floor_sum'].fillna(-1).astype(int)
    df.drop('floor', axis=1, inplace=True, errors='ignore')
    df['tradeTime'] = pd.to_datetime(df['tradeTime'], errors='coerce')
    df['trade_year'] = df['tradeTime'].dt.year
    df['trade_month'] = df['tradeTime'].dt.month
    df.drop('tradeTime', axis=1, inplace=True, errors='ignore')
    return df

def clean_abnormal(df):#处理异常值
    df = df[(df['square'] >= 20) & (df['square'] <= 500)]
    df= df[(df['totalPrice']>50)&(df['totalPrice']<3000)]
    df = df[(df['livingRoom'] !=0)&(df['drawingRoom'] !=0) & (df['kitchen'] !=0) & (df['bathRoom'] !=0)]
    df = df[df['trade_year'] > df['constructionTime']]
    return df

def full_clean(df):#清洗所有数据
    df=clean_columns (df)
    df=clean_duplicate(df)
    df=clean_missing(df)
    df=num_cols(df)
    df=clean_abnormal(df)
    return df
def Split_the_dataset(filepath):
    df = pd.read_csv(filepath)
    X = df.drop('totalPrice', axis=1)
    y = df['totalPrice']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test,X_Verify,y_test, y_Verify = train_test_split(X_temp, y_temp, test_size=1/2, random_state=42)
    return X_train,X_Verify,X_test,y_train,y_Verify,y_test

