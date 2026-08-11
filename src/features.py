import pandas as pd
import numpy as np

def create_features(df):
    df['house_age']=df['trade_year']-df['constructionTime']
    df['total_rooms']=df['drawingRoom']+df['livingRoom']+df['kitchen']+df['bathroom']
    df['room_density']=df['total_rooms']/df['square']
    return df