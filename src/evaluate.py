from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pr):
    mae = mean_absolute_error(y_true, y_pr)
    rmse = np.sqrt(mean_squared_error(y_true, y_pr))
    r2= r2_score(y_true, y_pr)
    print(f"平均误差: {mae:.2f} 均方差: {rmse:.2f}决定系数: {r2:.4f}")