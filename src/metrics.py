import numpy as np 
from sklearn import metrics

def calculate_mape(y_true, y_pred):
    # Remove pairs where y_true is zero
    mask = y_true != 0
    if not mask.any():
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Arrays contain NaN or infinite values")
    
    # Calculate metrics
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    
    # MAPE with zero handling
    mape = calculate_mape(y_true, y_pred)
    
    # Format and print results
    print('RMSE: {:,.1f}'.format(rmse),
          'MAE: {:,.1f}'.format(mae),
          'MAPE: {:.1f}'.format(mape),
          'R2: {:.2f}'.format(r2))
    
    return rmse, mae, mape, r2