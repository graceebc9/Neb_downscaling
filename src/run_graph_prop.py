import numpy as np 
import logging
import src.diffusion as diffusion 
from src.metrics import calculate_metrics 

def run_graph_prop(target_col, geo_df, percent_missing, adj, distance_metric, d_method, random_seed=42, setting='random', custom_config=None):
    """
    Run graph propagation algorithm with missing data.
    
    Args:
        target_col: Column name for target variable
        geo_df: Geodataframe with data
        percent_missing: Percentage of data to mask
        adj: Adjacency matrix
        distance_metric: Distance metric used
        d_method; kind of distance method (linear or adaptive)
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (rmse, mae, mape, r2) metrics
    """
    logger = logging.getLogger(__name__)
    
    # Set the random seed 
    logger.info('Creating missing data')
    np.random.seed(random_seed)
    og_data = geo_df[target_col].values 
    incomplete_postcode_data = og_data.copy()  
    
    logger.info('Generating mask')  
    
    if setting =='random':
        missing_mask = np.random.choice([0, 1], size=og_data.shape[0], 
                                  p=[percent_missing, 1-percent_missing])
    
        # Validate that mask is correct number 
        actual_missing_percent = 1 - (missing_mask.sum() / missing_mask.shape[0])
        if abs(actual_missing_percent - percent_missing) > 0.01:  # 1% tolerance
            logger.warning(f'Mask percentage deviation: Expected {percent_missing:.3f}, got {actual_missing_percent:.3f}')

    elif setting == 'seperate':
        logger.info('trying seperate masks')
        missing = geo_df[geo_df['ladcd'].isin(custom_config.testing_codes)].index
        missing_mask =  np.random.choice( [1], size = incomplete_postcode_data.shape)
        missing_mask[missing] = 0 
        logger.info(f'Number of missing values: {len(missing)}') 
    else:
        logger.info('error no setting set thats recognised') 
        sys.exit() 
    
    incomplete_postcode_data[missing_mask==0] = np.nan  
    
    if incomplete_postcode_data.ndim != 1:
        logger.error('Error: Unexpected dimensions in incomplete_postcode_data')
        raise ValueError('incomplete_postcode_data must be 1-dimensional')
    
    logger.info('Starting diffusion completion')
    completed_pc_df = diffusion.graph_prop(adj, incomplete_postcode_data, missing_mask)
    logger.info('Diffusion completion finished')

    # Check if all values are above 0 for completed_pc_df 
    if not (completed_pc_df > 0).all():
        logger.error('Error: Negative or zero values found in completed_pc_df')
        raise ValueError('Completed data contains invalid values')

    logger.info('Calculating metrics')
    missing_data_subset = og_data[missing_mask==0]
    filled_subset = completed_pc_df[missing_mask==0]

    y_true = missing_data_subset
    y_pred = filled_subset 
    
    logger.info(f'Computing errors for target column {target_col} with {distance_metric} distance, and method {d_method}')
    rmse, mae, mape, r2 = calculate_metrics(y_true, y_pred)
    
    # Log the metrics
    logger.info(f'Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}')
    
    logger.info('Experiment complete')
    return rmse, mae, mape, r2