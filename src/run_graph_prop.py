import numpy as np 
import src.diffusion as diffusion 
from src.metrics import calculate_metrics 

def run_graph_prop(target_col, geo_df, percent_missing, adj, distance, random_seed=42):
    # set the random seed 
    print('creating missing data')
    np.random.seed(random_seed)
    og_data =  geo_df[target_col].values 
    incomplete_postcode_data= og_data.copy()  
    print('generating mask')  
    # generate maks using percent_missing 
    missing_mask = np.random.choice([0, 1], size=og_data.shape[0], p=[percent_missing, 1-percent_missing])
    # validate that mask is correct number 
    actual_missing_percent = 1 - (missing_mask.sum() / missing_mask.shape[0])
    if abs(actual_missing_percent - percent_missing) > 0.01:  # 1% tolerance
        print(f'Error with mask: Expected {percent_missing}, got {actual_missing_percent}')
    
    incomplete_postcode_data[missing_mask==0] = np.nan  

    if incomplete_postcode_data.ndim != 1:
        print('error expecting diff dimesnions')
    
    print('starting diffusion completion')
    completed_pc_df = diffusion.graph_prop(adj, incomplete_postcode_data, missing_mask )
    print('diffusion completion done')

    # check if all above 0  for completed_pc_df 
    if not (completed_pc_df > 0).all():
        print('error in completed_pc_df')


    print('calculating metrics')
    missing_data_subset = og_data[missing_mask==0]
    filled_subset = completed_pc_df[missing_mask==0]


    y_true = missing_data_subset
    y_pred = filled_subset 
    print(f'The errors for filling target col: {target_col} with spatial graph on {distance} distance are:')
    rmse, mae, mape, r2 = calculate_metrics(y_true, y_pred)
    print('experiment complete')
    return rmse, mae, mape, r2


