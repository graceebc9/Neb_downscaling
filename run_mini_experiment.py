import os 
import itertools
import json
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time 
import hashlib

# def spatial_feature_neighbor_graph(df, k, feature_cols, spatial_weight, 
#                                  distance_method='linear', distance_metric='haversine',
#                                  lat_col='latitude', lon_col='longitude', k_density=3):
    
from src.get_graph import get_graph
from src.run_graph_prop import run_graph_prop
from src.feature_graph import spatial_feature_neighbor_graph 
from src.geo_df import create_geo_df
from src.timeout_handler import timeout_handler, TimeoutException 
from src.logger import setup_logging
from src.saving_results import generate_run_name, save_results , create_analysis_dataframe, visualize_results
from src.checkpoints   import load_checkpoint 


def create_experiment_params(ld_cd, target_col='total_gas', feature_cols=None):
    """
    Create a dictionary of all experiment parameters that define a unique run
    Ensures all values are JSON serializable
    """
    if feature_cols is None:
        feature_cols = ['avg_gas', 'all_types_total_buildings']
        
    # Convert numpy arrays to lists and ensure all numbers are native Python types
    params = {
        'ld_cd': str(ld_cd),
        'target_col': str(target_col),
        'feature_cols': list(feature_cols),
        'random_seeds': [int(x) for x in [ 125, 400, 432, 589]],
        'missing_percentages': [float(x) for x in np.arange(0.1, 1, 0.2).tolist()],
        'distance_metrics': ['euclidean', 'haversine'],
        'distance_method' : ['linear', 'adaptive'] , 
        # 'spatial_weights': [float(x) for x in np.arange(0, 1.1,  0.1).tolist()],
        'spatial_weights': [0, 0.1, 0.7 , 0.9 ],
        'k_options': [int(x) for x in np.arange(5, 55, 10).tolist()] + [9,11,12,13,14,16,17]
    }


    return params

def run_experiments(input_data, experiment_params, timeout_seconds=300):
    # Generate run name from parameters
    run_name = generate_run_name(experiment_params)
    logger = setup_logging(run_name)
    
    logger.info(f"Starting experiment run: {run_name}")
    logger.info(f"Experiment parameters: {json.dumps(experiment_params, indent=2)}")
    
    # Load checkpoint if it exists
    previous_results, completed_experiments = load_checkpoint(run_name)
    all_results = previous_results if previous_results else []
    
    logger.info(f"Loaded {len(completed_experiments)} completed experiments from checkpoint")
    
    # Create experiment combinations
    param_combinations = list(itertools.product(
        experiment_params['random_seeds'],
        experiment_params['missing_percentages'],
        experiment_params['distance_metrics'],
        experiment_params['distance_method'],
        experiment_params['spatial_weights'],
        experiment_params['k_options']
    ))
    
    # Filter out already completed experiments
    param_combinations = [
        params for params in param_combinations 
        if params not in completed_experiments
    ]
    
    total_experiments = len(param_combinations)
    logger.info(f"Starting experiments. Remaining combinations to run: {total_experiments}")
    logger.info(f"Input data shape: {input_data.shape}")
    
    # Base feature parameters
    base_params = {
        'feature_cols': experiment_params['feature_cols'], 
    }
    
    try:
        # Run experiments
        for index, (rs, missing_pct, distance_metric, distance_method, weight, k) in enumerate(param_combinations, 1):
            start_time = time.time()
            logger.info(f"Experiment {index}/{total_experiments} ({(index/total_experiments)*100:.1f}%)")
            logger.info(f"Parameters: seed={rs}, missing={missing_pct:.2f}, metric:{distance_metric}, distance_method={distance_method}, weight={weight:.2f}, k={k}")
            
            # Update feature parameters
            feature_params = base_params.copy()
            feature_params.update({
                'distance_method': distance_method,
                'spatial_weight': weight
            })
            
            try:
                with timeout_handler(timeout_seconds):
                    print('k is ' , k)
                    # Generate graph
                    feature_graph = get_graph(
                        dataset_name=experiment_params['ld_cd'],
                        geo_df=input_data,
                        graph_fn=spatial_feature_neighbor_graph,
                        k=k,
                        graph_params=feature_params
                    )
                    
                    # Run experiment
                    rmse, mae, mape, r2 = run_graph_prop(
                        experiment_params['target_col'],
                        input_data,
                        missing_pct,
                        feature_graph,
                        distance_metric,
                        distance_method,
                        random_seed=rs
                    )
                    
                    # Store results
                    result = {
                        'random_seed': rs,
                        'missing_percentage': missing_pct,
                        'distance_metric': distance_metric,
                        'distance_method': distance_method,
                        'spatial_weight': weight,
                        'k': k,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'r2': r2,
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': time.time() - start_time
                    }
                    
                    logger.info(f"Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
                    
                    # Add to results list
                    all_results.append(result)
                    
                    # Save checkpoint every 10 experiments
                    if len(all_results) % 10 == 0:
                        logger.info(f"Saving checkpoint after {len(all_results)} experiments...")
                        save_results(all_results, run_name)
                    
            except (TimeoutException, Exception) as e:
                logger.error(f"Error in experiment: {str(e)}", exc_info=True)
                result = {
                    'random_seed': rs,
                    'missing_percentage': missing_pct,
                    'distance_metric': distance_metric,
                    'distance_method': distance_method,
                    'spatial_weight': weight,
                    'k': k,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(result)
                continue
                
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Saving current progress...")
        save_results(all_results, run_name, prefix="interrupt")
        raise
        
    logger.info("All experiments completed!")
    # Save final results
    save_results(all_results, run_name, prefix="final")
    
    return all_results, run_name


if __name__ == "__main__":
    # Define experiment parameters
    # ld_cd = 'E06000060'
    ld_cd='E06000052'
    experiment_params = create_experiment_params(ld_cd)

    # Load data
    # if hpc 
    # pc_path = '/home/gb669/rds/hpc-work/energy_map/data/postcode_polygons'
    # df_path = '/home/gb669/rds/hpc-work/energy_map/data/automl_models/input_data/new_final/NEBULA_englandwales_domestic_filtered.csv'
    # if local
    pc_path='/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998'
    df_path = '/Users/gracecolverd/NebulaDataset/final_dataset/NEBULA_englandwales_domestic_filtered.csv'
    run_name = generate_run_name(experiment_params)
    # dump params to json 
    os.makedirs(f'results/{run_name}', exist_ok=True)
    with open(f'results/{run_name}/experiment_params.json', 'w') as f:
        json.dump(experiment_params, f, indent=2)
    
    df = pd.read_csv(df_path) 
    df = create_geo_df(df, pc_path)
    
    geo_df = df[df['ladcd']==ld_cd].copy()
    
    # Run experiments
    results, run_name = run_experiments(geo_df, experiment_params)

    # Create analysis DataFrame
    logger = logging.getLogger(__name__)
    logger.info("Creating analysis dataframe")
    analysis_df = create_analysis_dataframe(results)
    
    # Get visualization-ready DataFrames
    logger.info("Generating visualization metrics")
    viz_dfs = visualize_results(analysis_df)
    
    # Save processed results
    processed_path = Path(f"results/{run_name}/processed_results.csv")
    analysis_df.to_csv(processed_path, index=False)
    logger.info(f"Processed results saved to {processed_path}")
    
    logger.info("Experiment pipeline complete!")
    logger.info(f"Total experiments run: {len(analysis_df)}")

