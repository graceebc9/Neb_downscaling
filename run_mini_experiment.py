import itertools
import json
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.get_graph import get_graph
from src.run_graph_prop import run_graph_prop
from src.feature_graph import spatial_feature_neighbor_graph 
from src.geo_df import create_geo_df

def setup_logging():
    """Configure logging settings"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Set up file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/experiment_{timestamp}.log"
    
    # Configure logging format and settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    return logging.getLogger(__name__)

# def run_experiments(input_data, ld_cd, target_col='total_gas'):
#     logger = setup_logging()
    
#     # Define parameter ranges
#     random_seeds = [10, 50, 75, 100]
#     missing_percentages = np.arange(0.1, 1.0, 0.1)  # 10% to 90%
#     distance_metrics = ['haversine', 'euclidean']
#     spatial_weights = np.arange(0, 1.0, 0.1)  # 0 to 1 in 0.1 increments
    
#     # Base feature parameters
#     base_params = {
#         'k': 5,
#         'feature_cols': ['avg_gas', 'all_types_total_buildings']
#     }
    
#     # Results storage
#     all_results = []
    
#     # Create experiment combinations
#     param_combinations = list(itertools.product(
#         random_seeds,
#         missing_percentages,
#         distance_metrics,
#         spatial_weights
#     ))
    
#     total_experiments = len(param_combinations)
#     logger.info(f"Starting experiments. Total combinations to run: {total_experiments}")
#     logger.info(f"Input data shape: {input_data.shape}")
    
#     # Run experiments
#     for index, (rs, missing_pct, distance, weight) in enumerate(param_combinations, 1):
#         logger.info(f"Experiment {index}/{total_experiments} ({(index/total_experiments)*100:.1f}%)")
#         logger.info(f"Parameters: seed={rs}, missing={missing_pct:.2f}, distance={distance}, weight={weight:.2f}")
        
#         # Update feature parameters
#         feature_params = base_params.copy()
#         feature_params.update({
#             'distance_metric': distance,
#             'spatial_weight': weight
#         })
        
#         try:
#             # Generate graph
#             logger.debug("Generating feature graph...")
#             feature_graph = get_graph(
#                 dataset_name=ld_cd,
#                 geo_df=input_data,
#                 graph_fn=spatial_feature_neighbor_graph,
#                 graph_params=feature_params
#             )
            
#             # Run experiment
#             logger.debug("Running graph propagation...")
#             rmse, mae, mape, r2 = run_graph_prop(
#                 target_col,
#                 input_data,
#                 missing_pct,
#                 feature_graph,
#                 distance,
#                 random_seed=rs
#             )
            
#             # Store results
#             result = {
#                 'random_seed': rs,
#                 'missing_percentage': missing_pct,
#                 'distance_metric': distance,
#                 'spatial_weight': weight,
#                 'rmse': rmse,
#                 'mae': mae,
#                 'mape': mape,
#                 'r2': r2,
#                 'timestamp': datetime.now().isoformat()
#             }
            
#             logger.info(f"Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
            
#             # Add to results list
#             all_results.append(result)
            
#             # Periodically save results
#             if len(all_results) % 50 == 0:
#                 logger.info(f"Saving interim results after {len(all_results)} experiments...")
#                 save_interim_results(all_results)
                
#         except Exception as e:
#             logger.error(f"Error in experiment: {str(e)}", exc_info=True)
#             continue
    
#     logger.info("All experiments completed!")
#     return all_results
import signal
from functools import wraps
from contextlib import contextmanager
import time

class TimeoutException(Exception):
    pass

@contextmanager
def timeout_handler(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Experiment timed out")
    
    # Register signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable alarm
        signal.alarm(0)

def run_experiments(input_data, ld_cd, target_col='total_gas', timeout_seconds=300):  # Default 5 min timeout
    logger = setup_logging()
    
    # Define parameter ranges
    random_seeds = [10, 50, 75, 100]
    missing_percentages = np.arange(0.1, 1.0, 0.1)  # 10% to 90%
    distance_metrics = ['haversine', 'euclidean']
    spatial_weights = np.arange(0, 1.0, 0.1)  # 0 to 1 in 0.1 increments
    
    # Base feature parameters
    base_params = {
        'k': 5,
        'feature_cols': ['avg_gas', 'all_types_total_buildings']
    }
    
    # Results storage
    all_results = []
    
    # Create experiment combinations
    param_combinations = list(itertools.product(
        random_seeds,
        missing_percentages,
        distance_metrics,
        spatial_weights
    ))
    
    total_experiments = len(param_combinations)
    logger.info(f"Starting experiments. Total combinations to run: {total_experiments}")
    logger.info(f"Input data shape: {input_data.shape}")
    
    # Run experiments
    for index, (rs, missing_pct, distance, weight) in enumerate(param_combinations, 1):
        start_time = time.time()
        logger.info(f"Experiment {index}/{total_experiments} ({(index/total_experiments)*100:.1f}%)")
        logger.info(f"Parameters: seed={rs}, missing={missing_pct:.2f}, distance={distance}, weight={weight:.2f}")
        
        # Update feature parameters
        feature_params = base_params.copy()
        feature_params.update({
            'distance_metric': distance,
            'spatial_weight': weight
        })
        
        try:
            with timeout_handler(timeout_seconds):
                # Generate graph
                logger.debug("Generating feature graph...")
                feature_graph = get_graph(
                    dataset_name=ld_cd,
                    geo_df=input_data,
                    graph_fn=spatial_feature_neighbor_graph,
                    graph_params=feature_params
                )
                
                # Run experiment
                logger.debug("Running graph propagation...")
                rmse, mae, mape, r2 = run_graph_prop(
                    target_col,
                    input_data,
                    missing_pct,
                    feature_graph,
                    distance,
                    random_seed=rs
                )
                
                # Store results
                result = {
                    'random_seed': rs,
                    'missing_percentage': missing_pct,
                    'distance_metric': distance,
                    'spatial_weight': weight,
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
                
                # Periodically save results
                if len(all_results) % 50 == 0:
                    logger.info(f"Saving interim results after {len(all_results)} experiments...")
                    save_interim_results(all_results)
                
        except TimeoutException:
            logger.error(f"Experiment timed out after {timeout_seconds} seconds")
            result = {
                'random_seed': rs,
                'missing_percentage': missing_pct,
                'distance_metric': distance,
                'spatial_weight': weight,
                'error': 'timeout',
                'timeout_limit': timeout_seconds,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)
            continue
            
        except Exception as e:
            logger.error(f"Error in experiment: {str(e)}", exc_info=True)
            result = {
                'random_seed': rs,
                'missing_percentage': missing_pct,
                'distance_metric': distance,
                'spatial_weight': weight,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)
            continue
    
    logger.info("All experiments completed!")
    return all_results



def save_interim_results(results, prefix="interim"):
    """Save results both as JSON and CSV for backup"""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    Path("results_new").mkdir(exist_ok=True)
    
    # Save as JSON
    json_path = f"results_new/results_{prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = f"results_new/results_{prefix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Results saved to {json_path} and {csv_path}")

def create_analysis_dataframe(results):
    """Convert results to a pandas DataFrame with additional analysis"""
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(results)
    
    # Add some useful derived metrics
    df['experiment_id'] = range(len(df))
    df['missing_percentage_str'] = df['missing_percentage'].apply(lambda x: f"{x*100:.0f}%")
    
    logger.info(f"Created analysis dataframe with {len(df)} rows")
    return df

def visualize_results(df):
    """Create visualization-ready DataFrames for different aspects of the results"""
    logger = logging.getLogger(__name__)
    
    # Average metrics by missing percentage and distance metric
    missing_dist_metrics = df.groupby(['missing_percentage', 'distance_metric'])[
        ['rmse', 'mae', 'mape', 'r2']
    ].mean().reset_index()
    
    # Average metrics by spatial weight
    weight_metrics = df.groupby('spatial_weight')[
        ['rmse', 'mae', 'mape', 'r2']
    ].agg(['mean', 'std']).reset_index()
    
    # Get best performing parameters
    best_params = df.loc[df.groupby('distance_metric')['rmse'].idxmin()]
    
    logger.info("Generated visualization metrics")
    logger.info(f"Best RMSE by distance metric:\n{best_params[['distance_metric', 'rmse']]}")
    
    return {
        'missing_dist_metrics': missing_dist_metrics,
        'weight_metrics': weight_metrics,
        'best_params': best_params
    }

if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Starting main experiment pipeline")
    
    # Run experiments
    ld_cd = 'E06000060'
    logger.info(f"Loading data for region: {ld_cd}")
    
    df = pd.read_csv('/Users/gracecolverd/NebulaDataset/final_dataset/NEBULA_englandwales_domestic_filtered.csv')
    df = create_geo_df(df)
    geo_df = df[df['ladcd']==ld_cd].copy()
    
    logger.info(f"Input data shape: {geo_df.shape}")
    results = run_experiments(geo_df, ld_cd)
    
    # Save final results
    logger.info("Saving final results")
    save_interim_results(results, prefix="final")
    
    # Create analysis DataFrame
    logger.info("Creating analysis dataframe")
    df = create_analysis_dataframe(results)
    
    # Get visualization-ready DataFrames
    logger.info("Generating visualization metrics")
    viz_dfs = visualize_results(df)
    
    # Save processed results
    processed_path = "results/processed_results.csv"
    df.to_csv(processed_path, index=False)
    logger.info(f"Processed results saved to {processed_path}")
    
    logger.info("Experiment pipeline complete!")
    logger.info(f"Total experiments run: {len(df)}")