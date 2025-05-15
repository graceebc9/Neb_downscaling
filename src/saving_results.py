import logging 
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np 
import hashlib 
import os 

def generate_run_name(experiment_params):
    """
    Generate a deterministic run name based on experiment parameters
    Ensures all values are JSON serializable
    """
    # Convert any remaining numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        return obj

    # Convert all parameters to serializable format
    serializable_params = convert_to_serializable(experiment_params)
    print('params: ', serializable_params) 
    
    # Sort the parameters to ensure consistent ordering
    param_str = json.dumps(serializable_params, sort_keys=True)
    
    # Create a hash of the parameters
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    # Create a readable run name
    run_name = f"run_{experiment_params['name']}_{param_hash}"
    return run_name


def save_results(results, run_name, prefix="checkpoint", op_folder='results'):
    """Save results both as JSON and CSV in run-specific directory"""
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory for this run (using run_name as the directory)
    results_dir = os.path.join(op_folder, run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save as JSON
    json_path = os.path.join(results_dir, f"results_{prefix}_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metadata about the run
    metadata_path = os.path.join(results_dir, "run_metadata.json")
    
    if not os.path.exists(metadata_path):
        metadata = {
            'run_name': run_name,
            'start_time': timestamp,
            'last_update': timestamp,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    else:
        # Update the last_update time in metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        metadata['last_update'] = timestamp
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, f"results_{prefix}_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Results saved to {json_path} and {csv_path}")


def create_analysis_dataframe(results):
    """Convert results to a pandas DataFrame with additional analysis"""
    logger = logging.getLogger(__name__)
    df = pd.DataFrame(results)
    
    # Add some useful derived metrics
    df['experiment_id'] = range(len(df))
    try:
        df['missing_percentage_str'] = df['missing_percentage'].apply(lambda x: f"{x*100:.0f}%")
    except:
        None 
    
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
