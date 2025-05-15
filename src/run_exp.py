from src.get_graph import get_graph
from src.run_graph_prop import run_graph_prop
from src.feature_graph import spatial_feature_neighbor_graph 
from src.geo_df import create_geo_df
from src.timeout_handler import timeout_handler, TimeoutException 
from src.logger import setup_logging
from src.checkpoints   import load_checkpoint 
import itertools 
import logging 
import time 
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import json 

from datetime import datetime
from src.saving_results import generate_run_name, save_results , create_analysis_dataframe, visualize_results


def save_config(custom_config, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    config_path = os.path.join(output_path, 'custom_config.json')
    config_dict = {
        'region_codes': custom_config.region_codes,
        'target_column': custom_config.target_column,
        'k_neighbors': custom_config.k_neighbors,
        'region_type': custom_config.region_type,
        'region_name': custom_config.region_name
    }
    
    with open(config_path , 'w') as f:
        json.dump(config_dict, f, indent=4)
        
def run_graph_experiments(input_data, config, timeout_seconds=400, op_folder='resultssss'):
    run_name = config.region_name 
    logger = setup_logging(run_name)
    # Load checkpoint if it exists
    previous_results, completed_experiments = load_checkpoint(run_name, op_folder)
    all_results = previous_results if previous_results else []

        # Create experiment combinations
    param_combinations = list(itertools.product(
        config.random_seeds,
        config.k_neighbors,
        config.distance_metrics,
        config.distance_method ,
        config.spatial_weight  , 
        config.missing_data_percent
    ))
    print(param_combinations)   
    param_combinations = [
        params for params in param_combinations 
        if params not in completed_experiments
    ]

    total_experiments = len(param_combinations)
    logger.info(f"Starting experiments. Remaining combinations to run: {total_experiments}")
    logger.debug(f"Input data shape: {input_data.shape}")
      
    # Base feature parameters
    base_params = {
        'feature_cols': config.feature_columns , 
        'spatial_weight': config.spatial_weight
    }
    
    try:
        for index, (rs, k, dist_metric, distance_method, weight, missing_data_percent) in enumerate(param_combinations, 1):
            start_time = time.time()
            logger.debug(f"Experiment {index}/{total_experiments} ({(index/total_experiments)*100:.1f}%)")
            
            
            # Update feature parameters
            feature_params = base_params.copy()
                        # Update feature parameters
 
            feature_params.update({
                'distance_method': distance_method,
                'distance_metric': dist_metric, 
                'spatial_weight': weight,
                
            })
     
            
            try:
                with timeout_handler(timeout_seconds):

                    # Generate graph
                    feature_graph = get_graph(
                        dataset_name=run_name,
                        geo_df=input_data,
                        graph_fn=spatial_feature_neighbor_graph,
                        k=k,
                        graph_params=feature_params
                    )
                    
                    # Run experiment
                    rmse, mae, mape, r2 = run_graph_prop(
                        config.target_column,
                        input_data,
                        missing_data_percent,
                        feature_graph,
                        dist_metric, 
                        distance_method,
                        random_seed=rs,
                        setting = config.graph_setting, 
                        custom_config = config
                    )
                    
                    # Store results
                    result = {
                        'random_seed': rs,
                        'missing_percentage':  missing_data_percent,
                        'distance_metric':dist_metric, 
                        'distance_method':  distance_method,
                        'spatial_weight': weight,    
                        'k': k,
                        'rmse': rmse,
                        'mae': mae,
                        'mape': mape,
                        'r2': r2,
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': time.time() - start_time
                    }
                    
                    logger.debug(f"Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
                    
                    # Add to results list
                    all_results.append(result)
                    
                    # Save checkpoint every 10 experiments
                    if len(all_results) % 10 == 0:
                        logger.debug(f"Saving checkpoint after {len(all_results)} experiments...")
                        save_results(all_results, run_name,  prefix="checkpoint" , op_folder=op_folder)
                    
            except (TimeoutException, Exception) as e:
                logger.error(f"Error in experiment: {str(e)}", exc_info=True)
                result = {
                    'random_seed': rs,
                    'k': k,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                all_results.append(result)
                continue
                
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user. Saving current progress...")
        save_results(all_results, run_name, prefix="interrupt", op_folder= op_folder)
        raise
        
    logger.info("All experiments completed!")
    # Save final results
    save_results(all_results, run_name, prefix="final", op_folder= op_folder)
    
    return all_results, run_name
    
def plot_metrics_analysis(df, output_path='metrics_analysis.png', region_name=None):
    """
    Create and save a dual-panel plot showing RMSE and R² statistics across different k values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the columns: k, rmse, r2, random_seed
    output_path : str
        Path where the figure should be saved
    region_name : str, optional
        Name of the region for the plot title
    """
    # Calculate statistics for both metrics
    stats = df.groupby('k').agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()
    
    # Flatten column names
    stats.columns = ['k', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    
    # Create a figure with two subplots side by side
    plt.figure(figsize=(15, 6))  # Set figure size before creating subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: RMSE
    ax1.plot(stats['k'], stats['rmse_mean'], 
            marker='o', linewidth=2, color='#1f77b4', label='Mean RMSE')
    ax1.fill_between(stats['k'],
                    stats['rmse_mean'] - stats['rmse_std'],
                    stats['rmse_mean'] + stats['rmse_std'],
                    alpha=0.3, color='#1f77b4', label='±1 std dev')
    
    ax1.set_xlabel('k Value', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE Analysis Across Different k Values\n(with Standard Deviation)', 
                 fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    ax1.set_xticks(stats['k'])
    
    # Add RMSE summary statistics - fixed position
    rmse_stats_text = (f"Total Random Seeds: {df['random_seed'].nunique()}\n"
                      f"Min RMSE: {df['rmse'].min():.3f}\n"
                      f"Max RMSE: {df['rmse'].max():.3f}")
    ax1.text(0.05, 0.95, rmse_stats_text,
             transform=ax1.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    # Set y-axis limits with some padding
    ax1.set_ylim(0, df['rmse'].max() * 1.2)
    
    # Plot 2: R²
    ax2.plot(stats['k'], stats['r2_mean'], 
            marker='o', linewidth=2, color='#2ca02c', label='Mean R²')
    ax2.fill_between(stats['k'],
                    stats['r2_mean'] - stats['r2_std'],
                    stats['r2_mean'] + stats['r2_std'],
                    alpha=0.3, color='#2ca02c', label='±1 std dev')
    
    ax2.set_xlabel('k Value', fontsize=12)
    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_title('R² Analysis Across Different k Values\n(with Standard Deviation)', 
                 fontsize=14, pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    ax2.set_xticks(stats['k'])
    
    # Set reasonable y-axis limits for R²
    ax2.set_ylim(0, 1.0)
    
    # Add R² summary statistics - fixed position
    r2_stats_text = (f"Total Random Seeds: {df['random_seed'].nunique()}\n"
                    f"Min R²: {df['r2'].min():.3f}\n"
                    f"Max R²: {df['r2'].max():.3f}")
    ax2.text(0.05, 0.95, r2_stats_text,
             transform=ax2.transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top',
             fontsize=10)
    
    # Add region name if provided
    if region_name:
        fig.suptitle(f'Analysis for {region_name}', fontsize=16, y=1.05)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save with a fixed DPI to control file size
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    
def create_multiple_metric_plots(df, group_columns, output_dir='./'):
    """
    Create separate metric analysis plots for different experimental conditions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    group_columns : list
        List of columns to group by (e.g., ['distance_metric', 'spatial_weight'])
    output_dir : str
        Directory to save the plots
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by the specified columns
    for name, group in df.groupby(group_columns):
        # Create filename from group values
        filename = '_'.join([f"{col}_{str(val)}" for col, val in zip(group_columns, name)])
        output_path = os.path.join(output_dir, f"metrics_analysis_{filename}.png")
        
        # Create plot for this group
        plot_metrics_analysis(group, output_path)
        
    
    
def run_automl(df, config, output_directory, time_limit= 1000):
    model_preset= 'best_quality'
    random_seed=42 
    label = config.target_column 
    
    def transform(df, label, cols ):
        working_cols = cols + [label]
        df = df[working_cols]
        df = df[~df[label].isna()]
        return df
    
    def save_results(results, output_path):
        res_string = str(results)
        # summary = predictor.fit_summary()
        with open(os.path.join(output_path, 'model_summary.txt'), 'w') as f:
            f.write(res_string)
            
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=random_seed)
    train_data = transform(TabularDataset(train_data), label, config.feature_cols)

    required_files = ['model_summary.txt'] 
    if check_directory_and_files(output_directory, required_files):
        sys.exit(0)
    else:
        os.makedirs(output_directory, exist_ok=True)
        
    
    size_train = len(train_data) 
    predictor = TabularPredictor(label, path=output_directory).fit(train_data, 
                                                                time_limit=time_limit,
                                                                presets=model_preset,
                                                                 )
    
    test_data = transform(TabularDataset(test_data), label, config.feature_cols)
    test_data.to_csv(os.path.join(output_directory, 'test_data.csv'), index=False)
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    results = predictor.evaluate_predictions(y_true=test_data[label], y_pred=y_pred, auxiliary_metrics=True)
    size_test = len(test_data)
    
    print(results)

    sizett = {'len_train' :size_train, 'len_test':size_test  }
    results.update(sizett)

    save_results(results, output_directory)
    res = predictor.leaderboard(test_data)
    res.to_csv(os.path.join(output_directory, 'leaderboard_results.csv'))
    
def load_data(custom_config, df_path, pc_path):    
    import pandas as pd
    print(custom_config.region_codes)
    df = pd.read_csv(df_path) 
    if custom_config.region_type =='ldcd':
        df = df[df['ladcd'].isin(custom_config.region_codes)]
    elif custom_config.region_type =='region':
        df =df[df['region'].isin(custom_config.region_codes)]
    geo_df = create_geo_df(df, pc_path)
    return geo_df 
    
def run_graph_config(geo_df, custom_config, output_path='results_new'): 
   
    results, run_name  = run_graph_experiments(geo_df.fillna(0), custom_config , op_folder=output_path) 
    op_path  = os.path.join(output_path, run_name)
    save_config(custom_config, op_path )
    analysis_df = create_analysis_dataframe(results)
    plot_metrics_analysis(analysis_df, Path(f"{output_path}/{run_name}/results_graph_rmse_R2.png") ,custom_config.region_name ) 
    processed_path = os.path.join(op_path,   "processed_results.csv")
    analysis_df.to_csv(processed_path, index=False)
    return op_path
    
 

