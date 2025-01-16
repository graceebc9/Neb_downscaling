import itertools
import json
from datetime import datetime
import pandas as pd
import numpy as np

from src.get_graph import get_graph
from src.run_graph_prop import run_graph_prop
from src.feature_graph import spatial_feature_neighbor_graph 
from src.geo_df import create_geo_df

def run_experiments(input_data, ld_cd, target_col='total_gas'):
    # Define parameter ranges
    random_seeds = [10, 50, 75, 100]
    missing_percentages = np.arange(0.1, 1.0, 0.1)  # 10% to 90%
    distance_metrics = ['haversine', 'euclidean']
    spatial_weights = np.arange(0, 1.0, 0.1)  # 0 to 1 in 0.1 increments

    # random_seeds = [1, 2]
    # missing_percentages = [0.1]
    # distance_metrics = ['haversine']
    # spatial_weights = [0.7]
    
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
    print(f"Starting experiments. Total combinations to run: {total_experiments}")
    
    # Run experiments
    for index, (rs, missing_pct, distance, weight) in enumerate(param_combinations, 1):
        # Print progress
        print(f"\rRunning experiment {index}/{total_experiments} "
              f"({(index/total_experiments)*100:.1f}%)", end='')
        print(rs, missing_pct, distance, weight)
        # Update feature parameters
        feature_params = base_params.copy()
        feature_params.update({
            'distance_metric': distance,
            'spatial_weight': weight
        })
        
        try:
            # Generate graph
            feature_graph = get_graph(
                dataset_name=ld_cd,
                geo_df=input_data,
                graph_fn=spatial_feature_neighbor_graph,
                graph_params=feature_params
            )
            print('graph generated')
            # Run experiment
            print('starting experiment')
            rmse, mae, mape, r2 = run_graph_prop(
                target_col,
                input_data,
                missing_pct,
                feature_graph,
                distance,
                random_seed=rs
            )
            print('storing results')
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
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to results list
            all_results.append(result)
            
            # Periodically save results to prevent data loss
            if len(all_results) % 50 == 0:
                save_interim_results(all_results)
                
        except Exception as e:
            print(f"\nError in experiment: {str(e)}")
            print(f"Parameters: seed={rs}, missing={missing_pct}, distance={distance}, weight={weight}")
            continue
    
    print("\nExperiments completed!")
    return all_results

def save_interim_results(results, prefix="interim"):
    """Save results both as JSON and CSV for backup"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = f"results_{prefix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = f"results_{prefix}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

def create_analysis_dataframe(results):
    """Convert results to a pandas DataFrame with additional analysis"""
    df = pd.DataFrame(results)
    
    # Add some useful derived metrics
    df['experiment_id'] = range(len(df))
    df['missing_percentage_str'] = df['missing_percentage'].apply(lambda x: f"{x*100:.0f}%")
    
    return df

def visualize_results(df):
    """Create visualization-ready DataFrames for different aspects of the results"""
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
    
    return {
        'missing_dist_metrics': missing_dist_metrics,
        'weight_metrics': weight_metrics,
        'best_params': best_params
    }

if __name__ == "__main__":
    # Run experiments
    ld_cd = 'E06000060'
    df = pd.read_csv('/Users/gracecolverd/NebulaDataset/final_dataset/NEBULA_englandwales_domestic_filtered.csv')
    df = create_geo_df(df)
    geo_df = df[df['ladcd']==ld_cd].copy() 
    print('Input data shape', geo_df.shape) 
    results = run_experiments(geo_df, ld_cd)
    
    # Save final results
    save_interim_results(results, prefix="final")
    
    # Create analysis DataFrame
    df = create_analysis_dataframe(results)
    
    # Get visualization-ready DataFrames
    viz_dfs = visualize_results(df)
    print('vis for df') 
    print(viz_dfs)

    # Save processed results
    df.to_csv("processed_results.csv", index=False)
    
    print("Experiment complete! Results saved and processed for visualization.")
    print(f"Total experiments run: {len(df)}")
    
    # Print some basic statistics
    print("\nBest performing parameters for each distance metric:")
    print(viz_dfs['best_params'][['distance_metric', 'missing_percentage', 
                                 'spatial_weight', 'rmse', 'r2']])