# from pathlib import Path
# import pandas as pd
# import os 

# import glob 

# def load_checkpoint(run_name, op_folder):
#     """Load the latest checkpoint for a given run"""
    
#     # results_dir = Path(f"results/{run_name}")
#     results_dir = os.path.join(op_folder, run_name)
#     # if not results_dir.exists():
#     if not os.path.exists(results_dir):
#         return None, set()
        
#     # Find the latest checkpoint file
#     # checkpoint_files = list(results_dir.glob("results_checkpoint_*.csv"))
#     checkpoint_files  =  glob.glob(os.path.join(results_dir, "results_checkpoint_*.csv"))
#     if not checkpoint_files:
#         print('no checkpoint files found')
#         return None, set()
        
#     latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
#     # Load the results
#     df = pd.read_csv(latest_checkpoint)
    
#     # Create a set of completed experiment parameters
#     completed_experiments = {
#         (row['random_seed'], row['missing_percentage'], 
#          row['distance_metric'], row['spatial_weight'], row['k'])
#         for _, row in df.iterrows()
#     }
    
#     return df.to_dict('records'), completed_experiments



# def load_checkpoint(run_name, op_folder, prefix="checkpoint"):
#     """Load the latest checkpoint for a given run"""
    
#     # Create the directory path using prefix instead of run_name for consistency with save_results
#     results_dir = os.path.join(op_folder, prefix)
#     if not os.path.exists(results_dir):
#         return None, set()
        
#     # Find the latest checkpoint file - adjust the glob pattern to match save_results naming
#     checkpoint_files = glob.glob(os.path.join(results_dir, f"results_{prefix}_*.csv"))
#     if not checkpoint_files:
#         print('no checkpoint files found')
#         return None, set()
        
#     # Find the latest checkpoint file using modification time
#     latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    
#     # Load the results
#     df = pd.read_csv(latest_checkpoint)
    
#     # Create a set of completed experiment parameters
#     completed_experiments = {
#         (row['random_seed'], row['missing_percentage'], 
#          row['distance_metric'], row['spatial_weight'], row['k'])
#         for _, row in df.iterrows()
#     }
    
#     return df.to_dict('records'), completed_experiments


import pandas as pd
import os
import glob


def load_checkpoint(run_name, op_folder='results', prefix="checkpoint"):
    """Load the latest checkpoint for a given run"""
    
    # Create the directory path using run_name to match the save_results function
    results_dir = os.path.join(op_folder, run_name)
    if not os.path.exists(results_dir):
        print(f"No checkpoint directory found at {results_dir}")
        return None, set()
    
    # Find the latest checkpoint file - adjust the glob pattern to match save_results naming
    checkpoint_files = glob.glob(os.path.join(results_dir, f"results_{prefix}_*.csv"))
    if not checkpoint_files:
        print(f'No checkpoint files found in {results_dir}')
        return None, set()
    
    # Find the latest checkpoint file using modification time
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading checkpoint from: {latest_checkpoint}")
    
    # Load the results
    df = pd.read_csv(latest_checkpoint)
    
    # Create a set of completed experiment parameters
    # Make sure this matches the structure in your run_graph_experiments function
    completed_experiments = set()
    for _, row in df.iterrows():
        try:
            # Extract all parameters that define a unique experiment
            experiment = (
                row['random_seed'],
                row['k'],
                row['distance_metric'],
                row['distance_method'] if 'distance_method' in df.columns else None,
                row['spatial_weight'],
                row['missing_percentage'] if 'missing_percentage' in df.columns else None
            )
            completed_experiments.add(experiment)
        except KeyError as e:
            print(f"Warning: Missing key in checkpoint data: {e}")
            # Continue with partial information if some keys are missing
            continue
    
    print(f"Loaded {len(df)} completed experiments")
    return df.to_dict('records'), completed_experiments