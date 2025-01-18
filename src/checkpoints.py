from pathlib import Path
import pandas as pd


def load_checkpoint(run_name):
    """Load the latest checkpoint for a given run"""
    results_dir = Path(f"results/{run_name}")
    if not results_dir.exists():
        return None, set()
        
    # Find the latest checkpoint file
    checkpoint_files = list(results_dir.glob("results_checkpoint_*.csv"))
    if not checkpoint_files:
        return None, set()
        
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    
    # Load the results
    df = pd.read_csv(latest_checkpoint)
    
    # Create a set of completed experiment parameters
    completed_experiments = {
        (row['random_seed'], row['missing_percentage'], 
         row['distance_metric'], row['spatial_weight'], row['k'])
        for _, row in df.iterrows()
    }
    
    return df.to_dict('records'), completed_experiments