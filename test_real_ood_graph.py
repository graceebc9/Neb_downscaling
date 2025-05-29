import geopandas as gpd
import pandas as pd
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional
from src.run_exp import run_graph_config

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
SEED = 42
OUTPUT_DIR = "/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/results/real_odd_graph"
TIMEOUT_SECONDS = 1200

# Feature columns
FEATURE_COLUMNS = [
    'all_res_total_fl_area_H_total',
    'Pre 1919_pct',
    'Standard size semi detached',
    'HDD_winter',
    'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
    'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
]

def load_data():
    """Load the datasets."""
    df1 = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/ood_datasets/modern_trad/urban_modern.csv')
    df2 = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/ood_datasets/modern_trad/rural_traditional.csv')
    return df1, df2

def create_train_test_split(df_train, df_test, train_label='train', test_label='test'):
    """Create combined dataset with train/test labels."""
    df_train = df_train.copy()
    df_test = df_test.copy()
    
    df_train['ldcd'] = train_label
    df_test['ldcd'] = test_label
    
    combined_df = pd.concat([df_train, df_test], ignore_index=True)
    return combined_df

@dataclass
class ExperimentParams:
    """Configuration for experiments."""
    # Region information
    region_codes: List[str]
    region_name: Optional[str] = None
    region_type: str = 'ldcd'
    training_codes: Optional[List[str]] = None
    testing_codes: Optional[List[str]] = None
    graph_setting: Optional[str] = None 
    
    # Target and features
    target_column: str = 'total_gas'
    feature_columns: List[str] = field(default_factory=lambda: None)
    
    # Model parameters
    random_seeds: List[int] = field(default_factory=lambda: None)
    k_neighbors: List[int] = field(default_factory=lambda: None)
    distance_method: List[str] = field(default_factory=lambda: ['adaptive'])
    spatial_weight: List[float] = field(default_factory=lambda: [0.7])
    missing_data_percent: List[str] = field(default_factory=lambda: [0.2])
    distance_metrics: List[str] = field(default_factory=lambda: ['euclidean'])
    
    def __post_init__(self):
        # Set region name if not provided
        if self.region_name is None:
            self.region_name = '_'.join(self.region_codes)
            self.region_name = self.region_name + '_' + str(np.random.randint(200))
        
        # Set default feature columns if not provided
        if self.feature_columns is None:
            self.feature_columns = [
                'all_res_total_fl_area_H_total',
                'Pre 1919_pct',
                'Standard size semi detached',
                'postcode_area',
                'HDD_winter'
            ]
        
        # Set default random seeds if not provided
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456, 789, 101]
        
        # Set default k_neighbors if not provided
        if self.k_neighbors is None:
            self.k_neighbors = [1, 3, 5, 7, 9]

def create_experiment_config(exp_name, test_codes, seed):
    """Create experiment configuration."""
    all_codes = ['train', 'test']
    
    return ExperimentParams(
        region_codes=all_codes, 
        testing_codes=test_codes, 
        target_column='total_gas',
        k_neighbors=[5, 10, 15, 30],
        region_type='ldcd',
        region_name=f'{exp_name}_rs_{seed}',
        spatial_weight=[0, 0.2, 0.8],
        missing_data_percent=['NaN'],
        graph_setting='seperate', 
        feature_columns=FEATURE_COLUMNS,
        random_seeds=[seed]
    )

def run_experiment(data, config, experiment_name):
    """Run a single experiment."""
    logger.info(f"Running {experiment_name} with data size: {data.shape}")
    print(f"\n=== {experiment_name} ===")
    print(f"Data shape: {data.shape}")
    print("Sample data:")
    print(data[['latitude'] + FEATURE_COLUMNS].head())
    
    result_dir = run_graph_config(
        data, 
        config, 
        output_path=OUTPUT_DIR, 
        timeout_seconds=TIMEOUT_SECONDS
    )
    
    logger.info(f"Completed {experiment_name}. Results saved to: {result_dir}")
    return result_dir

def main():
    """Main execution function."""
    # Load data
    df_urban_modern, df_rural_traditional = load_data()
    
    print("Loaded datasets:")
    print(f"Urban Modern: {df_urban_modern.shape}")
    print(f"Rural Traditional: {df_rural_traditional.shape}")
    
    # Experiment 1: Train on Urban Modern, Test on Rural Traditional
    print("\n" + "="*60)
    print("EXPERIMENT 1: Train Urban Modern → Test Rural Traditional")
    print("="*60)
    
    data_exp1 = create_train_test_split(
        df_urban_modern, 
        df_rural_traditional, 
        'train', 
        'test'
    )
    
    config_exp1 = create_experiment_config(
        'train_urbmod_test_rurtrad', 
        ['test'], 
        SEED
    )
    
    result_exp1 = run_experiment(data_exp1, config_exp1, "Experiment 1")
    
    # Experiment 2: Train on Rural Traditional, Test on Urban Modern
    print("\n" + "="*60)
    print("EXPERIMENT 2: Train Rural Traditional → Test Urban Modern")
    print("="*60)
    
    data_exp2 = create_train_test_split(
        df_rural_traditional, 
        df_urban_modern, 
        'train', 
        'test'
    )
    
    config_exp2 = create_experiment_config(
        'train_rurtrad_test_urbmod', 
        ['test'], 
        SEED
    )
    
    result_exp2 = run_experiment(data_exp2, config_exp2, "Experiment 2")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED")
    print("="*60)
    print(f"Experiment 1 results: {result_exp1}")
    print(f"Experiment 2 results: {result_exp2}")

if __name__ == "__main__":
    main()