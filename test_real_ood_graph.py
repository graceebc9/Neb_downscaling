import geopandas as gpd
import pandas as pd
import logging
from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional
from src.run_exp import run_graph_config

# Set up logging
logger = logging.getLogger(__name__)

# Read the shapefiles
df1 = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/ood_datasets/modern_trad/urban_modern.csv')
df2 = pd.read_csv('/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/ood_datasets/modern_trad/rural_traditional.csv')

seed = 42

def set_test_train(df1, df2):
    df1['ldcd'] = 'train'
    df2['ldcd'] = 'test'
    combo_df = pd.concat([df1, df2], ignore_index=True)  # Fixed: added brackets and ignore_index
    return combo_df

combined_data = set_test_train(df1, df2)  # Fixed: renamed variable for consistency
print('data shape')
print(combined_data.shape)

# Create experiment config
all_codes = ['train', 'test']
test_codes = ['test']

# You'll need to define these variables or import them:
# Define feature columns
feature_columns = [
    'all_res_total_fl_area_H_total',
    'Pre 1919_pct',
    'Standard size semi detached',
    'HDD_winter',
    'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
    'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Full-time',
]
print(combined_data[['latitude'] + feature_columns].head()) 
 
output_dir =  "/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/results/real_odd_graph"

timeout_seconds=1200

@dataclass
class ExperimentParams:
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


custom_config = ExperimentParams(
    region_codes=all_codes, 
    testing_codes=test_codes, 
    target_column='total_gas',
    k_neighbors=[ 5, 10, 15, 30],
    region_type='ldcd',
    region_name=f'train_urbmod_test_rur_trad__rs_{seed}',
    spatial_weight=[0, 0.2, 0.8],
    missing_data_percent=['NaN'],
    graph_setting='seperate', 
    feature_columns=feature_columns,
    random_seeds=[seed]
)
     
# Run the experiment
logger.info(f"Running experiment with combined data size: {combined_data.shape}")
result_dir = run_graph_config(
    combined_data, 
    custom_config, 
    output_path=output_dir, 
    timeout_seconds=timeout_seconds
)