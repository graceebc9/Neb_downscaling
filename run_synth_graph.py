# use env xarr 

import pandas as pd 

from dataclasses import dataclass
from typing import List, Optional
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np 

@dataclass
class ExperimentParams:
    # Region information
    region_codes: List[str]
    region_name: Optional[str] = None
    region_type: str = 'ldcd'
    training_codes: Optional[ List[str]] = None
    testing_codes: Optional[ List[str]] = None
    graph_setting: Optional[str] = None 
    
    # Target and features
    target_column: str = 'total_gas'
    feature_columns: List[str] = field(default_factory=lambda: None)
    
    # Model parameters
    random_seeds: List[int] = field(default_factory=lambda: None)
    k_neighbors: List[int] = field(default_factory=lambda: None)
    distance_method: List[str] = field(default_factory=lambda: ['adaptive'])
    spatial_weight:  List[str] = field(default_factory=lambda: [0.7]) 
    missing_data_percent:  List[str] = field(default_factory=lambda: [0.2])
    distance_metrics: List[str] = field(default_factory=lambda: ['euclidean'])

    def __post_init__(self):
        # Set region name if not provided
        if self.region_name is None:
            self.region_name = '_'.join(self.region_codes)
            self.region_name = self.region_name + '_' + str(np.random.randint(200) )
        
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
 



 

import os 
import sys
sys.path.append('/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/src')
sys.path.append('../')
from src.run_exp import  run_graph_config

all_codes = ['train', 'test']
test_codes= ['test']    

feature_columns = [ 'all_res_total_fl_area_H_total',
 'Pre 1919_pct',
 'Standard size semi detached',
 'HDD_winter',
 'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
 'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',]


def run_graph_synth( folder, op_folder):
    for diff in ['hardest', 'hard', 'medium', 'easy', 'easiest']:
        print('starting ', diff)
        custom_config = ExperimentParams(
            region_codes= all_codes , 
            testing_codes = test_codes , 
            target_column='total_gas',
            k_neighbors=[4, 5, 6, 7, 10,15,25,35],
            region_type = 'ldcd',
            region_name =f'OOD_test_{diff}' , 
            spatial_weight = [0, 0.1, 0.2,  0.5,  0.7 ],
            missing_data_percent = ['NaN'] ,
            graph_setting = 'seperate' , 

            feature_columns=feature_columns
    )
        data = pd.read_csv(os.path.join(folder, f'OOD_{diff}_testtrain.csv') ) 
        os.makedirs(op_folder , exist_ok=True)
        output_dir = run_graph_config(data , custom_config, output_path=op_folder)


if __name__ == "__main__":
    run_graph_synth(folder = '/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/OOD_Data/mega_large', op_folder= '/rds/user/gb669/hpc-work/energy_map/Neb_downscaling/results/ood/mega_large')