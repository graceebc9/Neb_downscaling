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
 


# Predefined configurations
EXPERIMENT_CONFIGS = {
    'test': ExperimentParams(
        region_codes=['E08000025'],
        k_neighbors=[1, 3, 5]
    ),
    
    'full': ExperimentParams(
        region_codes=['E08000025', 'E06000052'],
        region_name='combined_regions',
        k_neighbors=[1, 3, 5, 7, 9, 11, 13, 15],
        spatial_weight=0.8,
        missing_data_percent=0.3
    )
}
 