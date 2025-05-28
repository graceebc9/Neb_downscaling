#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import time
import pandas as pd
import psutil
import gc
from itertools import product
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment_runner.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ExperimentRunner")

 

from src.run_exp import run_graph_config
 

from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional

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

def log_memory_usage(label=""):
    """Log the current memory usage with an optional label."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    logger.info(f"[MEMORY] {label} - RAM used: {memory_mb:.2f} MB")
    return memory_mb

def run_experiment(
    base_dir: str,
    seed: int,
    size: str,
    difficulty: str,
    output_base: str,
    timeout_seconds: int,
    feature_columns: List[str]
):
    """
    Run a single experiment with the given parameters
    
    Args:
        base_dir: Base directory containing the datasets
        seed: Random seed for the experiment
        size: Size of the dataset (small, medium, large)
        difficulty: Difficulty level (easiest, easy, medium, hard, hardest)
        output_base: Base directory for outputs
        timeout_seconds: Timeout for the experiment
        feature_columns: Feature columns to use
    """
    # Log experiment parameters
    logger.info(f"Running experiment: seed={seed}, size={size}, difficulty={difficulty}")
    
    # Create paths
    data_dir = os.path.join(base_dir, f"seed_{seed}", size)
    output_dir = os.path.join(output_base, f"seed_{seed}", size, difficulty)
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data exists
    train_file = os.path.join(data_dir, f"OOD_{difficulty}_train.csv")
    test_file = os.path.join(data_dir, f"OOD_{difficulty}_test.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        logger.error(f"Missing data files for seed={seed}, size={size}, difficulty={difficulty}")
        logger.error(f"Expected files: {train_file} and {test_file}")
        return False
    
    # Combine the train and test data for processing
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        print('combined_data cols: ', combined_data.columns.tolist() ) 
        # Add a 'train' or 'test' column if it doesn't exist
        if 'train' not in combined_data.columns:
            combined_data['train'] = [1] * len(train_data) + [0] * len(test_data)
            
        # Save combined data to a temporary file for processing
        combined_file = os.path.join(data_dir, f"OOD_{difficulty}_testtrain.csv")
        combined_data.to_csv(combined_file, index=False)
        logger.info(f"Created combined dataset at {combined_file}")
    except Exception as e:
        logger.error(f"Error processing data files: {e}")
        return False
    
    # Create experiment config
    all_codes = ['train', 'test']
    test_codes = ['test']
    
    custom_config = ExperimentParams(
        region_codes=all_codes, 
        testing_codes=test_codes, 
        target_column='total_gas',
        k_neighbors=[4, 5, 6, 7, 10, 15, 25, 35],
        region_type='ldcd',
        region_name=f'sample_{size}_OOD_test_{difficulty}_seed_{seed}', 
        spatial_weight=[0, 0.1, 0.2, 0.5, 0.7],
        missing_data_percent=['NaN'],
        graph_setting='seperate',  
        feature_columns=feature_columns,
        random_seeds=[seed]  # Use the provided seed
    )
    
    # Start memory tracking
    start_time = time.time()
    base_memory = log_memory_usage(f"Starting experiment {seed}_{size}_{difficulty}")
    
    try:
        # Run the experiment
        logger.info(f"Running experiment with combined data size: {combined_data.shape}")
        result_dir = run_graph_config(
            combined_data, 
            custom_config, 
            output_path=output_dir, 
            timeout_seconds=timeout_seconds
        )
        logger.info(f"Experiment completed successfully. Results in {result_dir}")
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return False
    finally:
        # Clean up
        del combined_data, train_data, test_data
        gc.collect()
        
        # Log final memory usage and execution time
        final_memory = log_memory_usage(f"Finished experiment {seed}_{size}_{difficulty}")
        memory_increase = final_memory - base_memory
        execution_time = time.time() - start_time
        
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        logger.info(f"Memory increase: {memory_increase:.2f} MB")
        
    return True

def get_timeouts(size, args):
        # Set custom timeout based on size
        custom_timeout = args.timeout
        if size == "large":
            custom_timeout = 600  # 10 minutes for large
        elif size == "medium":
            custom_timeout = 400  # Default for medium
        return custom_timeout

def main():
    """Main function to run multiple experiments"""
    parser = argparse.ArgumentParser(description="Run multiple graph experiments across seeds, sizes, and difficulties")
    
    # Add command-line arguments
    parser.add_argument("--base-dir", type=str, default="/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/ood_datasets",
                        help="Base directory containing the datasets")
    parser.add_argument("--output-dir", type=str, default="/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/results/multi_ood",
                        help="Base directory for outputs")
    parser.add_argument("--seeds", type=int, nargs="+", default= [42, 123, 7890, 55555, 987654],
                        help="Random seeds to use")
    parser.add_argument("--sizes", type=str, nargs="+", default=["small", "medium", "large"],
                        help="Dataset sizes to use")
    parser.add_argument("--difficulties", type=str, nargs="+", 
                        default=["easiest", "easy", "medium", "hard", "hardest"],
                        help="Difficulty levels to use")
    parser.add_argument("--timeout", type=int, default=400,
                        help="Timeout in seconds for each experiment")
    parser.add_argument("--single-experiment", action="store_true",
                        help="Run only a single experiment based on SLURM environment variables")
    
    args = parser.parse_args()
    
    # Define feature columns
    feature_columns = [
        'all_res_total_fl_area_H_total',
        'Pre 1919_pct',
        'Standard size semi detached',
        'HDD_winter',
        'ethnic_group_perc_White: English, Welsh, Scottish, Northern Irish or British',
        'economic_activity_perc_Economically active (excluding full-time students): In employment: Employee: Part-time',
    ]
    
    # Log system information
    total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
    available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
    logger.info(f"System memory - Total: {total_memory:.2f} GB, Available: {available_memory:.2f} GB")
    
    # Check if running as a SLURM array job
    if args.single_experiment:
        
        
        
        # Get parameters from environment variables
        array_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        
        if array_id == 0:
            logger.error("SLURM_ARRAY_TASK_ID not set")
            sys.exit(1)
            
        # Calculate total number of experiments
        total_exps = len(args.seeds) * len(args.sizes) * len(args.difficulties)
        
        if array_id > total_exps:
            logger.error(f"Invalid array_id: {array_id}. Must be between 1 and {total_exps}")
            sys.exit(1)
            
        # Calculate which experiment to run based on array_id
        array_id -= 1  # Convert to 0-based indexing
        
        # Get the experiment parameters for this array job
        seed_idx = array_id % len(args.seeds)
        size_idx = (array_id // len(args.seeds)) % len(args.sizes)
        diff_idx = array_id // (len(args.seeds) * len(args.sizes))
        
        seed = args.seeds[seed_idx]
        size = args.sizes[size_idx]
        difficulty = args.difficulties[diff_idx]
        
        custom_timeout = get_timeouts(size, args) 
        
        logger.info(f"Running single experiment with array_id={array_id+1}")
        logger.info(f"Parameters: seed={seed}, size={size}, difficulty={difficulty}")
        
        # Run single experiment
        success = run_experiment(
            args.base_dir, 
            seed, 
            size, 
            difficulty, 
            args.output_dir, 
           custom_timeout,
            feature_columns
        )
        
        if not success:
            logger.error("Experiment failed")
            sys.exit(1)
    else:
        # Run all experiments
        logger.info(f"Running {len(args.seeds) * len(args.sizes) * len(args.difficulties)} experiments")
        
        # Track overall results
        results = {
            "seed": [],
            "size": [],
            "difficulty": [],
            "success": [],
            "runtime": []
        }
        
        # Loop through all combinations
        for seed, size, difficulty in product(args.seeds, args.sizes, args.difficulties):
            start_time = time.time()
            
            # Set custom timeout based on size
            custom_timeout =  get_timeouts(size, args)
            
            # Run the experiment
            success = run_experiment(
                args.base_dir, 
                seed, 
                size, 
                difficulty, 
                args.output_dir, 
                custom_timeout,
                feature_columns
            )
            
            # Record results
            runtime = time.time() - start_time
            results["seed"].append(seed)
            results["size"].append(size)
            results["difficulty"].append(difficulty)
            results["success"].append(success)
            results["runtime"].append(runtime)
            
            logger.info(f"Experiment completed: seed={seed}, size={size}, difficulty={difficulty}, success={success}, runtime={runtime:.2f}s")
            
            # Force garbage collection between runs
            gc.collect()
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_path = os.path.join(args.output_dir, "experiment_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"All experiments completed. Results saved to {results_path}")
        
        # Print summary
        success_count = results_df["success"].sum()
        total_count = len(results_df)
        logger.info(f"Success rate: {success_count}/{total_count} ({100 * success_count / total_count:.1f}%)")

if __name__ == "__main__":
    main()