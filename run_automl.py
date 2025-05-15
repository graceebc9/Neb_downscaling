import sys
import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor
from src.cols import feature_cols

def check_directory_and_files(output_directory, required_files):
    """
    Check if the specified directory exists and contains the required files.
    
    Parameters:
        output_directory (str): The path to the directory to check.
        required_files (list): A list of filenames expected in the directory.
    
    Returns:
        bool: True if the directory exists and contains all required files, False otherwise.
    """
    # Check if the directory exists
    if not os.path.exists(output_directory):
        print(f"Directory {output_directory} does not exist. Will be created.")
        return False

    # Check for the presence of all required files in the directory
    missing_files = [file for file in required_files if not os.path.isfile(os.path.join(output_directory, file))]
    if missing_files:
        print(f"Missing files in {output_directory}: {', '.join(missing_files)}")
        return False
    
    return True
    

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


def main():
    
    random_seed = int(os.environ.get('RANDOM_SEED'))
    output_path = '/home/gb669/rds/hpc-work/energy_map/Neb_downscaling/results/automl'
    time_limit = int(os.environ.get('TIME_LIM'))
    model_preset = os.environ.get('MODEL_PRESET')
    target = os.environ.get('TARGET')
    if target == 'totalelec':   
        label = 'total_elec'
    elif target == 'totalgas':
        label = 'total_gas'
    else:
        label = 'total_gas'
    
    excl_models = []
    

    cols = feature_cols
    col_names='feature_cols'
    feat=False 
    
    data_path = '/home/gb669/rds/hpc-work/energy_map/data/automl_models/input_data/new_final/NEBULA_englandwales_domestic_filtered.csv'
    df = pd.read_csv(data_path)   
    dataset_name = os.path.basename(data_path).split('.')[0].split('_tr')[0]
    
    filt_type = 'ladcd'
    loc_type= filt_type
    codes=['E08000025','E06000052'] 
    code = 'E08000025_E06000052'
    if filt_type == 'ladcd': 
        df = df[df['ladcd'].isin(codes)].copy()
    elif filt_type =='region':
        df = df[df['region']==code].copy()

    
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=random_seed)
    train_data = transform(TabularDataset(train_data), label, cols)
        
    print(f'starting model run for {loc_type} target {label}, time lim {time_limit}, model preset {model_preset} and rs {random_seed}' )


    
    output_directory = f"{output_path}/{dataset_name}__{loc_type}__{label}__{time_limit}__{model_preset}___rs_{random_seed}_code_{code}_cols_{col_names}"
    required_files = ['model_summary.txt']  # List of files you expect to exist
    
    
    # Check if output directory exists and has all required files
    if check_directory_and_files(output_directory, required_files):
        print(f"Directory {output_directory} already contains all necessary files. Exiting to prevent data overwrite.")
        sys.exit(0)
    else:
        # Create directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        print(f"Directory {output_directory} is ready for use.")

    
 
        
    train_subset = train_data   
    size_train = len(train_subset) 
    predictor = TabularPredictor(label, path=output_directory).fit(train_subset, 
                                                                time_limit=time_limit,
                                                                presets=model_preset,
                                                                excluded_model_types=excl_models)
    
    test_data = transform(TabularDataset(test_data), label, cols)
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

    if feat==True: 
        pred = predictor.feature_importance(test_data)
        pred.to_csv(os.path.join(output_directory, 'feature_importance.csv'))



if __name__ == '__main__':
    main()



# export DATA_PATH='/Volumes/T9/Data_downloads/new-data-outputs/ml_input/final_V1_ml_data.csv'
# export OUTPUT_PATH='/Volumes/T9/Data_downloads/new-data-outputs/ml/results'
# export MODEL_PRESET='medium_quality'
# export TIME_LIM=5000
# export TRAIN_SUBSET_PROP=0.4
# export TARGET='totalgas'
# export COL_SETTING=0
# export RUN_REGIONAL='Yes'
# export run_census="No"
# export REGION_ID='NW'
# export RUN_GAS_FILTER='Yes'
# export GAS_THRESHOLD=5