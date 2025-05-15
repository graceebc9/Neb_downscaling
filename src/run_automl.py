from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import os 

def load_data(custom_config, df_path, pc_path, train=None):    
    import pandas as pd
    df = pd.read_csv(df_path) 
    if custom_config.region_type =='ldcd':
        df = df[df['ladcd'].isin(custom_config.region_codes)]
    elif custom_config.region_type =='region':
        df =df[df['region'].isin(custom_config.region_codes)]
    elif custom_config.region_type =='ldcd_seperate':
        if train == 'y':
            df =df[df['ladcd'].isin(custom_config.training_codes)]
        elif train== 'n':
            df =df[df['ladcd'].isin(custom_config.testing_codes)]
    return df 
    
def run_automl(df, config, output_directory, test_size, time_limit=1000):
    
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
            
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_seed)
    train_data = transform(TabularDataset(train_data), label, config.feature_columns)
 
        
    
    size_train = len(train_data) 
    predictor = TabularPredictor(label, path=output_directory).fit(train_data, 
                                                                time_limit=time_limit,
                                                                presets=model_preset,
                                                                 )
    
    test_data = transform(TabularDataset(test_data), label, config.feature_columns)
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