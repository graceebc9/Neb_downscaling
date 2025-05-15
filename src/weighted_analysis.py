import numpy as np 
from src.feature_graph import spatial_feature_neighbor_graph
from sklearn.metrics import mean_squared_error, r2_score
from src.diffusion import graph_prop

def run_weighted_analysis(df, target_col, k=35, spatial_weight=0.3, 
                         percent_missing=0.2, weighting_scheme='inverse'):
    """
    Run complete analysis with weighted graph.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Data with features and target
    target_col : str
        Column to predict
    k : int
        Number of neighbors
    spatial_weight : float
        Weight for spatial vs feature distances
    percent_missing : float
        Percentage of data to mask
    weighting_scheme : str
        'inverse' or 'gaussian'
    """
    # Create mask for missing values
    n_samples = len(df)
    n_missing = int(n_samples * percent_missing)
    mask = np.ones(n_samples)
    mask[np.random.choice(n_samples, n_missing, replace=False)] = 0
    
    # Create weighted adjacency
    feature_cols = ['avg_gas', 'all_types_']  
    adj = create_weighted_adj(
        df, k, feature_cols, 
        spatial_weight=spatial_weight,
        weighting_scheme=weighting_scheme
    )
    
    # Run propagation
    target_values = df[target_col].values

    completed = graph_prop(adj=adj, gappy_tens=target_values, omega=mask, 
                     thresh=0.001, iterative=True)
    
    return completed, mask




def create_weighted_adj(df, k, feature_cols, spatial_weight, weighting_scheme='inverse'):
    """
    Create weighted adjacency matrix.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Contains spatial and feature data
    k : int
        Number of neighbors
    feature_cols : list
        Feature columns to use
    spatial_weight : float
        Weight for spatial vs feature distances
    weighting_scheme : str
        'inverse' or 'gaussian'
    
    Returns:
    --------
    scipy.sparse.csr_matrix
        Weighted adjacency matrix
    """
    # Get distance-based adjacency matrix (not binary)
    adj_matrix = spatial_feature_neighbor_graph(
        df, k, feature_cols, 
        spatial_weight=spatial_weight,
        distance_method='adaptive',
        distance_metric ='euclidean',
        binary=False  # Important: keep the distances
    )
    
    # Convert distances to weights based on scheme
    if weighting_scheme == 'inverse':
        # Transform distances to similarities: smaller distance = larger weight
        adj_matrix.data = 1 / (1 + adj_matrix.data)
    elif weighting_scheme == 'gaussian':
        # Gaussian kernel weighting
        sigma = np.mean(adj_matrix.data)  # adaptive bandwidth
        adj_matrix.data = np.exp(-(adj_matrix.data ** 2) / (2 * sigma ** 2))
    
    # Normalize weights to [0,1]
    adj_matrix.data = adj_matrix.data / adj_matrix.data.max()
    
    return adj_matrix



def calculate_metrics(y_true, y_pred, mask=None):
    """
    Calculate RMSE and R2 for predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    mask : array-like, optional
        If provided, calculate metrics only where mask == 0 (held-out data)
        
    Returns:
    --------
    dict
        Dictionary with RMSE and R2 scores
    """
    if mask is not None:
        y_true = y_true[mask == 0]
        y_pred = y_pred[mask == 0]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2
    }
