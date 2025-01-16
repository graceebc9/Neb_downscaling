import os
import scipy.sparse as sp
import hashlib
import json

def generate_graph_id(graph_params):
    """
    Generate a unique identifier for a graph based on its parameters.
    
    Parameters:
    -----------
    graph_params : dict
        Dictionary containing all parameters that define the graph
        
    Returns:
    --------
    str
        A unique hash string representing the graph parameters
    """
    # Sort the parameters to ensure consistent ordering
    param_str = json.dumps(graph_params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:10]

def get_graph(dataset_name, geo_df, graph_fn, graph_params=None, cache_dir='graphs'):
    """
    Get or create a graph using the specified graph generation function and parameters.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    geo_df : pandas.DataFrame
        DataFrame containing the geographic data
    graph_fn : callable
        Function that generates the graph. Should accept geo_df and **graph_params
    graph_params : dict, optional
        Dictionary of parameters to pass to the graph generation function
        Example: {'k': 5, 'distance_metric': 'haversine', 'feature_cols': ['temp']}
    cache_dir : str, default='graphs'
        Directory to store cached graphs
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        The adjacency matrix of the graph
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Initialize graph parameters
    if graph_params is None:
        graph_params = {}
    
    # Create a unique identifier for this graph configuration
    params_for_id = {
        'dataset': dataset_name,
        'n_rows': len(geo_df),
        'graph_fn': graph_fn.__name__,
        **graph_params
    }
    graph_id = generate_graph_id(params_for_id)
    
    # Define the cache file path
    cache_file = os.path.join(cache_dir, f"{graph_id}.npz")
    params_file = os.path.join(cache_dir, f"{graph_id}_params.json")
    
    # Check if cached graph exists
    if os.path.exists(cache_file) and os.path.exists(params_file):
        # Verify the parameters match
        with open(params_file, 'r') as f:
            cached_params = json.load(f)
            
        if cached_params == params_for_id:
            print(f"Loading cached graph: {cache_file}")
            return sp.load_npz(cache_file)
    
    # Generate new graph
    print(f"Generating new graph with {graph_fn.__name__}")
    graph = graph_fn(geo_df, **graph_params)
    
    # Cache the graph and its parameters
    sp.save_npz(cache_file, graph)
    with open(params_file, 'w') as f:
        json.dump(params_for_id, f, indent=2)
        
    return graph
