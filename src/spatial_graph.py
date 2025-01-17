import logging
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist


def spatial_neighbor_graph(df, k,  distance_metric,  lat_col='latitude', lon_col='longitude'):
    """
    Build a spatial neighbor graph from GIS data where nodes are connected based on geographic proximity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the GIS data with latitude and longitude columns
    lat_col : str, default='latitude'
        Name of the latitude column
    lon_col : str, default='longitude'
        Name of the longitude column
    k : int, default=10
        Number of nearest neighbors to consider for each point
    distance_metric : str, default='haversine'
        Distance metric to use. Options: 'haversine' (great-circle distance), 'euclidean'
    
    Returns:
    --------
    scipy.sparse.csr_matrix
        Adjacency matrix representing the spatial neighbor graph
    """
    logging.info(f"Building spatial graph using k={k} nearest neighbours.")
    
    # Input validation
    if not all(col in df.columns for col in [lat_col, lon_col]):
        msg = f"Missing required columns. Expected {lat_col} and {lon_col}. Have {df.columns}"  
        logging.error(msg)
        raise ValueError(msg)
    if k >= len(df):
        msg = f"k must be less than the number of points ({len(df)})"
        logging.error(msg)
        raise ValueError(msg)
    
    # Extract coordinates
    coords = df[[lat_col, lon_col]].values
    
    if distance_metric == 'haversine':
        # Convert lat/lon to radians for haversine distance
        coords_rad = np.radians(coords)
        
        # Custom distance matrix using haversine formula
        def haversine_distance(X):
            # Radius of Earth in kilometers
            R = 6371.0
            
            # Broadcast to compute differences between all pairs
            lat1, lon1 = X[:, 0:1], X[:, 1:2]
            lat2, lon2 = X[:, 0], X[:, 1]
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            # Haversine formula
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        # Create distance matrix
        distances = haversine_distance(coords_rad)
        
    else:
        # Use Euclidean distance
        distances = cdist(coords, coords, metric='euclidean')
    
    # Create k-nearest neighbors graph
    a = kneighbors_graph(distances, k, mode='distance', include_self=False)
    
    # Make the graph symmetric (undirected)
    a = a + a.T
    
    # Binarize the graph (1 for connected, 0 for not connected)
    a = (a > 0).astype(int)
    
    # Remove double edges
    a[a > 1] = 1
    
    return a
