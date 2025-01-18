import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import logging


def haversine_distance(X):
            R = 6371.0  # Earth radius in kilometers
            lat1, lon1 = X[:, 0:1], X[:, 1:2]
            lat2, lon2 = X[:, 0], X[:, 1]
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c

def spatial_feature_neighbor_graph(df, k, feature_cols, spatial_weight, distance_metric, lat_col='latitude', lon_col='longitude'):
    """
    Build a spatial neighbor graph that considers both geographic proximity and feature similarity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the GIS data with latitude, longitude and feature columns
    feature_cols : list
        List of column names containing the features to consider for similarity
    lat_col : str, default='latitude'
        Name of the latitude column
    lon_col : str, default='longitude'
        Name of the longitude column
    k : int, default=10
        Number of nearest neighbors to consider for each point
    spatial_weight : float, default=0.7
        Weight given to spatial distance (1 - spatial_weight = feature weight)
    distance_metric : str, default='haversine'
        Distance metric to use for spatial distance. Options: 'haversine', 'euclidean'
    
    Returns:
    --------
    scipy.sparse.csr_matrix
        Adjacency matrix representing the combined spatial-feature neighbor graph
    """
    logging.info(f"Building spatial-feature graph using k={k} nearest neighbours.")
    # Extract coordinates and features
    coords = df[[lat_col, lon_col]].values
    features = df[feature_cols].values
   
    # Input validation

    required_cols = [lat_col, lon_col] + feature_cols
    if not all(col in df.columns for col in required_cols):
        msg = f"Missing required columns. Expected {required_cols}. Have {df.columns}"
        logging.error(msg)
        raise ValueError(msg)
    
    if k >= len(df):
        msg = f"k must be less than the number of points ({len(df)})"
        logging.error(msg)
        raise ValueError(msg)
        
    if not 0 <= spatial_weight <= 1:
        msg = "spatial_weight must be between 0 and 1"
        logging.error(msg)
        raise ValueError(msg)
    
    # Check for NaN values in input data
    if np.any(np.isnan(coords)) or np.any(np.isnan(features)):
        msg = "Input data contains NaN values. Please handle missing values before creating the graph."
        logging.error(msg)
        raise ValueError(msg)
    


    # Check if points are completely identical (both spatial and features)
    if (np.all(coords == coords[0]) and np.all(features == features[0])):
        msg = "All points have identical coordinates and features. Cannot create meaningful graph."
        logging.warning(msg)
        raise ValueError(msg)
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Calculate spatial distances
    if distance_metric == 'haversine':
        coords_rad = np.radians(coords)
        
        spatial_distances = haversine_distance(coords_rad)
    else:
        spatial_distances = cdist(coords, coords, metric='euclidean')

    # Replace any NaN values in spatial distances with large values
    spatial_distances = np.nan_to_num(spatial_distances, nan=np.nanmax(spatial_distances) if np.any(~np.isnan(spatial_distances)) else 1.0)
    
    # Calculate feature distances using cosine similarity
    feature_distances = cdist(features_normalized, features_normalized, metric='cosine')
    # Replace any NaN values in feature distances with large values
    feature_distances = np.nan_to_num(feature_distances, nan=np.nanmax(feature_distances) if np.any(~np.isnan(feature_distances)) else 1.0)
    
    # If all spatial distances are zero but features differ, 
    # rely entirely on feature distances by setting spatial_weight to 0
    if np.all(spatial_distances == 0) and not np.all(feature_distances == 0):
        logging.info("All spatial distances are zero but features differ. Using only feature distances.")
        spatial_weight = 0

    # Normalize distance matrices to [0, 1] range
    spatial_distances = spatial_distances / np.max(spatial_distances)
    feature_distances = feature_distances / np.max(feature_distances)
    
    # Combine spatial and feature distances
    combined_distances = (spatial_weight * spatial_distances + 
                        (1 - spatial_weight) * feature_distances)
    
    # Create k-nearest neighbors graph
    a = kneighbors_graph(combined_distances, k, mode='distance', include_self=False)
    
    # Make the graph symmetric (undirected)
    a = a + a.T
    
    # Binarize the graph (1 for connected, 0 for not connected)
    a = (a > 0).astype(int)
    
    # Remove double edges
    a[a > 1] = 1
    
    return a

def calculate_edge_weights(df, adj_matrix, feature_cols, lat_col='latitude', lon_col='longitude',
                         spatial_weight=0.7, distance_metric='haversine'):
    """
    Calculate edge weights for existing connections in the graph based on 
    combined spatial and feature distances.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the GIS data with latitude, longitude and feature columns
    adj_matrix : scipy.sparse.csr_matrix
        Adjacency matrix representing the graph structure
    feature_cols : list
        List of column names containing the features to consider for similarity
    lat_col, lon_col : str
        Names of the latitude and longitude columns
    spatial_weight : float
        Weight given to spatial distance in the combined metric
    distance_metric : str
        Distance metric to use for spatial distance
        
    Returns:
    --------
    scipy.sparse.csr_matrix
        Weighted adjacency matrix
    """
    # Extract coordinates and features
    coords = df[[lat_col, lon_col]].values
    features = df[feature_cols].values
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Calculate spatial distances
    if distance_metric == 'haversine':
        coords_rad = np.radians(coords)
        spatial_distances = haversine_distance(coords_rad)
    else:
        spatial_distances = cdist(coords, coords, metric='euclidean')
    
    # Calculate feature distances
    feature_distances = cdist(features_normalized, features_normalized, metric='cosine')
    
    # Normalize distances
    spatial_distances = spatial_distances / np.max(spatial_distances)
    feature_distances = feature_distances / np.max(feature_distances)
    
    # Calculate combined distances
    combined_distances = (spatial_weight * spatial_distances + 
                        (1 - spatial_weight) * feature_distances)
    
    # Create weighted adjacency matrix
    weighted_adj = adj_matrix.multiply(combined_distances)
    
    return weighted_adj