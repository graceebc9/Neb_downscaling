import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-8
MIN_POINTS = 100


def haversine_distance(X):
    """Calculate haversine distance for geographic coordinates in radians."""
    R = 6371.0  # Earth radius in kilometers
    lat1, lon1 = X[:, 0:1], X[:, 1:2]
    lat2, lon2 = X[:, 0], X[:, 1]
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_local_density(spatial_distances, k):
    """
    Calculate local density using k nearest neighbors.
    
    Parameters:
    -----------
    spatial_distances : np.ndarray
        Matrix of spatial distances
    k : int
        Number of neighbors for density calculation
    
    Returns:
    --------
    np.ndarray
        Local density for each point
    """
    # Add epsilon to prevent division by zero
    spatial_distances = spatial_distances + EPSILON
    
    # Get k nearest neighbor distances
    local_radii = np.sort(spatial_distances, axis=1)[:, k]
    
    # Calculate density (points per unit area)
    local_densities = k / (np.pi * local_radii**2)
    
    # Clip densities to reasonable range
    return np.clip(local_densities, 0.1, 10)

def compute_combined_distances(spatial_distances, feature_distances, spatial_weight, 
                             method='linear', k_density=3):
    """
    Compute combined distances using either linear or adaptive weighting.
    
    Parameters:
    -----------
    spatial_distances : np.ndarray
        Matrix of spatial distances
    feature_distances : np.ndarray
        Matrix of feature distances
    spatial_weight : float
        Base weight for spatial distances (0-1)
    method : str
        'linear' or 'adaptive'
    k_density : int
        Number of neighbors for density in adaptive method
    """
    # Check for very small spatial distances
    if np.all(spatial_distances < EPSILON):
        logger.info("All spatial distances very small. Using only feature distances.")
        return feature_distances
    
    # Add epsilon to spatial distances
    spatial_distances = spatial_distances + EPSILON
    
    # Normalize distances to [0, 1]
    spatial_distances = spatial_distances / np.max(spatial_distances)
    feature_distances = feature_distances / np.max(feature_distances)
    
    if method == 'linear':
        return ((spatial_weight * spatial_distances) + 
                (1 - spatial_weight) * feature_distances)
    
    elif method == 'adaptive':
        # Calculate local density for each point
        local_densities = calculate_local_density(spatial_distances, k=k_density)
        
        # Log density statistics
        logger.debug(f"Density stats - Mean: {np.mean(local_densities):.3f}, "
                    f"Std: {np.std(local_densities):.3f}")
        
        # Compute adaptive weights for each point
        adaptive_weights = spatial_weight / (1 + local_densities.reshape(-1, 1))
        
        # Log weight adjustments
        logger.debug(f"Weight adjustment - Mean: {np.mean(adaptive_weights):.3f}, "
                    f"Min: {np.min(adaptive_weights):.3f}")
        
        # Compute distances with point-specific weights
        combined_distances = np.zeros_like(spatial_distances)
        for i in range(len(spatial_distances)):
            combined_distances[i] = (adaptive_weights[i] * spatial_distances[i] + 
                                   (1 - adaptive_weights[i]) * feature_distances[i])
        return combined_distances
    
    else:
        raise ValueError(f"Unknown method: {method}")

def spatial_feature_neighbor_graph(df, k, feature_cols, spatial_weight, 
                                 distance_method='linear', distance_metric='haversine',
                                 lat_col='latitude', lon_col='longitude', k_density=3, debug = False ):
    """
    Build a spatial neighbor graph considering both geographic proximity and feature similarity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing GIS data
    k : int
        Number of nearest neighbors for final graph
    feature_cols : list
        List of feature column names
    spatial_weight : float
        Weight given to spatial distance (0-1)
    distance_method : str
        'linear' or 'adaptive'
    distance_metric : str
        'haversine' or 'euclidean' - for spatial distance metric only 
    lat_col, lon_col : str
        Names of latitude and longitude columns
    k_density : int
        Number of neighbors for density calculation
    
    Returns:
    --------
    scipy.sparse.csr_matrix
        Binary adjacency matrix of the neighbor graph
    """
    # Validate inputs
    if len(df) < MIN_POINTS:
        raise ValueError(f"Dataset too small. Need at least {MIN_POINTS} points.")
    
    if k_density >= k:
        raise ValueError("k_density must be less than k")
    
    if k >= len(df):
        raise ValueError(f"k ({k}) must be less than n_points ({len(df)})")
    
    if not 0 <= spatial_weight <= 1:
        raise ValueError("spatial_weight must be between 0 and 1")
    
    # Check columns
    required_cols = [lat_col, lon_col] + feature_cols
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing columns. Need {required_cols}")
    
    # Extract and check data
    coords = df[[lat_col, lon_col]].values
    features = df[feature_cols].values
    
    if np.any(np.isnan(coords)) or np.any(np.isnan(features)):
        raise ValueError("Input contains NaN values")
    
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
    feature_distances = cdist(features_normalized, features_normalized, 
                            metric='euclidean')
    
    # Check for spatial clustering
    spatial_range = np.max(spatial_distances) - np.min(spatial_distances)
    if spatial_range < EPSILON:
        logger.info(f"Spatial range ({spatial_range:.2e}) very small. "
                   "Reducing spatial weight influence.")
        spatial_weight *= 0.1
    
    # Compute combined distances
    combined_distances = compute_combined_distances(
        spatial_distances, feature_distances,
        spatial_weight, method=distance_method,
        k_density=k_density
    )
    
    # Create k-nearest neighbors graph
    adj_matrix = kneighbors_graph(combined_distances, k, mode='distance',
                                 include_self=False)
    
    # Make symmetric and binary
    adj_matrix = (adj_matrix + adj_matrix.T > 0).astype(int)
    
    # Log graph statistics
    n_edges = adj_matrix.sum() // 2
    avg_degree = adj_matrix.sum() / len(df)
    logger.info(f"Graph created with {n_edges} edges. "
               f"Average degree: {avg_degree:.2f}")
    
    if debug: 
        debug_info = {}
        debug_info['spatial_distances'] = spatial_distances
        debug_info['feature_distances_norm'] = feature_distances
        debug_info['combined_distances'] = combined_distances
        debug_info['coords'] = coords
        
        return adj_matrix, debug_info
    
    return adj_matrix