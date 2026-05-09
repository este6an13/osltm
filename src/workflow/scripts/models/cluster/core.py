import numpy as np
from sklearn.cluster import DBSCAN

def extract_cluster_parameters(
    t_sec: np.ndarray, 
    method: str = 'dbscan', 
    eps: float = 60, 
    min_samples: int = 3, 
    target_size: int = 5
):
    """
    Extracts clusters and their properties from a 1D array of timestamps.
    
    Args:
        t_sec: Array of event timestamps (in seconds from some origin).
        method: Clustering method ('dbscan', 'dbscan_hybrid', 'kmeans', 'fixed_size').
        eps: DBSCAN only - Maximum distance between two samples.
        min_samples: DBSCAN only - Number of samples in a neighborhood.
        target_size: Target size for clusters (used by kmeans, fixed_size, and dbscan_hybrid).
        
    Returns:
        dict containing 'centroids', 'sizes', 'dispersions', 'noise'
    """
    if len(t_sec) == 0:
        return {
            'centroids': np.array([]),
            'sizes': np.array([]),
            'dispersions': np.array([]),
            'noise': np.array([])
        }
        
    centroids, sizes, dispersions, noise_points = [], [], [], []
    
    if method in ['dbscan', 'dbscan_hybrid']:
        X = t_sec.reshape(-1, 1)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: continue
            cluster_mask = (labels == label)
            cluster_points = t_sec[cluster_mask]
            
            # Hybrid logic: chunk oversized clusters
            if method == 'dbscan_hybrid' and len(cluster_points) > target_size:
                cluster_points_sorted = np.sort(cluster_points)
                for i in range(0, len(cluster_points_sorted), target_size):
                    chunk = cluster_points_sorted[i:i+target_size]
                    if len(chunk) == 0: continue
                    centroid = np.mean(chunk)
                    centroids.append(centroid)
                    sizes.append(len(chunk))
                    dispersions.extend(chunk - centroid)
            else:
                centroid = np.mean(cluster_points)
                centroids.append(centroid)
                sizes.append(len(cluster_points))
                dispersions.extend(cluster_points - centroid)
                
        noise_points = t_sec[labels == -1]
        
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        n_clusters = max(1, len(t_sec) // target_size)
        X = t_sec.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(X)
        labels = kmeans.labels_
        for label in range(n_clusters):
            cluster_points = t_sec[labels == label]
            if len(cluster_points) == 0: continue
            centroid = np.mean(cluster_points)
            centroids.append(centroid)
            sizes.append(len(cluster_points))
            dispersions.extend(cluster_points - centroid)
            
    elif method == 'fixed_size':
        t_sorted = np.sort(t_sec)
        for i in range(0, len(t_sorted), target_size):
            cluster_points = t_sorted[i:i+target_size]
            if len(cluster_points) == 0: continue
            centroid = np.mean(cluster_points)
            centroids.append(centroid)
            sizes.append(len(cluster_points))
            dispersions.extend(cluster_points - centroid)
            
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return {
        'centroids': np.array(centroids),
        'sizes': np.array(sizes),
        'dispersions': np.array(dispersions),
        'noise': np.array(noise_points)
    }

def simulate_cluster_process(params: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Simulates a Neyman-Scott / cluster process day.
    
    Args:
        params: dict with:
            - 'centroid_mu_blocks': NHPP intensity per bin for parent centroids (events/sec)
            - 'noise_mu_blocks': NHPP intensity per bin for background noise (events/sec)
            - 'cluster_size_mean': mean of Poisson distribution for cluster size
            - 'dispersion_std': std deviation of Normal distribution for offsets
            - 'dt_sec': duration of each bin in seconds (e.g., 900)
            - 'T_total': total duration of the observation window in seconds
        rng: numpy random generator
        
    Returns:
        Sorted array of event timestamps (in seconds from window start).
    """
    dt_sec = params['dt_sec']
    T_total = params['T_total']
    
    events = []
    
    # 1. Simulate background noise (NHPP)
    noise_mu = params.get('noise_mu_blocks', np.zeros(int(T_total / dt_sec)))
    noise_counts = rng.poisson(noise_mu * dt_sec).astype(int)
    for k, n in enumerate(noise_counts):
        if n > 0:
            t_start = k * dt_sec
            t_end = min((k + 1) * dt_sec, T_total)
            events.append(rng.uniform(t_start, t_end, size=n))
            
    # 2. Simulate parent centroids (NHPP)
    centroid_mu = params.get('centroid_mu_blocks', np.zeros(int(T_total / dt_sec)))
    centroid_counts = rng.poisson(centroid_mu * dt_sec).astype(int)
    
    cluster_size_mean = params.get('cluster_size_mean', 0.0)
    dispersion_std = params.get('dispersion_std', 0.0)
    
    for k, n in enumerate(centroid_counts):
        if n > 0:
            t_start = k * dt_sec
            t_end = min((k + 1) * dt_sec, T_total)
            
            # Parent locations
            parents = rng.uniform(t_start, t_end, size=n)
            
            # Number of children per parent (Poisson)
            # Alternatively could use Negative Binomial if variance was provided
            children_counts = rng.poisson(cluster_size_mean, size=n)
            
            for parent_t, num_children in zip(parents, children_counts):
                if num_children > 0:
                    # Offsets (Normal)
                    offsets = rng.normal(loc=0.0, scale=dispersion_std, size=num_children)
                    children_t = parent_t + offsets
                    
                    # Optional: Clamp children within the day [0, T_total]?
                    # In true cluster processes they can spill over, but for daily window we clamp or filter.
                    # We will filter them to keep only those within [0, T_total]
                    children_t = children_t[(children_t >= 0) & (children_t <= T_total)]
                    events.append(children_t)
                    
    if len(events) == 0:
        return np.array([])
        
    all_events = np.concatenate(events)
    all_events.sort()
    return all_events
