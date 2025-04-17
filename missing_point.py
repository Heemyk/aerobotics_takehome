# missing_point.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
from pyproj import Transformer



# -----------------------------------------Polygon Handlers----------------------------------------------------


def parse_polygon_string(polygon_str):
    """
    Parse a polygon string of format 'lng1,lat1 lng2,lat2 ...' into a list of (x, y) UTM coordinates.
    Args:
        polygon_str: String of comma-separated lng,lat pairs, space-separated.
    Returns:
        List of (x, y) tuples in UTM coordinates.
    """
    # Initialize transformer: WGS84 (GPS) to UTM Zone 34S (adjust zone if needed)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32734")
    
    # Split the string into coordinate pairs
    coord_pairs = polygon_str.strip().split()
    
    # Parse each pair into (lng, lat) and convert to UTM
    boundary_utm = []
    for pair in coord_pairs:
        lng, lat = map(float, pair.split(','))
        x, y = transformer.transform(lat, lng)  # Note: transformer takes (lat, lng)
        boundary_utm.append((x, y))
        
    return Polygon(boundary_utm)

def create_cushioned_polygon(polygon: Polygon, buffer_distance: float) -> Polygon:
    """
    Shrink the polygon inward by some buffer (e.g., 1 meter or 10% of bbox width)
    Args:
        polygon: Polygon
        buffer_distance: Float of buffer distance to inset cushion by
    Returns:
        Polygon of cushioned polygon
    """
    return polygon.buffer(-buffer_distance)






def estimate_grid_axes(coords):
    # Dumb method for stable axis finding: try angles and pick one with max histogram sharpness
    angles = np.linspace(0, np.pi, 180)
    best_score = -np.inf
    best_basis = None

    for theta in angles:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        projected = coords @ R.T
        score = sum(histogram_sharpness(projected[:, i]) for i in range(2))
        if score > best_score:
            best_score = score
            best_basis = R

    return best_basis


def histogram_sharpness(coord_1d, num_bins=100):
    hist, _ = np.histogram(coord_1d, bins=num_bins)
    return np.sum(hist**2)



def cluster_axis(values, spacing_threshold):
    """Cluster 1D projected coordinates using agglomerative clustering."""
    values = np.sort(values).reshape(-1, 1)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=spacing_threshold,
        linkage='single'
    )
    labels = clustering.fit_predict(values)
    
    centroids = np.array([values[labels == i].mean() for i in np.unique(labels)])
    return np.sort(centroids)

def find_missing_trees(tree_df: pd.DataFrame, polygon: Polygon, debug: bool = True):
    
    # Coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32734")
    tree_df['x'], tree_df['y'] = transformer.transform(tree_df['lat'], tree_df['lng'])
    coords = tree_df[["x", "y"]].values

    # 1. Estimate stable axes
    grid_basis = estimate_grid_axes(coords)

    # 2. Project to grid space
    origin = coords.mean(axis=0)
    projected = (coords - origin) @ grid_basis.T

    # 3. Estimate row/column positions using histogram clustering
    spacing_est = np.median(np.diff(np.sort(projected[:, 0])))
    row_centroids = cluster_axis(projected[:, 0], spacing_threshold=spacing_est * 10)

    spacing_est = np.median(np.diff(np.sort(projected[:, 1])))
    col_centroids = cluster_axis(projected[:, 1], spacing_threshold=spacing_est * 10)

    print(f"Length of centroids is rows: {len(row_centroids)}; columns: {len(col_centroids)}")

    # 4. Snap existing trees to closest row/col centroid
    def snap_to_centroids(val, centroids):
        return centroids[np.argmin(np.abs(centroids - val))]

    snapped_coords = set()
    for x_proj, y_proj in projected:
        snapped_x = snap_to_centroids(x_proj, row_centroids)
        snapped_y = snap_to_centroids(y_proj, col_centroids)
        snapped_coords.add((snapped_x, snapped_y))

    # 5. Generate full grid of candidate points (row × column)
    all_grid_points = set((x, y) for x in row_centroids for y in col_centroids)

    # 6. Find missing grid locations
    missing_projected_coords = list(all_grid_points - snapped_coords)

    # 7. Convert to world space
    missing_world_coords = [grid_basis.T @ np.array([x, y]) + origin for x, y in missing_projected_coords]

    # 8. Filter: inside polygon and not too close to existing trees
    existing_kdtree = KDTree(coords)
    row_spacing = np.median(np.diff(row_centroids)) if len(row_centroids) > 1 else 1
    col_spacing = np.median(np.diff(col_centroids)) if len(col_centroids) > 1 else 1
    min_dist = 0.75 * min(row_spacing, col_spacing)
    max_dist = 1.8 * max(row_spacing, col_spacing)

    bbox = polygon.bounds  # (minx, miny, maxx, maxy)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    buffer_distance = min(width, height) * 0.043  # e.g., 5% inward
    cushioned_poly = create_cushioned_polygon(polygon, buffer_distance)

    inverse_transformer = Transformer.from_crs("EPSG:32734", "EPSG:4326")


    filtered_missing = []
    for x, y in missing_world_coords:
        if cushioned_poly.contains(Point(x, y)):
            dist, _ = existing_kdtree.query([x, y])
            if min_dist < dist < max_dist:
                # filtered_missing.append({"x": x, "y": y})
                lat, lng = inverse_transformer.transform(x, y)  # be careful of order
                filtered_missing.append({"lat": lat, "lng": lng})

    
    # 9. Debug plot
    if debug:
        plt.figure(figsize=(8, 8))
        plt.scatter(
            tree_df["lng"], 
            tree_df["lat"], 
            c='blue', s=10, alpha=0.6, label="Existing Trees"
        )
        # plt.scatter(np.array(missing_world_coords)[:,0], np.array(missing_world_coords)[:,1], c='green', s=10, alpha=0.6, label="Existing Trees")
        plt.scatter(
            [pt["lng"] for pt in filtered_missing],
            [pt["lat"] for pt in filtered_missing],
            c='red', s=12, alpha=0.8, label="Missing Trees"
        )
        # Transform polygon exteriors to lat/lng for plotting
        poly_x, poly_y = polygon.exterior.xy
        poly_lat, poly_lng = inverse_transformer.transform(poly_x, poly_y)
        plt.plot(poly_lng, poly_lat, 'k--', linewidth=2, label="Orchard Boundary")
        c_poly_x, c_poly_y = cushioned_poly.exterior.xy
        c_poly_lat, c_poly_lng = inverse_transformer.transform(c_poly_x, c_poly_y)
        plt.plot(c_poly_lng, c_poly_lat, 'g--', linewidth=2, label="Cushion")
        plt.title("Missing Tree Detection (Grid Histogram-Based)")
        plt.legend()
        plt.gca().set_aspect("equal")
        plt.grid(True)
        plt.show()

    

    return filtered_missing

