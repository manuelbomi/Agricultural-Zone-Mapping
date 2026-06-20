"""
03_management_zones.py
=======================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates data-driven management zone delineation using k-means clustering:
  - Generate simulated soil property raster data (pH, organic matter, EC)
  - Normalize and stack into a feature array
  - Apply k-means clustering to classify each grid cell into a zone
  - Convert the cluster raster to vector polygons using contour extraction
  - Export management zones as GeoJSON with cluster-averaged attribute values

Why k-means for management zones:
    K-means groups locations with similar soil properties into clusters,
    which become management zones. Each zone gets a uniform application rate
    calibrated to its mean soil properties.

    Compared to the simple grid approach (01_field_delineation.py):
      - K-means zones follow actual soil variability patterns
      - Irregular shapes that match real landscape features
      - Fewer zones needed to capture meaningful variability
      - Better agronomic response to variable rate applications

    Commercial precision ag software (SST Summit, Granular, Farmers Edge)
    uses variants of this approach, often adding elevation, NDVI history,
    and electrical conductivity (EC) survey layers.

Run:
    python 03_management_zones.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import unary_union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")
UTM_CRS = 'EPSG:32614'

# K-means configuration
N_ZONES = 4        # Number of management zones (clusters)
GRID_RES_M = 200   # Grid resolution for simulation (200m cells)
RANDOM_SEED = 42


def load_boundary() -> gpd.GeoDataFrame:
    """Load and project farm boundary."""
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(UTM_CRS)
    print(f"Farm boundary: {boundary.geometry.area.sum()/10_000:.1f} ha")
    return boundary


def simulate_soil_layers(
    boundary: gpd.GeoDataFrame,
    resolution_m: float
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict]:
    """
    Simulate multi-layer soil property data as raster grids.

    In production, these layers come from:
      - Soil electrical conductivity (EC) surveys (Veris, EM38)
      - Soil sampling interpolated to raster (kriging or IDW)
      - Remote sensing indices (NDVI history, bare soil NDVI)
      - Elevation-derived products (slope, curvature, TWI)

    We simulate spatially correlated fields using superposition of
    Gaussian basis functions — creating realistic patches of variation.

    Returns a dict of variable_name → 2D numpy array.
    """
    from scipy.ndimage import gaussian_filter

    farm_geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = farm_geom.bounds

    # Create coordinate grids
    x_arr = np.arange(minx, maxx, resolution_m)
    y_arr = np.arange(miny, maxy, resolution_m)
    xx, yy = np.meshgrid(x_arr, y_arr)
    nrows, ncols = xx.shape

    print(f"\nSoil layer grid: {nrows}×{ncols} = {nrows*ncols} cells "
          f"at {resolution_m}m resolution")

    np.random.seed(RANDOM_SEED)

    # Normalized coordinates (0-1)
    x_norm = (xx - minx) / (maxx - minx)
    y_norm = (yy - miny) / (maxy - miny)

    # --- pH: 6.0–7.5 with a gradient from west (acidic) to east (alkaline) ---
    ph_base = 6.2 + 1.0 * x_norm
    ph_noise = np.random.normal(0, 0.15, (nrows, ncols))
    ph = np.clip(gaussian_filter(ph_base + ph_noise, sigma=2), 5.8, 7.8)

    # --- Organic Matter: 1.5–4.0%, higher in north-center depression ---
    # Create a "low" zone in SW (sand) and a "high" zone in NE (clay)
    om_base = (1.8 + 2.0 * y_norm + 0.5 * (1 - x_norm)
               - 1.5 * np.exp(-((x_norm - 0.2)**2 + (y_norm - 0.2)**2) / 0.05))
    om_noise = np.random.normal(0, 0.2, (nrows, ncols))
    om = np.clip(gaussian_filter(om_base + om_noise, sigma=2.5), 1.2, 4.5)

    # --- EC (soil electrical conductivity, mS/m): indicator of clay content ---
    # High EC = high clay content = good water-holding capacity
    # Correlated with OM and inversely with x (east end = sandier)
    ec_base = 15 + 25 * (1 - x_norm) + 10 * y_norm
    ec_noise = np.random.normal(0, 3, (nrows, ncols))
    ec = np.clip(gaussian_filter(ec_base + ec_noise, sigma=3), 5, 55)

    # --- Elevation: gentle north-south slope (Kansas plateau tilts slightly south) ---
    elev_base = 490 + 8 * y_norm + 2 * x_norm
    elev_noise = np.random.normal(0, 1.5, (nrows, ncols))
    elevation = gaussian_filter(elev_base + elev_noise, sigma=4)

    # Mask cells outside the farm boundary
    from shapely.geometry import Point
    valid_mask = np.zeros((nrows, ncols), dtype=bool)
    for r in range(nrows):
        for c in range(ncols):
            cell_center = Point(x_arr[c], y_arr[r])
            valid_mask[r, c] = farm_geom.contains(cell_center)

    layers = {
        'ph': ph,
        'organic_matter_pct': om,
        'ec_ms_m': ec,
        'elevation_m': elevation,
    }

    meta = {
        'x_arr': x_arr, 'y_arr': y_arr,
        'xx': xx, 'yy': yy,
        'nrows': nrows, 'ncols': ncols,
        'resolution_m': resolution_m,
        'valid_mask': valid_mask,
    }

    print(f"\nSimulated layers:")
    for name, arr in layers.items():
        valid_vals = arr[valid_mask]
        print(f"  {name:<25}: min={valid_vals.min():.2f}, "
              f"max={valid_vals.max():.2f}, mean={valid_vals.mean():.2f}")

    return xx, yy, layers, meta


def determine_optimal_k(feature_matrix: np.ndarray, max_k: int = 7) -> None:
    """
    Compute inertia (elbow method) and silhouette scores to guide k selection.

    The elbow method: plot inertia vs k and look for the 'elbow' where
    adding more clusters provides diminishing returns.

    Silhouette score: measures how similar each point is to its own cluster
    vs neighboring clusters. Range: -1 to 1. Higher = better separation.

    In practice, agronomists choose k based on:
      - Economic feasibility (3-5 zones are typical)
      - Statistical metrics (elbow/silhouette)
      - Equipment capabilities (variable rate applicator resolution)
    """
    print("\n" + "=" * 65)
    print("OPTIMAL K DETERMINATION")
    print("=" * 65)

    inertias = []
    silhouette_scores = []
    k_range = range(2, min(max_k + 1, len(feature_matrix) // 2))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(feature_matrix)
        inertias.append(km.inertia_)
        if k > 1 and len(np.unique(labels)) > 1:
            sil = silhouette_score(
                feature_matrix, labels, sample_size=min(1000, len(feature_matrix))
            )
            silhouette_scores.append(sil)
        else:
            silhouette_scores.append(0)

    print(f"\n  k  | Inertia        | Silhouette")
    print(f"  ---+----------------+-----------")
    for k, inertia, sil in zip(k_range, inertias, silhouette_scores):
        marker = " ←" if k == N_ZONES else ""
        print(f"  {k:<3}| {inertia:>14.0f} | {sil:>10.4f}{marker}")

    best_sil_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    print(f"\n  Best silhouette at k={best_sil_k} | Using k={N_ZONES}")


def run_kmeans_clustering(
    layers: dict[str, np.ndarray],
    meta: dict
) -> np.ndarray:
    """
    Apply k-means clustering to the multi-layer soil feature grid.

    Steps:
      1. Stack all layers into a (n_valid_cells, n_features) array
      2. Standardize features (crucial — EC in mS/m would dominate pH in 0-14 range)
      3. Run KMeans with k=N_ZONES
      4. Map cluster labels back to 2D grid for raster visualization

    Parameters
    ----------
    layers : dict
        Variable name → 2D array of values.
    meta : dict
        Grid metadata including valid_mask.

    Returns
    -------
    np.ndarray
        2D array of cluster labels (same shape as input grids).
        Invalid cells (outside boundary) are labeled -1.
    """
    print("\n" + "=" * 65)
    print(f"K-MEANS CLUSTERING (k={N_ZONES} zones)")
    print("=" * 65)

    valid_mask = meta['valid_mask']

    # Stack features for valid cells
    feature_stack = np.column_stack([
        layer[valid_mask].ravel()
        for layer in layers.values()
    ])

    print(f"\n  Feature matrix: {feature_stack.shape[0]} cells × "
          f"{feature_stack.shape[1]} features")
    print(f"  Features: {list(layers.keys())}")

    # Standardize features — essential so all features contribute equally
    # Without this, EC (range ~5-55) would dominate pH (range ~5.8-7.8)
    scaler = StandardScaler()
    feature_scaled = scaler.fit_transform(feature_stack)

    # Determine optimal k (for informational output)
    determine_optimal_k(feature_scaled, max_k=6)

    # Run KMeans
    print(f"\n  Running KMeans with k={N_ZONES}, random_seed={RANDOM_SEED}...")
    kmeans = KMeans(
        n_clusters=N_ZONES,
        random_state=RANDOM_SEED,
        n_init=15,           # Multiple initializations for stable solution
        max_iter=300,
        tol=1e-4
    )
    cluster_labels = kmeans.fit_predict(feature_scaled)

    # Compute silhouette score for chosen k
    sil_score = silhouette_score(feature_scaled, cluster_labels,
                                  sample_size=min(1000, len(feature_scaled)))
    print(f"  Silhouette score (k={N_ZONES}): {sil_score:.4f}")
    print(f"  Inertia: {kmeans.inertia_:.0f}")

    # Map labels back to 2D grid
    label_grid = np.full(meta['valid_mask'].shape, -1, dtype=int)
    label_grid[valid_mask] = cluster_labels

    # Print cluster sizes
    print(f"\n  Cluster sizes:")
    for k in range(N_ZONES):
        count = (cluster_labels == k).sum()
        area_ha = count * (meta['resolution_m'] ** 2) / 10_000
        print(f"    Zone {k+1}: {count} cells ({area_ha:.1f} ha)")

    # Print cluster centroids (in original units after inverse transform)
    centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    print(f"\n  Zone centroids (mean soil properties per zone):")
    feature_names = list(layers.keys())
    header = "  Zone   " + "  ".join(f"{n[:8]:>8}" for n in feature_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(N_ZONES):
        vals = "  ".join(f"{v:>8.2f}" for v in centroids_orig[k])
        print(f"  Zone {k+1}: {vals}")

    return label_grid, kmeans, scaler, cluster_labels, feature_stack


def raster_to_vector_zones(
    label_grid: np.ndarray,
    meta: dict,
    boundary: gpd.GeoDataFrame,
    layers: dict,
    cluster_labels: np.ndarray,
    feature_stack: np.ndarray
) -> gpd.GeoDataFrame:
    """
    Convert raster cluster labels to vector polygon zones.

    Algorithm:
        1. For each cluster value k, create boolean mask where label_grid == k
        2. Find connected grid cells and merge into polygon(s) using box unions
        3. Dissolve touching/overlapping boxes using unary_union
        4. Clip result to farm boundary
        5. Attach zone statistics from clustering

    This is a simplified rasterization approach. For production,
    use rasterio.features.shapes() for efficient raster-to-vector conversion
    (it implements the GDAL Polygonize algorithm).

    Parameters
    ----------
    label_grid : ndarray
        2D array of cluster labels.
    meta : dict
        Grid metadata.
    boundary : GeoDataFrame
        Farm boundary for clipping.

    Returns
    -------
    GeoDataFrame
        Management zone polygons with attributes.
    """
    print("\n" + "=" * 65)
    print("CONVERTING RASTER CLUSTERS TO VECTOR ZONES")
    print("=" * 65)

    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    x_arr = meta['x_arr']
    y_arr = meta['y_arr']
    res = meta['resolution_m']
    farm_geom = boundary.geometry.iloc[0]

    # Compute per-zone statistics from the raw feature data
    valid_mask = meta['valid_mask']
    feature_names = list(layers.keys())

    zone_polygons = []

    for k in range(N_ZONES):
        # Boolean mask for this cluster
        zone_mask = (label_grid == k)
        n_cells = zone_mask.sum()

        if n_cells == 0:
            print(f"  Zone {k+1}: No cells — skipping")
            continue

        # Build individual cell boxes and union them
        cell_boxes = []
        for r in range(meta['nrows']):
            for c in range(meta['ncols']):
                if zone_mask[r, c]:
                    # Cell box: x_arr[c] ± res/2, y_arr[r] ± res/2
                    cx = x_arr[c]
                    cy = y_arr[r]
                    half = res / 2
                    cell_box = shapely_box(cx - half, cy - half,
                                           cx + half, cy + half)
                    cell_boxes.append(cell_box)

        # Union all cells for this zone
        zone_geom = unary_union(cell_boxes)

        # Clip to farm boundary
        zone_geom = zone_geom.intersection(farm_geom)

        area_ha = zone_geom.area / 10_000

        # Compute mean soil properties for this zone from the feature stack
        zone_cell_mask = cluster_labels == k
        zone_means = {
            f'mean_{feature_names[j]}': float(feature_stack[zone_cell_mask, j].mean())
            for j in range(len(feature_names))
        }

        zone_polygons.append({
            'zone_id': f'MZ-{k+1:02d}',
            'zone_num': k + 1,
            'n_cells': n_cells,
            'area_ha': round(area_ha, 1),
            **{k2: round(v, 3) for k2, v in zone_means.items()},
            'geometry': zone_geom
        })

        print(f"  Zone MZ-{k+1:02d}: {n_cells} cells, {area_ha:.1f} ha, "
              f"pH={zone_means.get('mean_ph', 0):.2f}, "
              f"OM={zone_means.get('mean_organic_matter_pct', 0):.2f}%")

    zones_gdf = gpd.GeoDataFrame(zone_polygons, crs=boundary.crs)
    print(f"\n  Generated {len(zones_gdf)} management zones")
    return zones_gdf


def visualize_management_zones(
    boundary: gpd.GeoDataFrame,
    label_grid: np.ndarray,
    zones_gdf: gpd.GeoDataFrame,
    meta: dict,
    layers: dict
) -> None:
    """Visualize k-means input layers and resulting management zones."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f'K-Means Management Zone Delineation (k={N_ZONES})\n'
        'Rolling Prairie Farm, KS | Author: Emmanuel Oyekanlu',
        fontsize=13, fontweight='bold'
    )

    xx, yy = meta['xx'], meta['yy']
    valid_mask = meta['valid_mask']

    # Mask out invalid cells for display
    display_labels = label_grid.astype(float)
    display_labels[~valid_mask] = np.nan

    var_configs = [
        ('ph', 'Soil pH', 'RdYlGn', (5.8, 7.8)),
        ('organic_matter_pct', 'Organic Matter %', 'YlOrBr', (1.2, 4.5)),
        ('ec_ms_m', 'EC (mS/m)', 'Blues', (5, 55)),
        ('elevation_m', 'Elevation (m)', 'terrain', (488, 500)),
    ]

    # --- Plots 0-3: Input layers ---
    for i, (var, title, cmap, vrange) in enumerate(var_configs):
        ax = axes.flat[i]
        display = layers[var].astype(float)
        display[~valid_mask] = np.nan

        im = ax.pcolormesh(xx, yy, display, cmap=cmap,
                           vmin=vrange[0], vmax=vrange[1], shading='auto')
        boundary.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=2)
        plt.colorbar(im, ax=ax, shrink=0.5, label=title)
        ax.set_title(f'Input: {title}', fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)

    # --- Plot 4: Cluster labels (raster) ---
    ax4 = axes.flat[4]
    zone_cmap = plt.get_cmap('tab10', N_ZONES)
    im4 = ax4.pcolormesh(xx, yy, display_labels, cmap=zone_cmap,
                          vmin=0, vmax=N_ZONES - 1, shading='auto')
    boundary.plot(ax=ax4, facecolor='none', edgecolor='black', linewidth=2)
    plt.colorbar(im4, ax=ax4, shrink=0.5, ticks=range(N_ZONES),
                 label='Zone (cluster)')
    ax4.set_title('K-Means Clusters (raster)', fontsize=10, fontweight='bold')
    ax4.tick_params(labelsize=7)

    # --- Plot 5: Vector zones ---
    ax5 = axes.flat[5]
    zone_colors = [zone_cmap(k / N_ZONES) for k in range(N_ZONES)]
    for i_zone, (_, zone_row) in enumerate(zones_gdf.iterrows()):
        color = zone_colors[i_zone % N_ZONES]
        gpd.GeoDataFrame(
            [{'geometry': zone_row['geometry']}],
            crs=zones_gdf.crs
        ).plot(ax=ax5, color=color, alpha=0.75, edgecolor='white', linewidth=1.5)

        centroid = zone_row['geometry'].centroid
        ax5.annotate(
            f"{zone_row['zone_id']}\n{zone_row['area_ha']:.0f}ha",
            xy=(centroid.x, centroid.y),
            ha='center', va='center', fontsize=8, fontweight='bold'
        )

    boundary.plot(ax=ax5, facecolor='none', edgecolor='black', linewidth=2)
    ax5.set_title('Management Zones (vector)', fontsize=10, fontweight='bold')
    ax5.tick_params(labelsize=7)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "03_management_zones.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("K-MEANS MANAGEMENT ZONE DELINEATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load boundary
    boundary = load_boundary()

    # Simulate soil layers
    print("\n" + "=" * 65)
    print("SIMULATING SOIL PROPERTY LAYERS")
    print("=" * 65)
    xx, yy, layers, meta = simulate_soil_layers(boundary, resolution_m=GRID_RES_M)

    # Run k-means clustering
    label_grid, kmeans, scaler, cluster_labels, feature_stack = run_kmeans_clustering(
        layers, meta
    )

    # Convert raster zones to vector polygons
    zones_gdf = raster_to_vector_zones(
        label_grid, meta, boundary, layers, cluster_labels, feature_stack
    )

    # Visualize
    visualize_management_zones(boundary, label_grid, zones_gdf, meta, layers)

    # Save outputs
    zones_gdf.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "management_zones_kmeans.geojson"),
        driver='GeoJSON'
    )

    # Save zone attributes as CSV for further analysis
    zone_attrs = zones_gdf.drop(columns='geometry')
    zone_attrs.to_csv(
        os.path.join(OUTPUT_DIR, "management_zones_attributes.csv"),
        index=False
    )

    print("\nOutputs saved:")
    print("  output/management_zones_kmeans.geojson")
    print("  output/management_zones_attributes.csv")
    print("  output/03_management_zones.png")


if __name__ == "__main__":
    main()
