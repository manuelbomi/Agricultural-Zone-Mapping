"""
04_yield_map_processing.py
===========================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates yield monitor data processing:
  - Simulate a realistic yield monitor point cloud (1,000 GPS-tagged harvest points)
  - Create a GeoDataFrame from the point cloud
  - Clean and filter yield data (remove outliers using IQR method)
  - Compute spatial interpolation summary (average yield per management zone)
  - Generate choropleth maps of yield distribution
  - Compute zone-level yield statistics

Yield monitor context:
    Modern grain combines are equipped with yield monitors that record:
      - GPS position (harvested at 1-second intervals)
      - Instantaneous yield (bu/ac or t/ha)
      - Grain moisture content
      - Speed and header width (for area calculation)

    Raw yield data requires substantial cleaning before use:
      - Remove start/stop transient outliers (first/last pass)
      - Remove values outside 3-sigma or IQR range
      - Filter GPS locations outside field boundary
      - Correct for moisture (standardize to 15.5% for corn)

    Clean yield data → spatial interpolation → zone-level summaries →
    multi-year analysis → management zone refinement

Run:
    python 04_yield_map_processing.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from shapely.vectorized import contains

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")
UTM_CRS = 'EPSG:32614'

# Yield simulation parameters
N_POINTS = 1000       # Simulated harvest points
CROP = 'corn'
YIELD_YEAR = 2024
TYPICAL_YIELD_T_HA = 10.5  # Kansas corn ~165 bu/ac ≈ 10.5 t/ha
YIELD_NOISE_STD = 1.8      # Standard deviation for realistic spread


def load_boundary() -> gpd.GeoDataFrame:
    """Load and project farm boundary."""
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(UTM_CRS)
    return boundary


def simulate_yield_monitor_points(
    boundary: gpd.GeoDataFrame,
    n_points: int
) -> gpd.GeoDataFrame:
    """
    Simulate yield monitor point cloud for a corn harvest.

    Realistic yield patterns include:
      - Spatial trend: higher yield in center (better soil), lower at edges
      - Row effect: combine harvest paths create north-south strips
      - Outliers: 3-5% of points are extreme (equipment malfunction, GPS errors)
      - Moisture variation: freshly harvested corn varies from 18-25% moisture

    Parameters
    ----------
    boundary : GeoDataFrame
        Farm boundary in metric CRS.
    n_points : int
        Number of yield monitor points to simulate.

    Returns
    -------
    GeoDataFrame
        Yield monitor points with harvest attributes.
    """
    print("=" * 65)
    print(f"SIMULATING {n_points} YIELD MONITOR POINTS")
    print("=" * 65)

    np.random.seed(42)
    farm_geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = farm_geom.bounds
    width = maxx - minx
    height = maxy - miny

    points = []
    attempts = 0
    max_attempts = n_points * 8

    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        # Sample points in strips (simulating combine rows ~12m apart)
        n_strips = 25
        strip_width = width / n_strips
        strip_idx = np.random.randint(0, n_strips)
        x = minx + strip_idx * strip_width + np.random.uniform(0, strip_width)
        y = miny + np.random.uniform(0, height)

        pt = Point(x, y)
        if not farm_geom.contains(pt):
            continue

        # Normalized position within field (0=edge, 1=center)
        x_norm = (x - minx) / width
        y_norm = (y - miny) / height

        # Yield trend: higher in center, lower at edges (typical soil pattern)
        edge_penalty = 0.8 * (min(x_norm, 1 - x_norm) * min(y_norm, 1 - y_norm))
        base_yield = TYPICAL_YIELD_T_HA + 2.0 * edge_penalty - 0.5

        # Additional variability from soil pattern (use simplified wave function)
        soil_effect = 1.2 * np.sin(x_norm * np.pi * 2) * np.cos(y_norm * np.pi)

        # Random noise (day-to-day weather, equipment variation)
        noise = np.random.normal(0, YIELD_NOISE_STD)

        yield_t_ha = base_yield + soil_effect + noise

        # Introduce outliers (5% of points)
        is_outlier = np.random.random() < 0.05
        if is_outlier:
            # Either very high (header surge) or very low (equipment stoppage)
            outlier_type = np.random.choice(['high', 'low'])
            if outlier_type == 'high':
                yield_t_ha = np.random.uniform(20, 30)  # Obviously too high
            else:
                yield_t_ha = np.random.uniform(0, 1.5)  # Near-zero

        # Moisture at harvest (18-24% for Kansas corn)
        moisture_pct = np.random.normal(21.5, 1.8)
        moisture_pct = np.clip(moisture_pct, 16, 26)

        # Standard yield at 15.5% moisture
        dry_matter_yield = yield_t_ha * (1 - moisture_pct / 100)
        std_yield_t_ha = dry_matter_yield / (1 - 0.155)

        points.append({
            'point_id': f'YP-{len(points)+1:04d}',
            'crop': CROP,
            'year': YIELD_YEAR,
            'raw_yield_t_ha': round(yield_t_ha, 2),
            'std_yield_t_ha': round(std_yield_t_ha, 2),
            'moisture_pct': round(moisture_pct, 1),
            'speed_kph': round(np.random.normal(6.5, 0.5), 1),
            'is_simulated_outlier': is_outlier,
            'geometry': pt
        })

    gdf = gpd.GeoDataFrame(points, crs=boundary.crs)
    print(f"\n  Generated {len(gdf)} yield monitor points")
    print(f"  Raw yield range: {gdf['raw_yield_t_ha'].min():.1f} – "
          f"{gdf['raw_yield_t_ha'].max():.1f} t/ha")
    print(f"  Raw yield mean : {gdf['raw_yield_t_ha'].mean():.2f} t/ha")
    print(f"  Outliers (simulated): {gdf['is_simulated_outlier'].sum()}")

    return gdf


def clean_yield_data(yield_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean yield monitor data using IQR outlier detection.

    Standard yield cleaning workflow:
      1. Remove physically impossible values (yield < 0 or > theoretical max)
      2. Apply IQR-based outlier filter
      3. Remove speed outliers (yield unreliable when combine is stopping/starting)
      4. Log cleaning statistics

    IQR method:
        Q1 = 25th percentile, Q3 = 75th percentile
        IQR = Q3 - Q1
        Lower fence = Q1 - 1.5 × IQR
        Upper fence = Q3 + 1.5 × IQR
        Values outside fences are flagged as outliers.

    Parameters
    ----------
    yield_gdf : GeoDataFrame
        Raw yield monitor points.

    Returns
    -------
    GeoDataFrame
        Cleaned yield points with 'is_outlier' and 'quality_flag' columns.
    """
    print("\n" + "=" * 65)
    print("YIELD DATA CLEANING (IQR METHOD)")
    print("=" * 65)

    yield_col = 'raw_yield_t_ha'
    gdf = yield_gdf.copy()

    # Step 1: Physical bounds filter
    gdf['flag_physical'] = (gdf[yield_col] < 0) | (gdf[yield_col] > 25)

    # Step 2: IQR outlier detection on remaining good data
    q1 = gdf.loc[~gdf['flag_physical'], yield_col].quantile(0.25)
    q3 = gdf.loc[~gdf['flag_physical'], yield_col].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    gdf['flag_iqr'] = (
        (gdf[yield_col] < lower_fence) |
        (gdf[yield_col] > upper_fence)
    ) & ~gdf['flag_physical']

    # Step 3: Speed filter (harvest speed should be 4-9 kph for corn)
    gdf['flag_speed'] = (gdf['speed_kph'] < 4.0) | (gdf['speed_kph'] > 9.0)

    # Combined outlier flag
    gdf['is_outlier'] = gdf['flag_physical'] | gdf['flag_iqr'] | gdf['flag_speed']

    # Quality flag text
    def quality_label(row):
        if row['flag_physical']:
            return 'INVALID_PHYSICAL'
        elif row['flag_speed']:
            return 'INVALID_SPEED'
        elif row['flag_iqr']:
            return 'OUTLIER_IQR'
        return 'GOOD'

    gdf['quality_flag'] = gdf.apply(quality_label, axis=1)

    clean_data = gdf[~gdf['is_outlier']]
    outlier_data = gdf[gdf['is_outlier']]

    print(f"\n  IQR statistics (before filtering):")
    print(f"    Q1  : {q1:.2f} t/ha")
    print(f"    Q3  : {q3:.2f} t/ha")
    print(f"    IQR : {iqr:.2f} t/ha")
    print(f"    Lower fence: {lower_fence:.2f} t/ha")
    print(f"    Upper fence: {upper_fence:.2f} t/ha")

    print(f"\n  Cleaning results:")
    for flag, count in gdf['quality_flag'].value_counts().items():
        pct = count / len(gdf) * 100
        print(f"    {flag:<20}: {count:>4} ({pct:.1f}%)")

    print(f"\n  Before cleaning: {len(gdf)} points, mean={gdf[yield_col].mean():.2f} t/ha")
    print(f"  After  cleaning: {len(clean_data)} points, "
          f"mean={clean_data[yield_col].mean():.2f} t/ha")

    return gdf


def compute_zone_yield_statistics(
    yield_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Create simple management zones (3×2 grid) and compute yield stats per zone.

    In production, uses the k-means zones from 03_management_zones.py.
    Here we generate a simple grid to demonstrate the zone averaging concept.

    Parameters
    ----------
    yield_gdf : GeoDataFrame
        Cleaned yield points.
    boundary : GeoDataFrame
        Farm boundary.

    Returns
    -------
    DataFrame
        Zone-level yield statistics.
    """
    print("\n" + "=" * 65)
    print("ZONE-LEVEL YIELD STATISTICS")
    print("=" * 65)

    # Use only clean points
    clean_points = yield_gdf[~yield_gdf['is_outlier']].copy()

    # Create a simple 3×2 zone grid for demonstration
    farm_geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = farm_geom.bounds
    n_cols, n_rows = 3, 2
    cell_w = (maxx - minx) / n_cols
    cell_h = (maxy - miny) / n_rows

    # Assign each clean point to a zone
    def assign_zone(row):
        x, y = row.geometry.x, row.geometry.y
        col = min(int((x - minx) / cell_w), n_cols - 1)
        row_idx = min(int((y - miny) / cell_h), n_rows - 1)
        return f"Z{row_idx+1}{col+1}"

    clean_points['zone_id'] = clean_points.apply(assign_zone, axis=1)

    # Compute statistics per zone
    zone_stats = clean_points.groupby('zone_id').agg(
        n_points=('raw_yield_t_ha', 'count'),
        mean_yield_t_ha=('raw_yield_t_ha', 'mean'),
        median_yield_t_ha=('raw_yield_t_ha', 'median'),
        std_yield_t_ha=('raw_yield_t_ha', 'std'),
        min_yield_t_ha=('raw_yield_t_ha', 'min'),
        max_yield_t_ha=('raw_yield_t_ha', 'max'),
        p25=('raw_yield_t_ha', lambda x: x.quantile(0.25)),
        p75=('raw_yield_t_ha', lambda x: x.quantile(0.75)),
    ).reset_index()

    zone_stats = zone_stats.round(2)

    print(f"\n  {len(clean_points)} clean points → {len(zone_stats)} zones")
    print(f"\n  {'Zone':<8} {'N Pts':>6} {'Mean':>7} {'Median':>8} "
          f"{'Std':>6} {'Min':>7} {'Max':>7}")
    print("  " + "-" * 60)

    for _, row in zone_stats.sort_values('zone_id').iterrows():
        print(f"  {row['zone_id']:<8} {int(row['n_points']):>6} "
              f"{row['mean_yield_t_ha']:>7.2f} "
              f"{row['median_yield_t_ha']:>8.2f} "
              f"{row['std_yield_t_ha']:>6.2f} "
              f"{row['min_yield_t_ha']:>7.2f} "
              f"{row['max_yield_t_ha']:>7.2f}")

    overall_mean = clean_points['raw_yield_t_ha'].mean()
    print(f"\n  Farm mean yield: {overall_mean:.2f} t/ha")
    print(f"  Coefficient of variation: "
          f"{clean_points['raw_yield_t_ha'].std() / overall_mean * 100:.1f}%")

    # Flag high/low performing zones
    threshold = overall_mean * 0.10  # 10% threshold
    print(f"\n  Zone performance vs farm mean:")
    for _, row in zone_stats.iterrows():
        diff = row['mean_yield_t_ha'] - overall_mean
        flag = "HIGH" if diff > threshold else ("LOW" if diff < -threshold else "AVG")
        print(f"    {row['zone_id']}: {row['mean_yield_t_ha']:.2f} t/ha "
              f"({diff:+.2f}) [{flag}]")

    return zone_stats


def visualize_yield_map(
    yield_gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> None:
    """Create yield map visualizations: raw, cleaned, and choropleth."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        f'Yield Monitor Data Processing — {CROP.title()} {YIELD_YEAR}\n'
        'Rolling Prairie Farm, KS | Author: Emmanuel Oyekanlu',
        fontsize=12, fontweight='bold'
    )

    clean = yield_gdf[~yield_gdf['is_outlier']]
    outliers = yield_gdf[yield_gdf['is_outlier']]

    vmin = clean['raw_yield_t_ha'].quantile(0.05)
    vmax = clean['raw_yield_t_ha'].quantile(0.95)

    # --- Plot 1: Raw yield with outliers highlighted ---
    ax1 = axes[0]
    sc = ax1.scatter(
        yield_gdf.geometry.x, yield_gdf.geometry.y,
        c=yield_gdf['raw_yield_t_ha'],
        cmap='RdYlGn', vmin=vmin, vmax=vmax,
        s=8, alpha=0.6
    )
    if len(outliers) > 0:
        ax1.scatter(
            outliers.geometry.x, outliers.geometry.y,
            c='black', s=25, marker='x', zorder=5, label='Outliers'
        )
    plt.colorbar(sc, ax=ax1, shrink=0.6, label='Yield (t/ha)')
    boundary.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=2)
    ax1.set_title(f'Raw Yield ({len(yield_gdf)} pts)\n(X = outliers)', fontsize=10)
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.tick_params(labelsize=7)

    # --- Plot 2: Cleaned yield only ---
    ax2 = axes[1]
    sc2 = ax2.scatter(
        clean.geometry.x, clean.geometry.y,
        c=clean['raw_yield_t_ha'],
        cmap='RdYlGn', vmin=vmin, vmax=vmax,
        s=8, alpha=0.6
    )
    plt.colorbar(sc2, ax=ax2, shrink=0.6, label='Yield (t/ha)')
    boundary.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2)
    ax2.set_title(f'Cleaned Yield ({len(clean)} pts)\n(IQR filter applied)',
                  fontsize=10)
    ax2.set_xlabel('Easting (m)')
    ax2.tick_params(labelsize=7)

    # --- Plot 3: Yield histogram ---
    ax3 = axes[2]
    ax3.hist(
        yield_gdf['raw_yield_t_ha'], bins=40,
        color='lightgray', alpha=0.7, edgecolor='gray',
        label='All points'
    )
    ax3.hist(
        clean['raw_yield_t_ha'], bins=40,
        color='#4CAF50', alpha=0.7, edgecolor='darkgreen',
        label='Clean points'
    )
    ax3.axvline(clean['raw_yield_t_ha'].mean(), color='navy',
                linestyle='--', linewidth=1.5,
                label=f"Mean: {clean['raw_yield_t_ha'].mean():.2f} t/ha")
    ax3.axvline(clean['raw_yield_t_ha'].median(), color='orange',
                linestyle='--', linewidth=1.5,
                label=f"Median: {clean['raw_yield_t_ha'].median():.2f} t/ha")
    ax3.set_xlabel('Yield (t/ha)')
    ax3.set_ylabel('Count')
    ax3.set_title('Yield Distribution', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "04_yield_map.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("YIELD MONITOR DATA PROCESSING")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load boundary
    boundary = load_boundary()

    # Simulate yield monitor point cloud
    yield_gdf = simulate_yield_monitor_points(boundary, N_POINTS)

    # Clean the yield data
    yield_gdf = clean_yield_data(yield_gdf)

    # Compute zone-level statistics
    zone_stats = compute_zone_yield_statistics(yield_gdf, boundary)

    # Visualize
    visualize_yield_map(yield_gdf, boundary)

    # Save outputs
    clean_gdf = yield_gdf[~yield_gdf['is_outlier']].copy()
    clean_gdf.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "yield_monitor_clean.geojson"),
        driver='GeoJSON'
    )

    # Save all points with quality flags as CSV
    yield_csv = yield_gdf.copy()
    yield_csv['x_utm'] = yield_gdf.geometry.x
    yield_csv['y_utm'] = yield_gdf.geometry.y
    yield_csv.drop(columns='geometry').to_csv(
        os.path.join(OUTPUT_DIR, "yield_monitor_all_points.csv"), index=False
    )

    zone_stats.to_csv(
        os.path.join(OUTPUT_DIR, "yield_zone_statistics.csv"), index=False
    )

    print("\nOutputs saved:")
    print(f"  output/yield_monitor_clean.geojson ({len(clean_gdf)} clean points)")
    print(f"  output/yield_monitor_all_points.csv ({len(yield_gdf)} all points)")
    print(f"  output/yield_zone_statistics.csv")
    print(f"  output/04_yield_map.png")


if __name__ == "__main__":
    main()
