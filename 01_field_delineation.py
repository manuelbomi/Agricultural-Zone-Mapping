"""
01_field_delineation.py
========================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates field boundary processing and management zone grid generation:
  - Load a farm boundary polygon
  - Subdivide into a regular grid of management zone sub-polygons
  - Assign unique zone IDs and compute zone properties
  - Export management zone grid as GeoJSON

Management zone grid approach:
    The simplest delineation method — divide the field into a regular N×M grid.
    Each cell becomes a management zone for sampling and variable-rate application.
    Typical grid sizes: 1 acre (2.5-acre sample sites), 2.5 acre, or 5 acre.

    More sophisticated methods (k-means on soil/yield layers) are demonstrated
    in 03_management_zones.py. The grid approach provides the spatial framework
    that more advanced methods refine.

Use case context:
    Before automated zone delineation was available, farmers sampled fields
    on a regular 1-acre grid. Even today, grid-based zones serve as:
      - Default management units before enough data is collected
      - Reference framework for validating algorithmic zone delineation
      - Spatial reporting structure for field trials

Run:
    python 01_field_delineation.py
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")

# Zone grid configuration
ZONE_COLS = 6   # Columns in grid (east-west)
ZONE_ROWS = 5   # Rows in grid (north-south)


def load_farm_boundary() -> gpd.GeoDataFrame:
    """
    Load farm boundary and project to a metric CRS for accurate grid generation.

    UTM Zone 14N (EPSG:32614) covers central Kansas (~96°-102°W).
    This ensures grid cell dimensions are in meters, not degrees.
    """
    boundary = gpd.read_file(BOUNDARY_PATH)
    boundary_utm = boundary.to_crs('EPSG:32614')  # UTM Zone 14N for Kansas

    print("=" * 65)
    print("FARM BOUNDARY LOADED")
    print("=" * 65)
    props = boundary.iloc[0]['properties'] if 'properties' in boundary.columns \
        else boundary.iloc[0]
    print(f"  Farm        : {boundary.iloc[0].get('farm_name', 'KS Farm')}")
    print(f"  County      : {boundary.iloc[0].get('county', 'Ellsworth')}, KS")
    print(f"  Area (attr) : {boundary.iloc[0].get('total_area_ha', '?')} ha")

    # Compute actual area from geometry
    actual_area = boundary_utm.geometry.area.sum() / 10_000
    print(f"  Area (geom) : {actual_area:.1f} ha")
    print(f"  CRS (input) : {boundary.crs}")
    print(f"  CRS (work)  : {boundary_utm.crs}")
    print(f"  Bounds (m)  : {boundary_utm.total_bounds.round(0)}")

    return boundary_utm


def generate_grid_zones(
    boundary: gpd.GeoDataFrame,
    n_cols: int,
    n_rows: int
) -> gpd.GeoDataFrame:
    """
    Subdivide the farm boundary into a regular rectangular grid of management zones.

    Algorithm:
        1. Get bounding box of the farm boundary
        2. Divide bounding box into n_rows × n_cols rectangles
        3. Clip each rectangle to the farm boundary (handles irregular field shapes)
        4. Discard clipped cells below area threshold (corner slivers)

    Parameters
    ----------
    boundary : GeoDataFrame
        Farm boundary polygon in metric CRS.
    n_cols : int
        Number of grid columns (east-west divisions).
    n_rows : int
        Number of grid rows (north-south divisions).

    Returns
    -------
    GeoDataFrame
        Management zone grid with zone IDs and area attributes.
    """
    print(f"\n{'='*65}")
    print(f"GENERATING {n_rows}×{n_cols} MANAGEMENT ZONE GRID")
    print("=" * 65)

    farm_polygon = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = farm_polygon.bounds

    # Cell dimensions in meters
    cell_width_m = (maxx - minx) / n_cols
    cell_height_m = (maxy - miny) / n_rows

    print(f"\n  Grid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} potential zones")
    print(f"  Cell size: {cell_width_m:.0f}m × {cell_height_m:.0f}m")
    print(f"  Nominal cell area: {cell_width_m * cell_height_m / 10_000:.1f} ha")

    zones = []
    zone_num = 0

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            # Build cell bounding box (row 0 = northernmost)
            cell_minx = minx + col_idx * cell_width_m
            cell_maxx = minx + (col_idx + 1) * cell_width_m
            cell_miny = maxy - (row_idx + 1) * cell_height_m
            cell_maxy = maxy - row_idx * cell_height_m

            cell_box = box(cell_minx, cell_miny, cell_maxx, cell_maxy)

            # Clip cell to farm boundary (handles irregular field edges)
            clipped = cell_box.intersection(farm_polygon)

            # Discard slivers: cells with <5% of nominal area
            nominal_area = cell_width_m * cell_height_m
            if clipped.area < 0.05 * nominal_area:
                continue

            zone_num += 1
            zone_area_ha = clipped.area / 10_000

            # Grid position labels (A-Z for columns, 1-N for rows)
            col_label = chr(ord('A') + col_idx)
            row_label = str(row_idx + 1)
            zone_id = f"Z{col_label}{row_label}"

            zones.append({
                'zone_id':        zone_id,
                'zone_num':       zone_num,
                'grid_row':       row_idx + 1,
                'grid_col':       col_idx + 1,
                'col_label':      col_label,
                'area_ha':        round(zone_area_ha, 2),
                'is_boundary_zone': not clipped.equals(cell_box),  # True if clipped
                'centroid_x_m':   round(clipped.centroid.x, 1),
                'centroid_y_m':   round(clipped.centroid.y, 1),
                'geometry':       clipped,
            })

    zones_gdf = gpd.GeoDataFrame(zones, crs=boundary.crs)

    print(f"\n  Generated {len(zones_gdf)} valid zones "
          f"({n_rows * n_cols - len(zones_gdf)} discarded as slivers)")
    print(f"  Total zone area: {zones_gdf['area_ha'].sum():.1f} ha")
    print(f"  Average zone area: {zones_gdf['area_ha'].mean():.1f} ha")
    print(f"  Boundary zones (clipped): {zones_gdf['is_boundary_zone'].sum()}")

    return zones_gdf


def assign_sampling_points(zones_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Generate a soil sampling point at the centroid of each management zone.

    In precision agriculture, each zone gets at least one soil sample to
    characterize its nutrient status. The sampling point is usually the
    zone centroid or a GPS-navigated point within the zone.

    Returns a Point GeoDataFrame for the sampling grid.
    """
    sampling_points = []

    for _, zone in zones_gdf.iterrows():
        centroid = zone['geometry'].centroid
        sampling_points.append({
            'sample_id': f"SP-{zone['zone_id']}",
            'zone_id': zone['zone_id'],
            'zone_area_ha': zone['area_ha'],
            'geometry': centroid
        })

    return gpd.GeoDataFrame(sampling_points, crs=zones_gdf.crs)


def print_zone_summary(zones_gdf: gpd.GeoDataFrame) -> None:
    """Print a formatted grid layout showing zone IDs and areas."""
    print("\n" + "=" * 65)
    print("MANAGEMENT ZONE GRID LAYOUT")
    print("=" * 65)

    max_col = zones_gdf['grid_col'].max()
    max_row = zones_gdf['grid_row'].max()

    # Build lookup
    zone_lookup = {
        (r['grid_row'], r['grid_col']): r
        for _, r in zones_gdf.iterrows()
    }

    print(f"\n  Grid (Zone ID / Area ha):")
    print(f"  {'':5}", end='')
    for col in range(1, max_col + 1):
        print(f"  {'Col '+str(col):>10}", end='')
    print()

    for row in range(1, max_row + 1):
        print(f"  {'Row '+str(row):<5}", end='')
        for col in range(1, max_col + 1):
            zone = zone_lookup.get((row, col))
            if zone:
                label = f"{zone['zone_id']}/{zone['area_ha']:.1f}ha"
            else:
                label = '---'
            print(f"  {label:>10}", end='')
        print()

    print(f"\n  Total zones: {len(zones_gdf)}")
    print(f"  Total area: {zones_gdf['area_ha'].sum():.1f} ha")


def visualize_zones(
    boundary: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    sampling_points: gpd.GeoDataFrame
) -> None:
    """Visualize the management zone grid with sampling points."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Color zones by row (alternating pattern for visibility)
    zone_colors = ['#E8F5E9', '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A']

    # --- Plot 1: Zone grid colored by row ---
    ax1 = axes[0]
    boundary.plot(ax=ax1, facecolor='none', edgecolor='black',
                  linewidth=2.5, zorder=10)

    for _, zone in zones_gdf.iterrows():
        color = zone_colors[(zone['grid_row'] - 1) % len(zone_colors)]
        gpd.GeoDataFrame([zone], geometry='geometry', crs=zones_gdf.crs).plot(
            ax=ax1, color=color, edgecolor='darkgreen', linewidth=1.0, alpha=0.8
        )
        centroid = zone['geometry'].centroid
        ax1.annotate(
            zone['zone_id'],
            xy=(centroid.x, centroid.y),
            ha='center', va='center',
            fontsize=7, fontweight='bold', color='black'
        )

    sampling_points.plot(
        ax=ax1, color='red', markersize=25, zorder=5, marker='+'
    )

    ax1.set_title(f'Management Zone Grid ({ZONE_ROWS}×{ZONE_COLS})\n'
                  'Rolling Prairie Farm, Ellsworth Co., KS',
                  fontsize=11, fontweight='bold')
    ax1.set_xlabel('Easting (m, UTM Zone 14N)')
    ax1.set_ylabel('Northing (m, UTM Zone 14N)')

    # --- Plot 2: Zone area distribution ---
    ax2 = axes[1]
    areas = zones_gdf['area_ha'].sort_values()
    zone_ids = [zones_gdf.loc[i, 'zone_id'] for i in areas.index]
    colors = ['#FF7043' if zones_gdf.loc[i, 'is_boundary_zone'] else '#4CAF50'
              for i in areas.index]

    bars = ax2.barh(zone_ids, areas.values, color=colors, alpha=0.8, edgecolor='white')

    ax2.set_xlabel('Zone Area (ha)')
    ax2.set_ylabel('Zone ID')
    ax2.set_title('Zone Area Distribution\n(Orange = boundary zones, clipped)',
                  fontsize=11, fontweight='bold')
    ax2.axvline(zones_gdf['area_ha'].mean(), color='navy', linestyle='--',
                linewidth=1.5, label=f"Mean: {zones_gdf['area_ha'].mean():.1f} ha")
    ax2.legend(fontsize=9)

    # Add area labels
    for bar, area in zip(bars, areas.values):
        ax2.text(area + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{area:.1f}', va='center', fontsize=7)

    fig.suptitle(
        'Field Delineation — Management Zone Grid\n'
        'Author: Emmanuel Oyekanlu — Principal Data Engineer',
        fontsize=12, fontweight='bold'
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "01_field_delineation.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("FIELD DELINEATION & MANAGEMENT ZONE GRID GENERATION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load farm boundary
    boundary = load_farm_boundary()

    # Generate management zone grid
    zones_gdf = generate_grid_zones(boundary, n_cols=ZONE_COLS, n_rows=ZONE_ROWS)

    # Assign sampling points
    sampling_points = assign_sampling_points(zones_gdf)

    # Print zone layout
    print_zone_summary(zones_gdf)

    # Visualize
    visualize_zones(boundary, zones_gdf, sampling_points)

    # Save outputs
    zones_wgs84 = zones_gdf.to_crs('EPSG:4326')
    zones_wgs84.to_file(
        os.path.join(OUTPUT_DIR, "management_zones_grid.geojson"),
        driver='GeoJSON'
    )

    sample_pts_wgs84 = sampling_points.to_crs('EPSG:4326')
    sample_pts_wgs84.to_file(
        os.path.join(OUTPUT_DIR, "sampling_points.geojson"),
        driver='GeoJSON'
    )

    print(f"\nSaved:")
    print(f"  output/management_zones_grid.geojson ({len(zones_gdf)} zones)")
    print(f"  output/sampling_points.geojson ({len(sampling_points)} points)")


if __name__ == "__main__":
    main()
