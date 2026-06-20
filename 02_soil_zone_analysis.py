"""
02_soil_zone_analysis.py
=========================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Demonstrates spatial analysis of soil type distribution within farm fields:
  - Load soil type polygons (SSURGO-style data)
  - Overlay soil polygons with farm boundary to clip to field extent
  - Compute area of each soil type within the field
  - Identify dominant soil type
  - Analyze soil chemical properties (pH, organic matter) across the farm
  - Visualize soil type map with legend

SSURGO context:
    The USDA SSURGO (Soil Survey Geographic) database is the primary source
    of soil data for US agricultural fields. It provides polygon-based soil
    maps with dozens of attributes per map unit including:
    - Soil series name (e.g., Harney, Richfield, Pratt)
    - pH, organic matter, CEC (cation exchange capacity)
    - Drainage class, available water capacity
    - Capability class (land capability classification)

    SSURGO data is available via the USDA Web Soil Survey or through
    the soilDB R package / Python equivalent.

Run:
    python 02_soil_zone_analysis.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")
SOIL_PATH = os.path.join(DATA_DIR, "soil_types.geojson")

# UTM Zone 14N for Kansas
UTM_CRS = 'EPSG:32614'


def load_and_project_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load boundary and soil data, project to metric CRS."""
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(UTM_CRS)
    soils = gpd.read_file(SOIL_PATH).to_crs(UTM_CRS)

    print("=" * 65)
    print("DATA LOADED")
    print("=" * 65)
    print(f"\nFarm boundary: {len(boundary)} polygon(s)")
    print(f"Soil polygons : {len(soils)} map units")
    print(f"\nSoil types present:")
    for _, row in soils.iterrows():
        print(f"  {row['soil_id']}: {row['soil_type']} ({row['series_name']}) "
              f"— pH={row['ph']}, OM={row['organic_matter_pct']}%")

    return boundary, soils


def clip_soils_to_boundary(
    soils: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Clip soil polygons to the farm boundary extent.

    This is the standard preprocessing step when working with SSURGO data:
    the soil map covers a much larger area than your field of interest,
    so you clip it to your field boundary before analysis.

    Uses gpd.clip() which performs intersection and clips geometries.
    Unlike overlay(how='intersection'), clip() preserves the left GeoDataFrame's
    schema and simply removes/trims geometries outside the clip boundary.

    Parameters
    ----------
    soils : GeoDataFrame
        Soil type polygons (larger than farm boundary).
    boundary : GeoDataFrame
        Farm boundary polygon to clip to.

    Returns
    -------
    GeoDataFrame
        Soil polygons clipped to farm boundary, with computed area added.
    """
    print("\n" + "=" * 65)
    print("CLIPPING SOILS TO FARM BOUNDARY")
    print("=" * 65)

    print(f"\nInput: {len(soils)} soil polygons (raw, unclipped)")

    # gpd.clip() clips the first GeoDataFrame to the boundary geometry
    # It handles both simple and complex (multi-polygon) boundaries
    clipped_soils = gpd.clip(soils, boundary)

    # Compute clipped area (accurate because we're in metric CRS)
    clipped_soils = clipped_soils.copy()
    clipped_soils['area_ha'] = (clipped_soils.geometry.area / 10_000).round(2)

    print(f"Output: {len(clipped_soils)} soil polygons (clipped to boundary)")

    # Check for any soil polygons that disappeared (fully outside boundary)
    original_ids = set(soils['soil_id'])
    clipped_ids = set(clipped_soils['soil_id'])
    removed = original_ids - clipped_ids
    if removed:
        print(f"  Removed (fully outside boundary): {removed}")

    print(f"\nClipped soil areas:")
    for _, row in clipped_soils.sort_values('area_ha', ascending=False).iterrows():
        pct = row['area_ha'] / clipped_soils['area_ha'].sum() * 100
        print(f"  {row['soil_id']:<10} {row['soil_type']:<20} "
              f"{row['area_ha']:>7.1f} ha  ({pct:.1f}%)")

    return clipped_soils


def compute_soil_type_statistics(clipped_soils: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compute area-weighted statistics for soil properties across the farm.

    Aggregate by soil type (a single soil type may appear in multiple
    disjoint polygons). Compute:
      - Total area per soil type
      - Percentage of farm area
      - Area-weighted mean pH and organic matter
      - Dominant soil type (largest area)

    Parameters
    ----------
    clipped_soils : GeoDataFrame
        Clipped soil polygons with area_ha.

    Returns
    -------
    DataFrame
        Per-soil-type statistics.
    """
    print("\n" + "=" * 65)
    print("SOIL TYPE STATISTICS")
    print("=" * 65)

    total_area = clipped_soils['area_ha'].sum()

    # Compute area-weighted pH and OM for each soil type
    clipped_soils = clipped_soils.copy()
    clipped_soils['ph_x_area'] = clipped_soils['ph'] * clipped_soils['area_ha']
    clipped_soils['om_x_area'] = (
        clipped_soils['organic_matter_pct'] * clipped_soils['area_ha']
    )

    # Group by soil_type (handles repeated soil types)
    soil_stats = clipped_soils.groupby('soil_type').agg(
        soil_id=('soil_id', 'first'),
        series_name=('series_name', 'first'),
        total_area_ha=('area_ha', 'sum'),
        ph_weighted_sum=('ph_x_area', 'sum'),
        om_weighted_sum=('om_x_area', 'sum'),
        cec_mean=('cec_meq_100g', 'mean'),
        drainage_class=('drainage_class', 'first'),
        capability_class=('capability_class', 'first'),
    ).reset_index()

    # Compute weighted averages
    soil_stats['pct_of_farm'] = (
        soil_stats['total_area_ha'] / total_area * 100
    ).round(1)
    soil_stats['wtd_avg_ph'] = (
        soil_stats['ph_weighted_sum'] / soil_stats['total_area_ha']
    ).round(2)
    soil_stats['wtd_avg_om_pct'] = (
        soil_stats['om_weighted_sum'] / soil_stats['total_area_ha']
    ).round(2)

    # Sort by area (dominant first)
    soil_stats = soil_stats.sort_values('total_area_ha', ascending=False)

    print(f"\nTotal farm area: {total_area:.1f} ha")
    print(f"\n{'Soil Type':<22} {'Area(ha)':>9} {'%Farm':>7} "
          f"{'Wtd pH':>8} {'Wtd OM%':>8} {'CEC':>6} {'Drainage'}")
    print("-" * 85)

    for _, row in soil_stats.iterrows():
        print(f"{row['soil_type']:<22} {row['total_area_ha']:>9.1f} "
              f"{row['pct_of_farm']:>6.1f}% "
              f"{row['wtd_avg_ph']:>8.2f} "
              f"{row['wtd_avg_om_pct']:>7.2f}% "
              f"{row['cec_mean']:>5.1f} "
              f"  {row['drainage_class']}")

    # Identify dominant soil
    dominant = soil_stats.iloc[0]
    print(f"\n  Dominant soil type: {dominant['soil_type']} "
          f"({dominant['series_name']} series)")
    print(f"    Covers {dominant['total_area_ha']:.1f} ha "
          f"({dominant['pct_of_farm']:.1f}% of farm)")
    print(f"    Weighted avg pH  : {dominant['wtd_avg_ph']:.2f}")
    print(f"    Weighted avg OM  : {dominant['wtd_avg_om_pct']:.2f}%")

    # Farm-level weighted averages
    farm_avg_ph = (
        (clipped_soils['ph'] * clipped_soils['area_ha']).sum() /
        clipped_soils['area_ha'].sum()
    )
    farm_avg_om = (
        (clipped_soils['organic_matter_pct'] * clipped_soils['area_ha']).sum() /
        clipped_soils['area_ha'].sum()
    )
    print(f"\n  Farm-wide averages (area-weighted):")
    print(f"    pH              : {farm_avg_ph:.2f}")
    print(f"    Organic Matter  : {farm_avg_om:.2f}%")

    # Flag pH problem areas (too acidic or too alkaline for corn/wheat)
    print(f"\n  Soil pH assessment for grain crops (optimal: 6.0-7.2):")
    for _, row in clipped_soils.iterrows():
        if row['ph'] < 6.0:
            print(f"    {row['soil_id']}: pH {row['ph']:.1f} — TOO ACIDIC "
                  f"({row['area_ha']:.1f} ha may need lime)")
        elif row['ph'] > 7.5:
            print(f"    {row['soil_id']}: pH {row['ph']:.1f} — ALKALINE "
                  f"({row['area_ha']:.1f} ha — micronutrient availability risk)")
        else:
            print(f"    {row['soil_id']}: pH {row['ph']:.1f} — OK")

    return soil_stats


def visualize_soil_map(
    boundary: gpd.GeoDataFrame,
    clipped_soils: gpd.GeoDataFrame,
    soil_stats: pd.DataFrame
) -> None:
    """
    Create a soil type map with legend, pH choropleth, and OM choropleth.
    """
    # Assign colors to soil types
    soil_types = clipped_soils['soil_type'].unique()
    color_palette = [
        '#8D6E63', '#A1887F', '#6D4C41', '#D7CCC8',
        '#BCAAA4', '#795548', '#4E342E', '#EFEBE9'
    ]
    color_map = {st: color_palette[i % len(color_palette)]
                 for i, st in enumerate(soil_types)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        'Soil Type Analysis — Rolling Prairie Farm, Ellsworth Co., KS\n'
        'Author: Emmanuel Oyekanlu — Principal Data Engineer',
        fontsize=12, fontweight='bold'
    )

    # --- Plot 1: Soil type map ---
    ax1 = axes[0]
    boundary.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=2)
    for _, row in clipped_soils.iterrows():
        color = color_map.get(row['soil_type'], '#CCCCCC')
        gpd.GeoDataFrame([row], geometry='geometry', crs=clipped_soils.crs).plot(
            ax=ax1, color=color, alpha=0.85, edgecolor='white', linewidth=0.5
        )
        centroid = row['geometry'].centroid
        ax1.annotate(
            row['series_name'],
            xy=(centroid.x, centroid.y),
            ha='center', va='center',
            fontsize=7, color='black', fontweight='bold'
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color=color_map[st], label=st)
        for st in sorted(soil_types)
    ]
    ax1.legend(handles=legend_handles, loc='lower right', fontsize=7,
               title='Soil Type', title_fontsize=8)
    ax1.set_title('Soil Types (SSURGO)', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # --- Plot 2: pH choropleth ---
    ax2 = axes[1]
    boundary.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, zorder=5)
    clipped_soils.plot(
        column='ph', ax=ax2,
        cmap='RdYlGn',
        vmin=5.5, vmax=8.0,
        edgecolor='white', linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Soil pH', 'shrink': 0.6}
    )
    for _, row in clipped_soils.iterrows():
        centroid = row['geometry'].centroid
        ax2.annotate(
            f"{row['ph']:.1f}",
            xy=(centroid.x, centroid.y),
            ha='center', va='center',
            fontsize=8, color='black', fontweight='bold'
        )
    ax2.set_title('Soil pH (Green=optimal, Red=suboptimal)',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Easting (m)')

    # --- Plot 3: Organic Matter choropleth ---
    ax3 = axes[2]
    boundary.plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=2, zorder=5)
    clipped_soils.plot(
        column='organic_matter_pct', ax=ax3,
        cmap='YlOrBr',
        edgecolor='white', linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Organic Matter (%)', 'shrink': 0.6}
    )
    for _, row in clipped_soils.iterrows():
        centroid = row['geometry'].centroid
        ax3.annotate(
            f"{row['organic_matter_pct']:.1f}%",
            xy=(centroid.x, centroid.y),
            ha='center', va='center',
            fontsize=8, color='black', fontweight='bold'
        )
    ax3.set_title('Organic Matter % (darker=higher)',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('Easting (m)')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "02_soil_zone_analysis.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("SOIL ZONE ANALYSIS")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load data
    boundary, soils = load_and_project_data()

    # Clip soils to boundary
    clipped_soils = clip_soils_to_boundary(soils, boundary)

    # Compute statistics
    soil_stats = compute_soil_type_statistics(clipped_soils)

    # Visualize
    visualize_soil_map(boundary, clipped_soils, soil_stats)

    # Save outputs
    clipped_soils.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "clipped_soil_types.geojson"),
        driver='GeoJSON'
    )
    soil_stats.drop(columns=['ph_weighted_sum', 'om_weighted_sum'], errors='ignore') \
        .to_csv(os.path.join(OUTPUT_DIR, "soil_type_statistics.csv"), index=False)

    print("\nSaved:")
    print("  output/clipped_soil_types.geojson")
    print("  output/soil_type_statistics.csv")
    print("  output/02_soil_zone_analysis.png")


if __name__ == "__main__":
    main()
