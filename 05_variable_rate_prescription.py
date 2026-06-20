"""
05_variable_rate_prescription.py
==================================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Generates a variable rate fertilizer prescription map from management zones:
  - Load management zones with soil properties
  - Apply agronomic calculation to determine N application rate per zone
  - Export prescription as GeoJSON (for GPS-enabled equipment)
  - Export prescription as CSV (for rate controller import)
  - Compute summary statistics: total fertilizer needed, cost estimate

Variable Rate Application (VRA) background:
    Precision agriculture enables applying different rates of fertilizer,
    seed, or chemicals in different zones of the same field. Benefits:
      - Reduced input cost: apply less where soils already supply nutrients
      - Improved yield: apply more where yield response is highest
      - Environmental protection: avoid over-application near water bodies
      - ROI: typical VRA saves $15-40/acre in fertilizer costs

Nitrogen prescription calculation:
    The most common approach is the "yield goal" method:
        N_needed = (Yield_Goal × N_per_unit) - Soil_N_credit

    For corn in Kansas:
        Yield goal: target yield (based on soil potential and multi-year history)
        N per bushel corn: ~1 lb N per bushel
        Soil N credit: estimated from soil organic matter × 0.2
        Previous legume credit: soybeans fix 50-80 lb N/acre

    This is simplified — actual prescription software uses more complex
    algorithms accounting for N testing, rainfall, efficiency factors, etc.

Output formats:
    - GeoJSON: human-readable, GIS-compatible prescription map
    - CSV: flat file with zone_id and rate_lb_n_ac for rate controller import
    - ISO XML: (mentioned, not implemented here) standard for OEM compatibility

Run:
    python 05_variable_rate_prescription.py
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")
SOIL_PATH = os.path.join(DATA_DIR, "soil_types.geojson")
UTM_CRS = 'EPSG:32614'

# ---------------------------------------------------------------------------
# Agronomic constants
# ---------------------------------------------------------------------------
# Nitrogen application rates (lb N/acre) for Kansas corn
N_PER_BUSHEL = 1.0          # lb N per bushel of corn grain
LB_PER_TON = 22.05          # bushels per tonne (corn)
N_EFFICIENCY = 0.70         # Assumed nitrogen use efficiency (plant uptake/applied)
PRICE_PER_LB_N = 0.65       # USD per lb of nitrogen
ACRES_PER_HA = 2.471        # conversion factor

# Crop-specific yield goals (t/ha) based on soil capability class
YIELD_GOAL_BY_CAPABILITY = {
    'IIe': 13.5,   # Prime cropland, slight erosion hazard
    'IIw': 12.0,   # Prime cropland, wet
    'IIIs': 9.5,   # Moderate limitation, sandy
    'IIIw': 9.0,   # Moderate limitation, wet
    'IVs':  6.5,   # Severe limitation, sandy
}

# Nitrogen rate limits (lb N/acre) per Kansas State Extension
N_RATE_MIN_LB_AC = 80
N_RATE_MAX_LB_AC = 200


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load boundary and soil data, project to metric CRS."""
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(UTM_CRS)
    soils = gpd.read_file(SOIL_PATH).to_crs(UTM_CRS)
    soils = gpd.clip(soils, boundary)  # Clip to boundary
    soils['area_ha'] = soils.geometry.area / 10_000
    return boundary, soils


def compute_n_rate_per_zone(
    soils: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Compute nitrogen application rate for each soil management zone.

    Algorithm (simplified KSU N recommendation):
        1. Assign yield goal based on soil capability class
        2. Convert yield goal to bushel equivalent
        3. Compute N demand: N_demand = yield_goal_bu × N_per_bu / efficiency
        4. Subtract soil N credit: organic matter × mineralization factor × depth
        5. Apply min/max limits
        6. Round to nearest 5 lb/acre (practical rate controller resolution)

    Parameters
    ----------
    soils : GeoDataFrame
        Soil type polygons with ph, organic_matter_pct, capability_class.

    Returns
    -------
    GeoDataFrame
        Soil zones with N rate prescription added.
    """
    print("=" * 65)
    print("NITROGEN PRESCRIPTION CALCULATION")
    print("=" * 65)

    soils = soils.copy()

    # Map capability class to yield goal
    soils['yield_goal_t_ha'] = soils['capability_class'].map(YIELD_GOAL_BY_CAPABILITY)
    soils['yield_goal_t_ha'] = soils['yield_goal_t_ha'].fillna(10.0)  # Default

    # Convert yield goal to bushels/acre (for N calculation in US customary)
    soils['yield_goal_bu_ac'] = (
        soils['yield_goal_t_ha'] * LB_PER_TON * ACRES_PER_HA / 56  # 56 lb/bu corn
    ).round(0)

    # N demand
    soils['n_demand_lb_ac'] = (
        soils['yield_goal_bu_ac'] * N_PER_BUSHEL / N_EFFICIENCY
    ).round(0)

    # Soil N credit from organic matter
    # Rule of thumb: 20 lb N/ac per 1% organic matter in top 12 inches
    soils['soil_n_credit_lb_ac'] = (soils['organic_matter_pct'] * 20).round(0)

    # pH adjustment: very low pH reduces N efficiency
    soils['ph_adjustment_lb_ac'] = np.where(
        soils['ph'] < 6.0, 15,    # Add 15 lb/ac for very acidic soils (less uptake)
        np.where(soils['ph'] > 7.5, -10, 0)  # Reduce 10 lb/ac for alkaline
    )

    # Net N recommendation
    soils['n_rate_raw_lb_ac'] = (
        soils['n_demand_lb_ac']
        - soils['soil_n_credit_lb_ac']
        + soils['ph_adjustment_lb_ac']
    )

    # Apply min/max limits
    soils['n_rate_lb_ac'] = soils['n_rate_raw_lb_ac'].clip(
        lower=N_RATE_MIN_LB_AC, upper=N_RATE_MAX_LB_AC
    )

    # Round to nearest 5 lb/ac (rate controller resolution)
    soils['n_rate_lb_ac'] = (soils['n_rate_lb_ac'] / 5).round() * 5

    # Convert to kg/ha for metric reference
    soils['n_rate_kg_ha'] = (soils['n_rate_lb_ac'] * 1.121).round(1)

    # Compute total N needed for each zone
    soils['zone_area_ac'] = soils['area_ha'] * ACRES_PER_HA
    soils['total_n_lb'] = (soils['n_rate_lb_ac'] * soils['zone_area_ac']).round(0)

    # Assign prescription zone labels
    soils = soils.sort_values('n_rate_lb_ac')
    n_min = soils['n_rate_lb_ac'].min()
    n_max = soils['n_rate_lb_ac'].max()
    n_range = n_max - n_min if n_max > n_min else 1

    def rate_label(rate):
        pct = (rate - n_min) / n_range
        if pct < 0.33:
            return 'Low Rate'
        elif pct < 0.66:
            return 'Medium Rate'
        else:
            return 'High Rate'

    soils['rate_zone'] = soils['n_rate_lb_ac'].apply(rate_label)

    return soils


def print_prescription_table(soils: gpd.GeoDataFrame) -> None:
    """Print the complete prescription table."""
    print("\n" + "=" * 65)
    print("PRESCRIPTION TABLE")
    print("=" * 65)

    display_cols = [
        'soil_id', 'soil_type', 'capability_class',
        'yield_goal_bu_ac', 'n_demand_lb_ac', 'soil_n_credit_lb_ac',
        'n_rate_lb_ac', 'n_rate_kg_ha', 'zone_area_ac', 'rate_zone'
    ]

    print(f"\n{'Zone':<10} {'Soil Type':<22} {'Cap':<5} {'YldGoal':>8} "
          f"{'NDemand':>9} {'SoilCrd':>8} {'N Rate':>8} {'Area(ac)':>9} {'Zone':<12}")
    print("-" * 105)

    for _, row in soils.sort_values('n_rate_lb_ac').iterrows():
        print(f"{row['soil_id']:<10} {row['soil_type']:<22} "
              f"{row['capability_class']:<5} "
              f"{row['yield_goal_bu_ac']:>7.0f}bu "
              f"{row['n_demand_lb_ac']:>8.0f}lb "
              f"{row['soil_n_credit_lb_ac']:>7.0f}lb "
              f"{row['n_rate_lb_ac']:>7.0f}lb "
              f"{row['zone_area_ac']:>8.1f}ac "
              f"  {row['rate_zone']}")

    # Totals
    total_area_ac = soils['zone_area_ac'].sum()
    total_n_lb = soils['total_n_lb'].sum()
    avg_rate = total_n_lb / total_area_ac

    print("\n" + "=" * 65)
    print("PRESCRIPTION SUMMARY")
    print("=" * 65)
    print(f"\n  Total field area     : {total_area_ac:.1f} acres "
          f"({total_area_ac / ACRES_PER_HA:.1f} ha)")
    print(f"  Total N required     : {total_n_lb:,.0f} lb "
          f"({total_n_lb * 0.4536:.0f} kg)")
    print(f"  Average N rate       : {avg_rate:.0f} lb/acre "
          f"({avg_rate * 1.121:.0f} kg/ha)")
    print(f"  Estimated cost       : ${total_n_lb * PRICE_PER_LB_N:,.0f} "
          f"(at ${PRICE_PER_LB_N:.2f}/lb N)")

    # Compare to uniform rate (traditional approach)
    uniform_rate = 160  # lb/ac uniform rate (typical KS corn)
    uniform_total = uniform_rate * total_area_ac
    vra_savings_lb = uniform_total - total_n_lb
    vra_savings_usd = vra_savings_lb * PRICE_PER_LB_N

    print(f"\n  Comparison to uniform rate ({uniform_rate} lb/ac):")
    print(f"    Uniform total N    : {uniform_total:,.0f} lb "
          f"(${uniform_total * PRICE_PER_LB_N:,.0f})")
    print(f"    VRA total N        : {total_n_lb:,.0f} lb "
          f"(${total_n_lb * PRICE_PER_LB_N:,.0f})")
    print(f"    VRA savings        : {abs(vra_savings_lb):,.0f} lb N "
          f"(${abs(vra_savings_usd):,.0f})")
    print(f"    Savings per acre   : ${abs(vra_savings_usd)/total_area_ac:.2f}/acre")


def export_prescription(soils: gpd.GeoDataFrame) -> None:
    """
    Export prescription in multiple formats:
      1. GeoJSON — for GIS visualization and GPS equipment
      2. CSV — for rate controller import (compatible with most field computers)
      3. Summary JSON — for farm management software API
    """
    print("\n" + "=" * 65)
    print("EXPORTING PRESCRIPTION FILES")
    print("=" * 65)

    # --- GeoJSON export ---
    prescription_cols = [
        'soil_id', 'soil_type', 'capability_class',
        'yield_goal_bu_ac', 'n_rate_lb_ac', 'n_rate_kg_ha',
        'zone_area_ac', 'rate_zone', 'total_n_lb', 'geometry'
    ]

    prescription_gdf = soils[prescription_cols].copy()
    prescription_wgs84 = prescription_gdf.to_crs('EPSG:4326')
    geojson_path = os.path.join(OUTPUT_DIR, "nitrogen_prescription.geojson")
    prescription_wgs84.to_file(geojson_path, driver='GeoJSON')
    print(f"\n  GeoJSON: {geojson_path}")

    # --- CSV export (for rate controller) ---
    csv_cols = [
        'soil_id', 'soil_type', 'zone_area_ac',
        'n_rate_lb_ac', 'n_rate_kg_ha', 'rate_zone', 'total_n_lb'
    ]
    csv_df = soils[csv_cols].copy()
    csv_df.columns = [
        'zone_id', 'zone_description', 'area_acres',
        'n_rate_lb_acre', 'n_rate_kg_ha', 'rate_category', 'total_n_lbs'
    ]
    csv_path = os.path.join(OUTPUT_DIR, "nitrogen_prescription.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"  CSV    : {csv_path}")

    # --- Summary JSON for farm management software ---
    summary = {
        'prescription_type': 'nitrogen',
        'crop': 'corn',
        'application_timing': 'pre-plant',
        'year': 2025,
        'farm_id': 'KS-FARM-001',
        'total_area_ac': round(soils['zone_area_ac'].sum(), 1),
        'total_n_lb': round(soils['total_n_lb'].sum(), 0),
        'avg_n_rate_lb_ac': round(soils['total_n_lb'].sum() / soils['zone_area_ac'].sum(), 1),
        'n_zones': len(soils),
        'zones': [
            {
                'zone_id': row['soil_id'],
                'area_ac': round(row['zone_area_ac'], 1),
                'n_rate_lb_ac': int(row['n_rate_lb_ac']),
            }
            for _, row in soils.iterrows()
        ]
    }

    json_path = os.path.join(OUTPUT_DIR, "prescription_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON   : {json_path}")


def visualize_prescription(
    boundary: gpd.GeoDataFrame,
    soils: gpd.GeoDataFrame
) -> None:
    """Visualize the variable rate prescription map."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        'Variable Rate Nitrogen Prescription Map\n'
        'Rolling Prairie Farm, KS — Corn 2025 | Author: Emmanuel Oyekanlu',
        fontsize=12, fontweight='bold'
    )

    # --- Plot 1: N rate choropleth ---
    ax1 = axes[0]
    boundary.plot(ax=ax1, facecolor='none', edgecolor='black', linewidth=2, zorder=5)
    soils.plot(
        column='n_rate_lb_ac', ax=ax1,
        cmap='YlOrRd', edgecolor='white', linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'N Rate (lb/acre)', 'shrink': 0.6}
    )

    for _, row in soils.iterrows():
        centroid = row['geometry'].centroid
        ax1.annotate(
            f"{row['soil_id']}\n{int(row['n_rate_lb_ac'])} lb/ac",
            xy=(centroid.x, centroid.y),
            ha='center', va='center', fontsize=7.5,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
        )

    ax1.set_title('N Application Rate (lb/acre)\nHigher rate = higher yield potential',
                  fontsize=10, fontweight='bold')
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')

    # --- Plot 2: Rate zone categories with yield goal ---
    ax2 = axes[1]
    zone_colors = {
        'Low Rate': '#4CAF50',
        'Medium Rate': '#FF9800',
        'High Rate': '#F44336'
    }

    for _, row in soils.iterrows():
        color = zone_colors.get(row['rate_zone'], '#CCCCCC')
        gpd.GeoDataFrame(
            [{'geometry': row['geometry']}],
            crs=soils.crs
        ).plot(ax=ax2, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)

        centroid = row['geometry'].centroid
        ax2.annotate(
            f"{row['series_name']}\n{row['rate_zone']}\nYG: {row['yield_goal_bu_ac']:.0f}bu",
            xy=(centroid.x, centroid.y),
            ha='center', va='center', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6)
        )

    boundary.plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, zorder=5)

    legend_handles = [
        mpatches.Patch(color=c, label=f'{label} '
                       f'({soils[soils["rate_zone"]==label]["n_rate_lb_ac"].mean():.0f} lb/ac avg)')
        for label, c in zone_colors.items()
        if label in soils['rate_zone'].values
    ]
    ax2.legend(handles=legend_handles, loc='lower right', fontsize=8,
               title='Prescription Zone')
    ax2.set_title('Prescription Zones by Rate Category\nYG = Yield Goal (bu/acre)',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Easting (m)')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "05_prescription_map.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("VARIABLE RATE NITROGEN PRESCRIPTION")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load data
    boundary, soils = load_data()
    print(f"Loaded {len(soils)} soil zones, {soils['area_ha'].sum():.1f} ha total")

    # Compute N prescription
    soils_with_rx = compute_n_rate_per_zone(soils)

    # Print prescription table
    print_prescription_table(soils_with_rx)

    # Export files
    export_prescription(soils_with_rx)

    # Visualize
    visualize_prescription(boundary, soils_with_rx)

    print("\nPrescription generation complete.")
    print("Ready for upload to John Deere Operations Center / Climate FieldView.")


if __name__ == "__main__":
    main()
