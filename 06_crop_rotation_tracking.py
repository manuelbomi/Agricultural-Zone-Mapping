"""
06_crop_rotation_tracking.py
=============================
Author: Emmanuel Oyekanlu — Principal Data Engineer

Tracks multi-year crop rotation data and identifies rotation patterns:
  - Simulate crop history data for fields across 2021–2025
  - Merge GeoDataFrames across years
  - Compute rotation sequences per field
  - Flag fields violating rotation best practices
  - Generate a crop history timeline visualization
  - Export rotation database as GeoJSON + CSV

Crop rotation context:
    Rotating crops (changing crop species each year) on the same land:
      - Breaks pest and disease cycles specific to one crop
      - Reduces nitrogen fertilizer needs (legumes fix atmospheric N)
      - Improves soil structure and organic matter
      - Required by some USDA conservation programs

    Best practice rotations for Kansas:
      - Corn → Soybean (most common — corn benefits from soybean N credit)
      - Corn → Wheat → Soybean (3-year rotation)
      - Avoid: continuous corn (>2 years) — rootworm pressure, yield drag
      - Avoid: continuous soybean — SCN (soybean cyst nematode) build-up

Data engineering aspects:
    Tracking rotation across years requires:
      - Consistent field boundary IDs across years (stable identifiers)
      - Temporal join: merge yearly crop records to field geometries
      - Sequence detection: identify the pattern across N years
      - Anomaly detection: flag violations of rotation guidelines

    In production: this data comes from:
      - USDA FSA annual acreage reports (Form FSA-578)
      - Farm management software (Granular, Farmers Edge)
      - Remote sensing classification (Cropland Data Layer from USDA NASS)

Run:
    python 06_crop_rotation_tracking.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BOUNDARY_PATH = os.path.join(DATA_DIR, "farm_boundary.geojson")
UTM_CRS = 'EPSG:32614'

# Crop rotation study years
YEARS = [2021, 2022, 2023, 2024, 2025]

# ---------------------------------------------------------------------------
# Best practice rotation rules
# ---------------------------------------------------------------------------
ROTATION_RULES = {
    'max_consecutive_corn': 2,        # Max years corn can be grown continuously
    'max_consecutive_soybean': 2,     # Max years soy can be continuous (SCN risk)
    'required_break_after_corn': 1,   # Minimum 1 non-corn year after continuous corn
    'discouraged_sequences': [        # Flag these as high concern
        ('corn', 'corn', 'corn'),
        ('soybean', 'soybean', 'soybean'),
    ],
    'preferred_sequences': [
        ('corn', 'soybean'),
        ('corn', 'soybean', 'corn'),
        ('corn', 'wheat', 'soybean'),
        ('soybean', 'corn'),
    ]
}

# Crop nitrogen credits (lb N/acre) from previous year's crop
N_CREDIT_TABLE = {
    'soybean': 40,    # Legume — fixes atmospheric N
    'alfalfa': 80,    # High N fixer (if full stand)
    'corn': 0,        # Grass — no N credit
    'wheat': 0,       # Grass — no N credit
    'fallow': 0,      # No credit
    'sorghum': 0,
}

# Crop colors for visualization
CROP_COLORS = {
    'corn':     '#F9A825',   # Corn yellow
    'soybean':  '#2E7D32',   # Soybean green
    'wheat':    '#D4AC0D',   # Wheat golden
    'sorghum':  '#BF360C',   # Sorghum red-orange
    'fallow':   '#9E9E9E',   # Gray for fallow
    'alfalfa':  '#1B5E20',   # Dark green
}


def create_field_grid(boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a grid of field sub-units within the farm boundary.
    Each sub-unit represents an independently managed field.
    """
    from shapely.geometry import box as shapely_box

    farm_geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = farm_geom.bounds

    # 3×2 grid = 6 fields
    n_cols, n_rows = 3, 2
    cell_w = (maxx - minx) / n_cols
    cell_h = (maxy - miny) / n_rows

    fields = []
    field_num = 1

    for row in range(n_rows):
        for col in range(n_cols):
            cell_minx = minx + col * cell_w
            cell_miny = miny + row * cell_h
            cell_box = shapely_box(cell_minx, cell_miny,
                                   cell_minx + cell_w, cell_miny + cell_h)
            clipped = cell_box.intersection(farm_geom)

            if clipped.area > 0:
                fields.append({
                    'field_id': f'F{field_num:02d}',
                    'field_name': f'Block {chr(ord("A") + col)}{row+1}',
                    'area_ha': round(clipped.area / 10_000, 1),
                    'geometry': clipped
                })
                field_num += 1

    return gpd.GeoDataFrame(fields, crs=boundary.crs)


def simulate_crop_history(fields_gdf: gpd.GeoDataFrame) -> dict[int, gpd.GeoDataFrame]:
    """
    Simulate crop history for each field for each year (2021-2025).

    Uses realistic rotation patterns:
      - Some fields follow good corn-soybean rotation
      - Some fields have continuous corn (to demonstrate violation detection)
      - Some fields have 3-year rotations with wheat

    Returns a dict: year → GeoDataFrame with crop planted that year.
    """
    np.random.seed(7)
    field_ids = fields_gdf['field_id'].tolist()
    n_fields = len(field_ids)

    # Pre-define rotation sequences for each field
    # These represent realistic farm management decisions
    rotations = {
        'F01': ['corn', 'soybean', 'corn', 'soybean', 'corn'],
        'F02': ['soybean', 'corn', 'soybean', 'corn', 'soybean'],
        'F03': ['corn', 'corn', 'corn', 'soybean', 'corn'],    # Violation: 3 consecutive corn
        'F04': ['wheat', 'soybean', 'corn', 'wheat', 'soybean'],
        'F05': ['soybean', 'soybean', 'soybean', 'corn', 'soybean'],  # Violation: 3 consec soy
        'F06': ['corn', 'wheat', 'soybean', 'corn', 'wheat'],
    }

    # Fill any fields not pre-defined with random but valid rotation
    for fid in field_ids:
        if fid not in rotations:
            rotation = []
            prev = np.random.choice(['corn', 'soybean'])
            for _ in YEARS:
                if prev == 'corn':
                    next_crop = 'soybean' if np.random.random() > 0.2 else 'corn'
                else:
                    next_crop = 'corn' if np.random.random() > 0.3 else 'soybean'
                rotation.append(next_crop)
                prev = next_crop
            rotations[fid] = rotation

    # Build per-year GeoDataFrames
    yearly_gdfs = {}
    for year_idx, year in enumerate(YEARS):
        year_records = []
        for _, field in fields_gdf.iterrows():
            fid = field['field_id']
            crop = rotations.get(fid, ['corn'] * len(YEARS))[year_idx]

            # Simulate yield based on crop and rotation history
            if year_idx > 0:
                prev_crop = rotations.get(fid, ['corn'] * len(YEARS))[year_idx - 1]
                n_credit = N_CREDIT_TABLE.get(prev_crop, 0)
            else:
                n_credit = 0
                prev_crop = 'unknown'

            # Yield varies by crop type and rotation benefit
            base_yields = {
                'corn': 10.5, 'soybean': 3.2, 'wheat': 3.8,
                'sorghum': 7.5, 'fallow': 0, 'alfalfa': 14.0
            }
            base = base_yields.get(crop, 8.0)
            rotation_bonus = 0.5 if prev_crop != crop else -0.3  # Rotation benefit
            weather_variation = np.random.normal(0, 0.4)
            simulated_yield = round(max(0, base + rotation_bonus + weather_variation), 1)

            year_records.append({
                'field_id': fid,
                'field_name': field['field_name'],
                'year': year,
                'crop': crop,
                'prev_crop': prev_crop,
                'yield_t_ha': simulated_yield,
                'n_credit_prev_lb_ac': n_credit,
                'area_ha': field['area_ha'],
                'geometry': field['geometry']
            })

        yearly_gdfs[year] = gpd.GeoDataFrame(year_records, crs=fields_gdf.crs)

    return yearly_gdfs, rotations


def detect_rotation_violations(
    rotations: dict[str, list]
) -> pd.DataFrame:
    """
    Analyze rotation sequences and flag violations.

    Checks:
      1. Consecutive same-crop (>max_consecutive threshold)
      2. Known high-risk sequences (continuous corn/soy)
      3. Missing rotation diversity (only 1 crop in 5 years)

    Parameters
    ----------
    rotations : dict
        field_id → list of crop names (one per year in YEARS order).

    Returns
    -------
    DataFrame
        Violation report per field.
    """
    print("=" * 65)
    print("ROTATION VIOLATION ANALYSIS")
    print("=" * 65)

    records = []

    for field_id, sequence in rotations.items():
        violations = []
        warnings = []
        sequence_str = ' → '.join(sequence)

        # Check consecutive crop counts
        max_consec = {'corn': 0, 'soybean': 0}
        current_consec = {}

        for crop in sequence:
            for c in max_consec:
                if crop == c:
                    current_consec[c] = current_consec.get(c, 0) + 1
                    max_consec[c] = max(max_consec[c], current_consec[c])
                else:
                    current_consec[c] = 0

        if max_consec.get('corn', 0) > ROTATION_RULES['max_consecutive_corn']:
            violations.append(
                f"CONTINUOUS_CORN: {max_consec['corn']} consecutive years"
            )

        if max_consec.get('soybean', 0) > ROTATION_RULES['max_consecutive_soybean']:
            violations.append(
                f"CONTINUOUS_SOYBEAN: {max_consec['soybean']} consecutive years"
            )

        # Check for discouraged sequences (as sub-tuples)
        for dis_seq in ROTATION_RULES['discouraged_sequences']:
            n = len(dis_seq)
            for i in range(len(sequence) - n + 1):
                window = tuple(sequence[i:i+n])
                if window == dis_seq:
                    violations.append(
                        f"DISCOURAGED_SEQUENCE: {' → '.join(dis_seq)}"
                    )
                    break

        # Check diversity
        unique_crops = set(sequence)
        if len(unique_crops) == 1:
            warnings.append(f"LOW_DIVERSITY: Only {list(unique_crops)[0]} in {len(YEARS)} years")

        # Check if preferred sequence exists
        has_preferred = False
        for pref_seq in ROTATION_RULES['preferred_sequences']:
            n = len(pref_seq)
            for i in range(len(sequence) - n + 1):
                if tuple(sequence[i:i+n]) == pref_seq:
                    has_preferred = True
                    break
            if has_preferred:
                break

        if not has_preferred:
            warnings.append("NO_PREFERRED_SEQUENCE: Consider corn-soybean alternation")

        status = 'VIOLATION' if violations else ('WARNING' if warnings else 'OK')

        records.append({
            'field_id': field_id,
            'sequence': sequence_str,
            'unique_crops': len(unique_crops),
            'max_consec_corn': max_consec.get('corn', 0),
            'max_consec_soy': max_consec.get('soybean', 0),
            'status': status,
            'violations': ' | '.join(violations) if violations else 'None',
            'warnings': ' | '.join(warnings) if warnings else 'None',
        })

    report_df = pd.DataFrame(records)

    # Print report
    print(f"\n{'Field':<8} {'Status':<12} {'Max Corn':>9} {'Max Soy':>8} {'Crops':>6}")
    print("-" * 55)
    for _, row in report_df.iterrows():
        print(f"{row['field_id']:<8} {row['status']:<12} "
              f"{row['max_consec_corn']:>9} {row['max_consec_soy']:>8} "
              f"{row['unique_crops']:>6}")

    violations_only = report_df[report_df['status'] == 'VIOLATION']
    warnings_only = report_df[report_df['status'] == 'WARNING']
    ok_only = report_df[report_df['status'] == 'OK']

    print(f"\nSummary: {len(ok_only)} OK | {len(warnings_only)} WARNING | "
          f"{len(violations_only)} VIOLATION")

    if len(violations_only) > 0:
        print(f"\nViolations:")
        for _, row in violations_only.iterrows():
            print(f"  {row['field_id']}: {row['sequence']}")
            print(f"    → {row['violations']}")

    if len(warnings_only) > 0:
        print(f"\nWarnings:")
        for _, row in warnings_only.iterrows():
            print(f"  {row['field_id']}: {row['sequence']}")
            print(f"    → {row['warnings']}")

    return report_df


def compute_yield_trends(
    yearly_gdfs: dict[int, gpd.GeoDataFrame]
) -> pd.DataFrame:
    """
    Compute multi-year yield trends per field.

    Yield trends help identify:
      - Fields with declining yield (soil health concern)
      - Rotation benefit quantification (soybean after corn vs continuous corn)
      - Weather year effects (all fields drop → drought year)
    """
    print("\n" + "=" * 65)
    print("MULTI-YEAR YIELD TRENDS")
    print("=" * 65)

    all_records = []
    for year, gdf in yearly_gdfs.items():
        for _, row in gdf.iterrows():
            all_records.append({
                'field_id': row['field_id'],
                'year': year,
                'crop': row['crop'],
                'yield_t_ha': row['yield_t_ha'],
            })

    history_df = pd.DataFrame(all_records)

    # Pivot to wide format for trend analysis
    pivot = history_df.pivot(
        index='field_id', columns='year', values='yield_t_ha'
    )

    # Compute 5-year average and trend (slope of linear fit)
    trends = []
    for field_id in pivot.index:
        yields = pivot.loc[field_id].values
        years_arr = np.array(YEARS)
        if len(yields[~np.isnan(yields)]) >= 2:
            slope, intercept = np.polyfit(years_arr, yields, 1)
        else:
            slope = 0

        trends.append({
            'field_id': field_id,
            'five_yr_avg': round(yields.mean(), 2),
            'trend_t_ha_per_yr': round(slope, 3),
            'trend_direction': 'increasing' if slope > 0.05
            else ('decreasing' if slope < -0.05 else 'stable'),
        })

    trend_df = pd.DataFrame(trends)

    print(f"\n{'Field':<8} {'5yr Avg':>9} {'Trend':>12} {'Direction'}")
    print("-" * 45)
    for _, row in trend_df.iterrows():
        print(f"{row['field_id']:<8} {row['five_yr_avg']:>9.2f} "
              f"{row['trend_t_ha_per_yr']:>+11.3f} "
              f"  {row['trend_direction']}")

    return history_df, trend_df


def visualize_rotation_history(
    fields_gdf: gpd.GeoDataFrame,
    yearly_gdfs: dict[int, gpd.GeoDataFrame],
    rotations: dict,
    violation_df: pd.DataFrame
) -> None:
    """
    Create a multi-panel visualization of crop rotation history.
    Shows each year as a separate map + a rotation timeline chart.
    """
    n_years = len(YEARS)
    fig, axes = plt.subplots(2, n_years, figsize=(5 * n_years, 12))
    fig.suptitle(
        'Crop Rotation History (2021–2025) — Rolling Prairie Farm, KS\n'
        'Author: Emmanuel Oyekanlu — Principal Data Engineer',
        fontsize=13, fontweight='bold', y=1.01
    )

    # --- Top row: Crop maps per year ---
    for col, year in enumerate(YEARS):
        ax = axes[0, col]
        year_gdf = yearly_gdfs[year]

        for _, row in year_gdf.iterrows():
            color = CROP_COLORS.get(row['crop'], '#CCCCCC')
            gpd.GeoDataFrame(
                [{'geometry': row['geometry']}],
                crs=year_gdf.crs
            ).plot(ax=ax, color=color, alpha=0.85, edgecolor='white', linewidth=0.8)

            centroid = row['geometry'].centroid
            ax.annotate(
                f"{row['field_id']}\n{row['crop'][:3].upper()}\n{row['yield_t_ha']}t",
                xy=(centroid.x, centroid.y),
                ha='center', va='center', fontsize=6.5, fontweight='bold'
            )

        ax.set_title(str(year), fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel('Northing (m)', fontsize=8)

    # --- Bottom row: Rotation timeline per field ---
    crop_list = sorted(CROP_COLORS.keys())
    crop_to_num = {c: i for i, c in enumerate(crop_list)}

    for col, year in enumerate(YEARS):
        ax = axes[1, col]
        year_gdf = yearly_gdfs[year]

        for _, row in year_gdf.iterrows():
            color = CROP_COLORS.get(row['crop'], '#CCCCCC')
            gpd.GeoDataFrame(
                [{'geometry': row['geometry']}],
                crs=year_gdf.crs
            ).plot(ax=ax, color=color, alpha=0.9, edgecolor='black', linewidth=1.5)

            # Flag violation fields with red border
            fid = row['field_id']
            viol_row = violation_df[violation_df['field_id'] == fid]
            if len(viol_row) > 0 and viol_row.iloc[0]['status'] == 'VIOLATION':
                gpd.GeoDataFrame(
                    [{'geometry': row['geometry']}],
                    crs=year_gdf.crs
                ).plot(ax=ax, facecolor='none', edgecolor='red',
                       linewidth=3.0, zorder=10)

        ax.set_title(f'{year}\n(Red border = violation)', fontsize=9)
        ax.tick_params(labelsize=6)

    # Shared crop legend
    legend_patches = [
        mpatches.Patch(color=CROP_COLORS[c], label=c.capitalize())
        for c in sorted(CROP_COLORS.keys())
        if c in ['corn', 'soybean', 'wheat']
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               ncol=len(legend_patches), fontsize=10,
               title='Crop Type', title_fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "06_crop_rotation.png")
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved: {out_path}")


def main():
    print("\n" + "=" * 65)
    print("CROP ROTATION TRACKING")
    print("Author: Emmanuel Oyekanlu — Principal Data Engineer")
    print("=" * 65 + "\n")

    # Load boundary and create field grid
    boundary = gpd.read_file(BOUNDARY_PATH).to_crs(UTM_CRS)
    fields_gdf = create_field_grid(boundary)
    print(f"Created {len(fields_gdf)} field sub-units")

    # Simulate 5-year crop history
    print(f"\nSimulating {len(YEARS)}-year ({min(YEARS)}-{max(YEARS)}) crop history...")
    yearly_gdfs, rotations = simulate_crop_history(fields_gdf)

    # Print rotation sequences
    print("\n" + "=" * 65)
    print("FIELD ROTATION SEQUENCES")
    print("=" * 65)
    print(f"\n{'Field':<8} {'2021':<10} {'2022':<10} {'2023':<10} "
          f"{'2024':<10} {'2025':<10}")
    print("-" * 60)
    for fid, seq in sorted(rotations.items()):
        print(f"{fid:<8}", end="")
        for crop in seq:
            print(f" {crop:<10}", end="")
        print()

    # Detect rotation violations
    violation_df = detect_rotation_violations(rotations)

    # Compute yield trends
    history_df, trend_df = compute_yield_trends(yearly_gdfs)

    # Visualize
    visualize_rotation_history(fields_gdf, yearly_gdfs, rotations, violation_df)

    # --- Export rotation database ---
    # Full history as CSV
    history_df.to_csv(
        os.path.join(OUTPUT_DIR, "crop_rotation_history.csv"), index=False
    )

    # Violation report
    violation_df.to_csv(
        os.path.join(OUTPUT_DIR, "rotation_violation_report.csv"), index=False
    )

    # 2025 prescription year crop map
    year_2025 = yearly_gdfs[2025]
    year_2025.to_crs('EPSG:4326').to_file(
        os.path.join(OUTPUT_DIR, "crop_map_2025.geojson"),
        driver='GeoJSON'
    )

    print("\nOutputs saved:")
    print("  output/crop_rotation_history.csv")
    print("  output/rotation_violation_report.csv")
    print("  output/crop_map_2025.geojson")
    print("  output/06_crop_rotation.png")

    # Print N credit summary (agronomic value of rotation)
    print("\n" + "=" * 65)
    print("NITROGEN CREDIT FROM ROTATION (2025 crop year)")
    print("=" * 65)
    year_2025_df = yearly_gdfs[2025]
    total_n_credit = (year_2025_df['n_credit_prev_lb_ac'] *
                      year_2025_df['area_ha'] * 2.471).sum()
    print(f"\n  Fields with soybean preceding 2025 crop:")
    for _, row in year_2025_df[year_2025_df['n_credit_prev_lb_ac'] > 0].iterrows():
        print(f"    {row['field_id']}: {row['prev_crop']} → {row['crop']} | "
              f"N credit: {row['n_credit_prev_lb_ac']} lb/ac | "
              f"Area: {row['area_ha']:.1f} ha")
    print(f"\n  Total N credit across farm: {total_n_credit:,.0f} lb N "
          f"(${total_n_credit * 0.65:,.0f} saved at $0.65/lb N)")


if __name__ == "__main__":
    main()
