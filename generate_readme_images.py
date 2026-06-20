"""
generate_readme_images.py - Repo 06: Agricultural Zone Mapping
Generates illustrative images using only matplotlib + numpy.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os

os.makedirs("images", exist_ok=True)

BG = "#f8f9fa"
DARK = "#212121"
rng = np.random.default_rng(42)


# =============================================================
# IMAGE 1: management_zones.png
# k-means management zone delineation + field grid
# =============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Precision Agriculture — Management Zone Delineation",
             fontsize=14, fontweight='bold', color=DARK, y=0.98)

# Simulate a 20x20 grid of soil/NDVI data
N = 25
x_grid, y_grid = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))

# Soil EC (electrical conductivity) layer
ec = (0.5 + 0.4 * np.sin(3 * x_grid * np.pi) * np.cos(2.5 * y_grid * np.pi)
      + 0.1 * rng.standard_normal((N, N)))
ec = np.clip(ec, 0, 1)

# NDVI layer
ndvi = (0.6 + 0.3 * np.cos(2 * x_grid * np.pi) * np.sin(3 * y_grid * np.pi)
        + 0.08 * rng.standard_normal((N, N)))
ndvi = np.clip(ndvi, 0, 1)

# Combined zone layer (simple average)
combined = (ec + ndvi) / 2

# Assign 4 zones by quartile
zone_map = np.zeros_like(combined, dtype=int)
q25, q50, q75 = np.percentile(combined, [25, 50, 75])
zone_map[combined >= q75] = 0  # High
zone_map[(combined >= q50) & (combined < q75)] = 1
zone_map[(combined >= q25) & (combined < q50)] = 2
zone_map[combined < q25] = 3  # Low

zone_colors = ["#1B5E20", "#66BB6A", "#FFA726", "#B71C1C"]
zone_labels = ["Zone 1\n(High potential)", "Zone 2\n(Med-High)",
               "Zone 3\n(Med-Low)", "Zone 4\n(Low potential)"]

# Panel 1: Soil EC input layer
ax = axes[0]
im = ax.imshow(ec, cmap='YlOrBr', aspect='equal', origin='lower',
               extent=[0, 1, 0, 1])
ax.set_title("Input Layer 1:\nSoil EC (electrical conductivity)",
             fontsize=10, fontweight='bold', color=DARK, pad=6)
ax.set_xlabel("X (field fraction)", fontsize=8)
ax.set_ylabel("Y (field fraction)", fontsize=8)
ax.tick_params(labelsize=7)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("EC (dS/m)", fontsize=7.5)
cbar.ax.tick_params(labelsize=7)

# Panel 2: NDVI layer
ax = axes[1]
im2 = ax.imshow(ndvi, cmap='RdYlGn', aspect='equal', origin='lower',
                extent=[0, 1, 0, 1])
ax.set_title("Input Layer 2:\nNDVI (vegetation index)",
             fontsize=10, fontweight='bold', color=DARK, pad=6)
ax.set_xlabel("X (field fraction)", fontsize=8)
ax.set_ylabel("Y (field fraction)", fontsize=8)
ax.tick_params(labelsize=7)
cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cbar2.set_label("NDVI", fontsize=7.5)
cbar2.ax.tick_params(labelsize=7)

# Panel 3: Management zones output
ax = axes[2]
zone_cmap = mcolors.ListedColormap(zone_colors)
im3 = ax.imshow(zone_map, cmap=zone_cmap, aspect='equal', origin='lower',
                extent=[0, 1, 0, 1], vmin=-0.5, vmax=3.5)
ax.set_title("Output:\nk-means Management Zones (k=4)",
             fontsize=10, fontweight='bold', color=DARK, pad=6)
ax.set_xlabel("X (field fraction)", fontsize=8)
ax.set_ylabel("Y (field fraction)", fontsize=8)
ax.tick_params(labelsize=7)

legend_patches = [mpatches.Patch(facecolor=c, label=l, edgecolor='white', linewidth=1)
                  for c, l in zip(zone_colors, zone_labels)]
ax.legend(handles=legend_patches, loc='lower right', fontsize=7,
          framealpha=0.9, title='Zones', title_fontsize=7.5)

fig.tight_layout(pad=2)
fig.savefig("images/management_zones.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: images/management_zones.png")


# =============================================================
# IMAGE 2: prescription_map.png
# Variable Rate prescription map with application rates per zone
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("Variable Rate Prescription Map — Nitrogen Application",
             fontsize=14, fontweight='bold', color=DARK, y=0.98)

# Prescription zones (irregular polygons representing field zones)
# Simulated as colored regions on a 500x300m field
field_zones = [
    # (x, y, w, h, zone, n_rate_kg_ha, label)
    (0.02, 0.55, 0.3,  0.42, 1, 180, "Zone 1\n180 kg/ha N"),
    (0.34, 0.55, 0.28, 0.42, 2, 150, "Zone 2\n150 kg/ha N"),
    (0.64, 0.55, 0.34, 0.42, 1, 180, "Zone 1\n180 kg/ha N"),
    (0.02, 0.08, 0.20, 0.44, 3, 120, "Zone 3\n120 kg/ha N"),
    (0.24, 0.08, 0.38, 0.44, 4,  80, "Zone 4\n 80 kg/ha N"),
    (0.64, 0.08, 0.20, 0.44, 2, 150, "Zone 2\n150 kg/ha N"),
    (0.86, 0.08, 0.12, 0.44, 3, 120, "Zone 3\n120 kg/ha N"),
    (0.02, 0.04, 0.96, 0.02, 5,   0, "Road / boundary"),
]

n_rate_cmap = cm.get_cmap('RdYlGn')
n_rates = [180, 150, 120, 80, 0]
n_norm = mcolors.Normalize(vmin=0, vmax=200)

# LEFT: Prescription choropleth
ax = axes[0]
ax.set_facecolor("#F5F5DC")
ax.set_title("Prescription Map — Variable N Rate\n(Zone-based VRA)",
             fontsize=11, fontweight='bold', color=DARK, pad=8)

for (x, y, w, h, zone, rate, lbl) in field_zones:
    color = n_rate_cmap(n_norm(rate))
    poly = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                           facecolor=color, edgecolor='white',
                           linewidth=2, alpha=0.9, zorder=2)
    ax.add_patch(poly)
    if rate > 0:
        ax.text(x + w / 2, y + h / 2, lbl, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.35),
                zorder=3)

ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Field X (relative)", fontsize=9)
ax.set_ylabel("Field Y (relative)", fontsize=9)
ax.grid(True, linestyle='--', alpha=0.2)
ax.tick_params(labelsize=8)

sm = plt.cm.ScalarMappable(cmap=n_rate_cmap, norm=n_norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("N Application Rate (kg/ha)", fontsize=9)
cbar.ax.tick_params(labelsize=8)

# RIGHT: Bar chart of per-zone application rates
ax = axes[1]
ax.set_facecolor(BG)
ax.set_title("Per-Zone Nitrogen Rates vs Uniform Application\n(Cost & Environmental Analysis)",
             fontsize=11, fontweight='bold', color=DARK, pad=8)

zones_x = ["Zone 1\n(High)", "Zone 2\n(Med-H)", "Zone 3\n(Med-L)", "Zone 4\n(Low)"]
vra_rates = [180, 150, 120, 80]
uniform_rate = 140

x_pos = np.arange(len(zones_x))
bar_colors = [n_rate_cmap(n_norm(r)) for r in vra_rates]

bars = ax.bar(x_pos, vra_rates, color=bar_colors, edgecolor='white',
              linewidth=1.5, width=0.55, label='VRA prescription', zorder=3)
ax.axhline(uniform_rate, color='#D32F2F', linewidth=2.5, linestyle='--',
           label=f'Uniform rate ({uniform_rate} kg/ha)', zorder=4)
ax.fill_between([-0.4, 3.4], uniform_rate, uniform_rate, alpha=0.1,
                color='#D32F2F')

for bar, val in zip(bars, vra_rates):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
            f"{val} kg/ha", ha='center', fontsize=9.5, fontweight='bold',
            color=DARK, zorder=5)
    diff = val - uniform_rate
    diff_txt = f"+{diff}" if diff > 0 else str(diff)
    diff_color = "#D32F2F" if diff > 0 else "#1B5E20"
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
            diff_txt, ha='center', va='center', fontsize=9, color='white',
            fontweight='bold', zorder=5)

ax.set_xticks(x_pos)
ax.set_xticklabels(zones_x, fontsize=9)
ax.set_ylabel("N Application Rate (kg/ha)", fontsize=10)
ax.set_ylim(0, 210)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(fontsize=9, framealpha=0.9)
ax.tick_params(labelsize=9)

savings = sum((uniform_rate - r) * 50 for r in vra_rates if r < uniform_rate)
ax.text(0.5, 0.04,
        f"Estimated input savings vs uniform: ~${abs(savings):.0f}/season",
        ha='center', transform=ax.transAxes, fontsize=9,
        color='#1B5E20', fontweight='bold',
        bbox=dict(boxstyle='round', fc='#E8F5E9', ec='#1B5E20', lw=1.5))

fig.tight_layout(pad=2)
fig.savefig("images/prescription_map.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: images/prescription_map.png")


# =============================================================
# IMAGE 3: crop_rotation.png
# Multi-year crop rotation tracking
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("Crop Rotation Tracking — Multi-Year Field Management",
             fontsize=14, fontweight='bold', color=DARK, y=0.98)

years = [2021, 2022, 2023, 2024, 2025]
fields_cr = ["Field A", "Field B", "Field C", "Field D", "Field E",
             "Field F", "Field G", "Field H"]

crop_color_map = {
    "corn":    "#FFD600",
    "soybean": "#43A047",
    "wheat":   "#FF8F00",
    "fallow":  "#BDBDBD",
    "canola":  "#8BC34A",
}

# Rotation sequences per field
rotations = [
    ["corn",    "soybean", "wheat",   "corn",    "soybean"],
    ["soybean", "wheat",   "corn",    "soybean", "wheat"  ],
    ["wheat",   "corn",    "soybean", "wheat",   "corn"   ],
    ["corn",    "corn",    "soybean", "fallow",  "wheat"  ],
    ["fallow",  "corn",    "corn",    "soybean", "wheat"  ],
    ["soybean", "canola",  "wheat",   "corn",    "soybean"],
    ["corn",    "soybean", "soybean", "wheat",   "corn"   ],
    ["wheat",   "fallow",  "corn",    "canola",  "soybean"],
]

# LEFT: Rotation grid (heat map style)
ax = axes[0]
ax.set_facecolor(BG)
ax.set_title("Crop Rotation Matrix\n(5-year history per field)",
             fontsize=11, fontweight='bold', color=DARK, pad=8)

for fi, field_rot in enumerate(rotations):
    for yi, crop in enumerate(field_rot):
        clr = crop_color_map[crop]
        rect = FancyBboxPatch((yi + 0.05, fi + 0.05), 0.9, 0.9,
                               boxstyle="round,pad=0.05",
                               facecolor=clr, edgecolor='white',
                               linewidth=2, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(yi + 0.5, fi + 0.5, crop[:4], ha='center', va='center',
                fontsize=8, fontweight='bold', color=DARK, zorder=3)

ax.set_xlim(0, 5)
ax.set_ylim(0, len(rotations))
ax.set_xticks(np.arange(5) + 0.5)
ax.set_xticklabels(years, fontsize=10)
ax.set_yticks(np.arange(len(fields_cr)) + 0.5)
ax.set_yticklabels(fields_cr, fontsize=10)
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Field", fontsize=10)
ax.tick_params(length=0)

legend_handles = [mpatches.Patch(facecolor=c, label=k.capitalize(), edgecolor='gray')
                  for k, c in crop_color_map.items()]
ax.legend(handles=legend_handles, loc='lower right', fontsize=8,
          title='Crop', title_fontsize=8.5, framealpha=0.95)

# RIGHT: Crop frequency analysis
ax = axes[1]
ax.set_facecolor(BG)
ax.set_title("Crop Frequency Analysis\n(portfolio composition per year)",
             fontsize=11, fontweight='bold', color=DARK, pad=8)

crop_counts = {}
for year_idx, year in enumerate(years):
    counts = {crop: 0 for crop in crop_color_map}
    for field_rot in rotations:
        counts[field_rot[year_idx]] += 1
    crop_counts[year] = counts

bottoms = np.zeros(len(years))
for crop, clr in crop_color_map.items():
    vals = [crop_counts[yr][crop] for yr in years]
    ax.bar(years, vals, bottom=bottoms, color=clr, edgecolor='white',
           linewidth=1.5, label=crop.capitalize(), width=0.6, zorder=3)
    for yi, (yr, v) in enumerate(zip(years, vals)):
        if v > 0:
            ax.text(yr, bottoms[yi] + v / 2, str(v), ha='center', va='center',
                    fontsize=10, fontweight='bold', color=DARK, zorder=4)
    bottoms += np.array(vals, dtype=float)

ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Number of Fields", fontsize=10)
ax.set_xticks(years)
ax.set_xticklabels(years, fontsize=10)
ax.set_ylim(0, len(fields_cr) + 1)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(fontsize=9, framealpha=0.9, loc='upper right')
ax.tick_params(labelsize=9)
ax.set_yticks(range(0, len(fields_cr) + 1))

fig.tight_layout(pad=2)
fig.savefig("images/crop_rotation.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print("Saved: images/crop_rotation.png")

print("\nAll images generated in images/")
