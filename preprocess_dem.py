"""
Meging DEMS, reprojecting for easier calculations, and clipping the DEM.

Input tiles (download from USGS TNM):
    USGS_13_n20w155_20130911.tif
    USGS_13_n20w156_20130911.tif

Output:
    data/lerz_dem_2013.tif   (EPSG:32605, ~9.8 m cells)
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import from_bounds

# ── Input tiles ──────────────────────────────────────────────────────
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
TILE_PATHS = [
    os.path.join(DOWNLOADS, "USGS_13_n20w155_20130911.tif"),
    os.path.join(DOWNLOADS, "USGS_13_n20w156_20130911.tif"),
]

# ── Output ───────────────────────────────────────────────────────────
OUTPUT_PATH = Path("data/lerz_dem_2013.tif")
DST_CRS = "EPSG:32605"

# ── Clip bounds (UTM 32605, metres) ─────────────────────────────────
# Covers Fissure 8 vent through the Kapoho coast
XMIN, XMAX = 289_000, 321_000   # ~32 km E-W
YMIN, YMAX = 2_148_000, 2_165_000  # ~17 km N-S


def main() -> None:
    #  1. Merge tiles 
    print("Merging tiles...")
    srcs = [rasterio.open(p) for p in TILE_PATHS]
    mosaic, mosaic_tf = merge(srcs)
    src_crs = srcs[0].crs
    nodata = srcs[0].nodata
    for s in srcs:
        s.close()
    print(f"  Merged shape: {mosaic.shape}, CRS: {src_crs}")

    #  2. Reproject to UTM
    print("Reprojecting to UTM (EPSG:32605)...")
    h, w = mosaic.shape[1], mosaic.shape[2]
    left = mosaic_tf.c
    top = mosaic_tf.f
    right = left + mosaic_tf.a * w
    bottom = top + mosaic_tf.e * h
    bounds = (left, bottom, right, top)

    transform, width, height = calculate_default_transform(
        src_crs, DST_CRS, w, h, *bounds
    )
    utm = np.empty((height, width), dtype=np.float32)
    reproject(
        mosaic[0], utm,
        src_transform=mosaic_tf, src_crs=src_crs,
        dst_transform=transform, dst_crs=DST_CRS,
        resampling=Resampling.cubic,
        src_nodata=nodata, dst_nodata=np.nan,
    )
    print(f"  Reprojected: {utm.shape}, resolution: {abs(transform.a):.2f} m")

    # 3. Clip to the study area
    print("Clipping to LERZ bounding box...")
    window = from_bounds(XMIN, YMIN, XMAX, YMAX, transform)
    r0 = int(window.row_off)
    r1 = r0 + int(window.height)
    c0 = int(window.col_off)
    c1 = c0 + int(window.width)

    clipped = utm[r0:r1, c0:c1].copy()
    clip_tf = Affine(transform.a, 0, XMIN, 0, transform.e, YMAX)

    print(f"  Clipped: {clipped.shape}, cell_size: {abs(clip_tf.a):.2f} m")
    print(f"  NaN fraction: {np.isnan(clipped).sum() / clipped.size:.4f}")
    print(f"  Elevation range: {np.nanmin(clipped):.1f} – {np.nanmax(clipped):.1f} m")

    # savin
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": clipped.shape[1],
        "height": clipped.shape[0],
        "count": 1,
        "crs": DST_CRS,
        "transform": clip_tf,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(OUTPUT_PATH, "w", **profile) as dst:
        dst.write(clipped, 1)
    print(f"  Saved: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
