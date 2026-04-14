"""
Fissure 8 Lava Flow Simulation
================================
Physical lava flow simulation for the 2018 Fissure 8 eruption in the Lower East Rift Zone of Kīlauea, Hawai'i. The model uses a cellular automaton approach with MFD routing and a simple yield-stress-based flow rule. The simulation runs for 48 hours with a fixed eruption rate and outputs GeoTIFFs of lava thickness, arrival time, inundation count, and max thickness, as well as summary statistics and plots.
Requires:
    numpy, rasterio, matplotlib, earthpy, xrspatial, xarray
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import earthpy.spatial as _es
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from rasterio.crs import CRS
from rasterio.transform import Affine
from xrspatial import slope as _xrs_slope

# Parameters

# -- Paths --
DEM_PATH = "data/lerz_dem_2013.tif"
OUTPUT_DIR = "outputs_fissure8/"

# -- Vent (UTM Zone 5N, EPSG:32605) --
VENT_X = 299615.6
VENT_Y = 2153025.5

# -- Time --
DT = 10.0               # timestep (s)
TOTAL_TIME = 172800.0    # 48 hours (s)

# -- Source --
Q_VENT = 150.0           # eruption rate (m³/s)

# -- Physics --
RHO = 2600.0             # lava density (kg/m³)
G = 9.81                 # gravity (m/s²)
TAU_Y = 200.0            # yield strength (Pa)
MIN_THICKNESS = 0.01     # minimum active thickness (m)
MAX_MOVABLE_FRACTION = 0.3

# -- Output --
OUTPUT_EVERY = 200       # print progress every N steps

# -- Domain / Routing --
ALLOW_EDGE_OUTFLOW = False
SLOPE_WEIGHT_EXPONENT = 1.0

N_STEPS = int(TOTAL_TIME / DT)

# Utilities

@dataclass
class RasterMeta:
    transform: Affine
    crs: CRS
    shape: Tuple[int, int]
    nodata: Optional[float]
    bounds: rasterio.coords.BoundingBox
    cell_size: float


def load_dem(path: str | Path) -> Tuple[np.ndarray, RasterMeta]:
    path = Path(path)
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        bounds = src.bounds
        shape = src.shape

    if crs.is_geographic:
        raise ValueError(
            f"DEM CRS ({crs}) is geographic. A projected CRS is required."
        )

    cell_size = abs(transform.a)
    if nodata is not None:
        array[array == nodata] = np.nan

    meta = RasterMeta(
        transform=transform, crs=crs, shape=shape,
        nodata=nodata, bounds=bounds, cell_size=cell_size,
    )
    return array, meta


def xy_to_rowcol(transform: Affine, x: float, y: float) -> Tuple[int, int]:
    col, row = ~transform * (x, y)
    return int(row), int(col)


def rowcol_to_xy(transform: Affine, row: int, col: int) -> Tuple[float, float]:
    x, y = transform * (col + 0.5, row + 0.5)
    return x, y


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    meta: RasterMeta,
    nodata: Optional[float] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nd = nodata if nodata is not None else meta.nodata
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1], count=1,
        dtype="float64", crs=meta.crs, transform=meta.transform, nodata=nd,
    ) as dst:
        dst.write(array.astype(np.float64), 1)

# GIS calls that do slopes and hillshades. getting distances from neighbors off pythagorean theorem and such...

D8_OFFSETS = [
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
]


def neighbor_distances(cell_size: float) -> np.ndarray:
    diag = cell_size * np.sqrt(2.0)
    return np.array([
        cell_size, diag, cell_size, diag,
        cell_size, diag, cell_size, diag,
    ])


def compute_slope_magnitude(z: np.ndarray, cell_size: float) -> np.ndarray:
    ys = np.arange(z.shape[0]) * cell_size
    xs = np.arange(z.shape[1]) * cell_size
    da = xr.DataArray(
        np.nan_to_num(z, nan=0.0), dims=["y", "x"],
        coords={"y": ys, "x": xs},
    )
    slope_deg = _xrs_slope(da).values
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)
    return np.tan(np.radians(slope_deg))


def compute_hillshade(
    z: np.ndarray, cell_size: float,
    azimuth: float = 315.0, altitude: float = 45.0,
) -> np.ndarray:
    hs = _es.hillshade(z, azimuth=azimuth, altitude=altitude)
    return np.clip(hs, 0, 255).astype(np.uint8)


def make_nodata_mask(z: np.ndarray) -> np.ndarray:
    return ~np.isnan(z)

# Multiflow but not outputting a tif everytime or else that would absolutely destroy compute time

def compute_mfd_weights(
    z_surface: np.ndarray, cell_size: float,
    valid_mask: np.ndarray, slope_exponent: float = 1.0,
) -> np.ndarray:
    nrows, ncols = z_surface.shape
    dists = neighbor_distances(cell_size)
    weights = np.zeros((8, nrows, ncols), dtype=np.float64)

    for k, (dr, dc) in enumerate(D8_OFFSETS):
        nr_start = max(0, dr)
        nr_end = nrows + min(0, dr)
        nc_start = max(0, dc)
        nc_end = ncols + min(0, dc)

        src_r_start = max(0, -dr)
        src_r_end = nrows + min(0, -dr)
        src_c_start = max(0, -dc)
        src_c_end = ncols + min(0, -dc)

        dz = (
            z_surface[src_r_start:src_r_end, src_c_start:src_c_end]
            - z_surface[nr_start:nr_end, nc_start:nc_end]
        )
        slope = dz / dists[k]

        nbr_valid = valid_mask[nr_start:nr_end, nc_start:nc_end]
        src_valid = valid_mask[src_r_start:src_r_end, src_c_start:src_c_end]
        good = (slope > 0) & nbr_valid & src_valid

        wt = np.zeros_like(slope)
        wt[good] = slope[good] ** slope_exponent
        weights[k, src_r_start:src_r_end, src_c_start:src_c_end] = wt

    total = weights.sum(axis=0)
    nonzero = total > 0
    for k in range(8):
        weights[k][nonzero] /= total[nonzero]

    return weights


def compute_effective_slope(
    weights: np.ndarray, z_surface: np.ndarray, cell_size: float,
) -> np.ndarray:
    nrows, ncols = z_surface.shape
    dists = neighbor_distances(cell_size)
    s_eff = np.zeros((nrows, ncols), dtype=np.float64)

    for k, (dr, dc) in enumerate(D8_OFFSETS):
        nr_start = max(0, dr)
        nr_end = nrows + min(0, dr)
        nc_start = max(0, dc)
        nc_end = ncols + min(0, dc)

        src_r_start = max(0, -dr)
        src_r_end = nrows + min(0, -dr)
        src_c_start = max(0, -dc)
        src_c_end = ncols + min(0, -dc)

        dz = (
            z_surface[src_r_start:src_r_end, src_c_start:src_c_end]
            - z_surface[nr_start:nr_end, nc_start:nc_end]
        )
        slope = dz / dists[k]
        slope = np.where(np.isfinite(slope) & (slope > 0), slope, 0.0)

        w = weights[k, src_r_start:src_r_end, src_c_start:src_c_end]
        s_eff[src_r_start:src_r_end, src_c_start:src_c_end] += w * slope

    return s_eff


def distribute_lava(
    h_move: np.ndarray, weights: np.ndarray,
    allow_edge_outflow: bool = False,
) -> np.ndarray:
    nrows, ncols = h_move.shape
    h_in = np.zeros((nrows, ncols), dtype=np.float64)

    for k, (dr, dc) in enumerate(D8_OFFSETS):
        src_r_start = max(0, -dr)
        src_r_end = nrows + min(0, -dr)
        src_c_start = max(0, -dc)
        src_c_end = ncols + min(0, -dc)

        dst_r_start = max(0, dr)
        dst_r_end = nrows + min(0, dr)
        dst_c_start = max(0, dc)
        dst_c_end = ncols + min(0, dc)

        contribution = (
            h_move[src_r_start:src_r_end, src_c_start:src_c_end]
            * weights[k, src_r_start:src_r_end, src_c_start:src_c_end]
        )
        h_in[dst_r_start:dst_r_end, dst_c_start:dst_c_end] += contribution

    return h_in

# Physics

def driving_stress(rho, g, h, s_eff):
    return rho * g * h * s_eff


def movement_fraction(tau_d, tau_y, f_max=0.25, epsilon=1e-10):
    f = (tau_d - tau_y) / (tau_d + epsilon)
    return np.clip(f, 0.0, f_max)


def compute_h_move(h_lava, s_eff, rho, g, tau_y, f_max, min_thickness):
    tau_d = driving_stress(rho, g, h_lava, s_eff)
    f = movement_fraction(tau_d, tau_y, f_max)
    h_move = f * h_lava
    h_move[h_lava < min_thickness] = 0.0
    return h_move

# Simulation State

@dataclass
class SimulationState:
    z_base: np.ndarray
    h_lava: np.ndarray
    z_surface: np.ndarray
    arrival_time: np.ndarray
    inundation_count: np.ndarray
    max_thickness: np.ndarray
    time: float = 0.0
    step: int = 0
    total_volume_added: float = 0.0

    @staticmethod
    def initialize(z_base: np.ndarray) -> "SimulationState":
        shape = z_base.shape
        return SimulationState(
            z_base=z_base.copy(),
            h_lava=np.zeros(shape, dtype=np.float64),
            z_surface=z_base.copy(),
            arrival_time=np.full(shape, np.nan, dtype=np.float64),
            inundation_count=np.zeros(shape, dtype=np.int64),
            max_thickness=np.zeros(shape, dtype=np.float64),
        )

    def update_surface(self):
        np.add(self.z_base, self.h_lava, out=self.z_surface)

    def update_diagnostics(self, min_thickness: float):
        active = self.h_lava >= min_thickness
        newly_reached = active & np.isnan(self.arrival_time)
        self.arrival_time[newly_reached] = self.time
        self.inundation_count[active] += 1
        np.maximum(self.max_thickness, self.h_lava, out=self.max_thickness)

# Model with a little trick i learned when i was making PyRo-FOAMS. I only compute within a box where the lava actually lives, rather than over the entire DEM every step. I'm not a HPC guy (see: Python) but I can do this at least.

def _active_bbox(
    h_lava: np.ndarray, vent_mask: np.ndarray,
    shape: Tuple[int, int], margin: int = 10,
) -> Tuple[int, int, int, int]:
    active = (h_lava > 0) | vent_mask
    rows, cols = np.where(active)
    if len(rows) == 0:
        vr, vc = np.where(vent_mask)
        rows, cols = vr, vc
    r0 = max(0, int(rows.min()) - margin)
    r1 = min(shape[0], int(rows.max()) + 1 + margin)
    c0 = max(0, int(cols.min()) - margin)
    c1 = min(shape[1], int(cols.max()) + 1 + margin)
    return r0, r1, c0, c1


class LavaFlowModel:
    def __init__(
        self, z_base: np.ndarray, meta: RasterMeta,
        vent_cells: List[Tuple[int, int]],
    ) -> None:
        self.z_base = z_base
        self.meta = meta
        self.vent_cells = vent_cells
        self.cell_area = meta.cell_size ** 2
        self.valid_mask = make_nodata_mask(z_base)

        self.state = SimulationState.initialize(z_base)

        self.vent_mask = np.zeros(z_base.shape, dtype=bool)
        for r, c in vent_cells:
            self.vent_mask[r, c] = True
        self.n_vent_cells = int(self.vent_mask.sum())

    def run(self) -> SimulationState:
        print(f"Running {N_STEPS} timesteps ({TOTAL_TIME} s, dt={DT} s)")
        for i in range(N_STEPS):
            self._step()
            if (i + 1) % OUTPUT_EVERY == 0:
                vol_stored = float(np.nansum(self.state.h_lava) * self.cell_area)
                r0, r1, c0, c1 = _active_bbox(
                    self.state.h_lava, self.vent_mask, self.z_base.shape, margin=0,
                )
                active_size = (r1 - r0) * (c1 - c0)
                print(
                    f"  Step {i + 1}/{N_STEPS} | "
                    f"t = {self.state.time:.1f} s | "
                    f"vol_added = {self.state.total_volume_added:.1f} m³ | "
                    f"vol_stored = {vol_stored:.1f} m³ | "
                    f"bbox = {r1-r0}×{c1-c0} ({active_size} cells)"
                )
        print("Simulation complete.")
        return self.state

    def _step(self) -> None:
        state = self.state
        nrows, ncols = self.z_base.shape

        # Add lava at vent
        dv = Q_VENT * DT
        if self.n_vent_cells > 0:
            dh = dv / (self.n_vent_cells * self.cell_area)
            state.h_lava[self.vent_mask] += dh
            state.total_volume_added += dv

        # Active bounding box
        r0, r1, c0, c1 = _active_bbox(
            state.h_lava, self.vent_mask, (nrows, ncols), margin=5,
        )

        state.update_surface()

        z_sub = state.z_surface[r0:r1, c0:c1]
        h_sub = state.h_lava[r0:r1, c0:c1]
        valid_sub = self.valid_mask[r0:r1, c0:c1]

        # MFD routing (sub-region)
        weights_sub = compute_mfd_weights(
            z_sub, self.meta.cell_size, valid_sub,
            slope_exponent=SLOPE_WEIGHT_EXPONENT,
        )

        # Physics (sub-region)
        s_eff_sub = compute_effective_slope(weights_sub, z_sub, self.meta.cell_size)
        h_move_sub = compute_h_move(
            h_sub, s_eff_sub,
            rho=RHO, g=G, tau_y=TAU_Y,
            f_max=MAX_MOVABLE_FRACTION,
            min_thickness=MIN_THICKNESS,
        )

        # Distribute lava
        h_in_sub = distribute_lava(
            h_move_sub, weights_sub, allow_edge_outflow=ALLOW_EDGE_OUTFLOW,
        )
        state.h_lava[r0:r1, c0:c1] += h_in_sub - h_move_sub

        np.clip(state.h_lava, 0.0, None, out=state.h_lava)
        state.h_lava[state.h_lava < MIN_THICKNESS * 0.01] = 0.0

        # Diagnostics
        state.time += DT
        state.step += 1
        state.update_diagnostics(MIN_THICKNESS)

# Outputs

def save_outputs(state: SimulationState, meta: RasterMeta, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_geotiff(output_dir / "lava_thickness.tif", state.h_lava, meta, nodata=-9999.0)
    write_geotiff(output_dir / "z_surface.tif", state.z_surface, meta, nodata=-9999.0)

    arrival = state.arrival_time.copy()
    arrival[np.isnan(arrival)] = -9999.0
    write_geotiff(output_dir / "arrival_time.tif", arrival, meta, nodata=-9999.0)

    write_geotiff(
        output_dir / "inundation_count.tif",
        state.inundation_count.astype(np.float64), meta, nodata=-9999.0,
    )
    write_geotiff(output_dir / "max_thickness.tif", state.max_thickness, meta, nodata=-9999.0)
    print(f"GeoTIFFs saved to {output_dir.resolve()}")


def save_summary(state: SimulationState, meta: RasterMeta, output_dir: Path) -> None:
    cell_area = meta.cell_size ** 2
    vol_stored = float(np.nansum(state.h_lava) * cell_area)
    active_cells = int(np.sum(state.h_lava >= 0.01))
    max_h = float(np.nanmax(state.h_lava)) if np.any(state.h_lava > 0) else 0.0
    area_inundated = active_cells * cell_area

    lines = [
        "=== Lava Flow Simulation Summary ===",
        f"Total time:        {state.time:.1f} s",
        f"Total steps:       {state.step}",
        f"Volume added:      {state.total_volume_added:.1f} m³",
        f"Volume stored:     {vol_stored:.1f} m³",
        f"Volume balance:    {state.total_volume_added - vol_stored:.2f} m³ (difference)",
        f"Active cells:      {active_cells}",
        f"Inundated area:    {area_inundated:.1f} m²",
        f"Max thickness:     {max_h:.3f} m",
    ]
    text = "\n".join(lines) + "\n"
    (output_dir / "summary.txt").write_text(text)
    print(text)

# Plots for me to make sure this is doing what it needs to

def _extent(meta: RasterMeta) -> list:
    b = meta.bounds
    return [b.left, b.right, b.bottom, b.top]


def plot_dem(z, meta, title="Base DEM", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(z, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    im = ax.imshow(z, extent=ext, cmap="terrain", alpha=0.6, origin="upper")
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_dem_with_vent(z, meta, vent_cells, title="DEM with Vent", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(z, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    im = ax.imshow(z, extent=ext, cmap="terrain", alpha=0.6, origin="upper")
    plt.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    for r, c in vent_cells:
        vx, vy = rowcol_to_xy(meta.transform, r, c)
        ax.plot(vx, vy, "r^", markersize=14, markeredgecolor="k", label="Vent")
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    ax.legend()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_lava_thickness(state, meta, title="Final Lava Thickness", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(state.z_base, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    h_masked = np.ma.masked_where(state.h_lava < 0.01, state.h_lava)
    im = ax.imshow(h_masked, extent=ext, cmap="hot_r", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Lava Thickness (m)", shrink=0.7)
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_arrival_time(state, meta, title="First-Arrival Time", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(state.z_base, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    arr_masked = np.ma.masked_where(np.isnan(state.arrival_time), state.arrival_time)
    im = ax.imshow(arr_masked, extent=ext, cmap="viridis_r", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Arrival Time (s)", shrink=0.7)
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_inundation_count(state, meta, title="Inundation Count", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(state.z_base, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    cnt_masked = np.ma.masked_where(state.inundation_count == 0, state.inundation_count)
    im = ax.imshow(cnt_masked, extent=ext, cmap="YlOrRd", origin="upper", alpha=0.8)
    plt.colorbar(im, ax=ax, label="Inundation Count (timesteps)", shrink=0.7)
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig


def plot_extent_over_dem(state, meta, vent_cells=None, title="Lava Extent over DEM", save_path=None, show=True):
    ext = _extent(meta)
    fig, ax = plt.subplots(figsize=(10, 8))
    hs = compute_hillshade(np.nan_to_num(state.z_base, nan=0.0), meta.cell_size)
    ax.imshow(hs, extent=ext, cmap="gray", origin="upper")
    ax.imshow(state.z_base, extent=ext, cmap="terrain", alpha=0.5, origin="upper")
    lava_extent = np.ma.masked_where(state.h_lava < 0.01, np.ones_like(state.h_lava))
    ax.imshow(lava_extent, extent=ext, cmap="Reds", alpha=0.5, origin="upper")
    if vent_cells:
        for r, c in vent_cells:
            vx, vy = rowcol_to_xy(meta.transform, r, c)
            ax.plot(vx, vy, "r^", markersize=14, markeredgecolor="k", label="Vent")
    ax.set_title(title); ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show: plt.show()
    return fig

# Main

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load DEM
    print(f"Loading DEM: {DEM_PATH}")
    z_base, meta = load_dem(DEM_PATH)
    print(f"  Shape: {meta.shape}  Cell size: {meta.cell_size:.2f} m  CRS: {meta.crs}")

    # Resolve vent
    row, col = xy_to_rowcol(meta.transform, VENT_X, VENT_Y)
    nrows, ncols = z_base.shape
    if row < 0 or row >= nrows or col < 0 or col >= ncols:
        raise ValueError(f"Vent ({row}, {col}) is outside DEM bounds.")
    if np.isnan(z_base[row, col]):
        raise ValueError(f"Vent ({row}, {col}) falls on a nodata cell.")
    vent_cells = [(row, col)]
    print(f"Vent cell: ({row}, {col}), elevation: {z_base[row, col]:.1f} m")

    # Init & run
    model = LavaFlowModel(z_base, meta, vent_cells)
    t_start = time.perf_counter()
    state = model.run()
    t_sim = time.perf_counter() - t_start

    # Save outputs
    save_outputs(state, meta, output_dir)
    save_summary(state, meta, output_dir)

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_dem(z_base, meta, save_path=plots_dir / "dem.png", show=False)
    plot_dem_with_vent(z_base, meta, vent_cells, save_path=plots_dir / "dem_vent.png", show=False)
    plot_lava_thickness(state, meta, save_path=plots_dir / "lava_thickness.png", show=False)
    plot_extent_over_dem(state, meta, vent_cells=vent_cells, save_path=plots_dir / "lava_extent.png", show=False)
    plot_arrival_time(state, meta, save_path=plots_dir / "arrival_time.png", show=False)
    plot_inundation_count(state, meta, save_path=plots_dir / "inundation_count.png", show=False)
    print(f"Plots saved to {plots_dir.resolve()}")

    t_total = time.perf_counter() - t_start
    print(f"\nTotal compute time: {t_total:.1f} s  (simulation: {t_sim:.1f} s, outputs/plots: {t_total - t_sim:.1f} s)")
    print("Done.")


if __name__ == "__main__":
    main()
