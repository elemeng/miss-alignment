"""Iterative anchoring optimization for tilt series alignment.

This module addresses high-tilt images getting stuck in local minima by
progressively expanding the set of reliable tilts while preserving
relative offsets of unreliable tilts.
"""

import os
from typing import Callable
from warpylib import TiltSeries
import torch

from miss_alignment.models import MissAlignment
from .optimize_global import optimize_shifts

_DEBUG = os.environ.get("MISS_DEBUG", "").lower() in ("1", "true", "yes")


def run_iterative_anchoring(
    tilt_series: TiltSeries,
    optimize_fn: Callable[[TiltSeries], tuple[TiltSeries, list[float]]],
    initial_reliable_fraction: float = 1 / 2,
) -> tuple[TiltSeries, list[float]]:
    """Core iterative anchoring logic for global shift optimization.

    Addresses high-tilt images getting stuck in local minima by:
    1. Treating only the central fraction of tilts as "reliable"
    2. Storing relative offsets of unreliable (high angle) tilts
    3. Running the optimizer
    4. Restoring relative structure of unreliable tilts while
       propagating the movement of boundary reliable tilts
    5. Gradually expanding the reliable set (1 tilt per side per iteration)

    Parameters
    ----------
    tilt_series : TiltSeries
        Tilt series to optimize.
    optimize_fn : Callable[[TiltSeries], tuple[TiltSeries, list[float]]]
        Optimization function that takes a TiltSeries and returns
        (optimized_tilt_series, loss_values).
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """
    n_tilts = tilt_series.n_tilts
    sorted_indices = tilt_series.indices_sorted_angle()

    # Calculate initial number of unreliable tilts per side
    n_reliable = int(n_tilts * initial_reliable_fraction)
    n_reliable = max(1, min(n_reliable, n_tilts))
    n_unreliable_total = n_tilts - n_reliable
    n_unreliable_per_side = n_unreliable_total // 2

    all_loss_values = []

    # Track best solution - initialize with current state
    best_loss = float("inf")
    best_offsets_x = tilt_series.tilt_axis_offset_x.clone()
    best_offsets_y = tilt_series.tilt_axis_offset_y.clone()

    while n_unreliable_per_side > 0:
        if _DEBUG:
            print(f"n_unreliable_per_side: {n_unreliable_per_side}")
        # Define boundary in sorted index space
        # sorted_indices[0] is most negative angle, sorted_indices[-1] most positive
        pos_boundary_sorted_idx = n_tilts - 1 - n_unreliable_per_side

        # Store relative offsets for unreliable tilts (chain-like, toward center)
        # gathered in one shot: for each unreliable tilt (sorted-index space),
        # rel = offset[curr] - offset[neighbor toward center].
        # Negative side: sorted indices [0, n_unreliable_per_side), each
        # subtracting the neighbor at i+1; boundary is the first reliable tilt.
        n = n_unreliable_per_side
        neg_sorted = sorted_indices[:n]
        neg_neighbors = sorted_indices[1 : n + 1]
        neg_relative_offsets_x = (
            tilt_series.tilt_axis_offset_x[neg_sorted]
            - tilt_series.tilt_axis_offset_x[neg_neighbors]
        )
        neg_relative_offsets_y = (
            tilt_series.tilt_axis_offset_y[neg_sorted]
            - tilt_series.tilt_axis_offset_y[neg_neighbors]
        )

        # Positive side: sorted indices (pos_boundary_sorted_idx, n_tilts),
        # each subtracting the neighbor at i-1; boundary is the last reliable
        # tilt (sorted_indices[pos_boundary_sorted_idx]).
        pos_sorted = sorted_indices[pos_boundary_sorted_idx + 1 :]
        pos_neighbors = sorted_indices[pos_boundary_sorted_idx : n_tilts - 1]
        pos_relative_offsets_x = (
            tilt_series.tilt_axis_offset_x[pos_sorted]
            - tilt_series.tilt_axis_offset_x[pos_neighbors]
        )
        pos_relative_offsets_y = (
            tilt_series.tilt_axis_offset_y[pos_sorted]
            - tilt_series.tilt_axis_offset_y[pos_neighbors]
        )

        # Run optimization
        tilt_series, loss_values = optimize_fn(tilt_series)
        current_loss = loss_values[-1]
        if _DEBUG:
            print(f"Loss went from {loss_values[0]} to {current_loss}")

        # Check if optimization improved
        if current_loss >= best_loss:
            if _DEBUG:
                print(
                    f"  -> Loss worse ({current_loss:.4f} >= "
                    f"{best_loss:.4f}), reverting"
                )
            # Restore best solution
            tilt_series.tilt_axis_offset_x = best_offsets_x.clone()
            tilt_series.tilt_axis_offset_y = best_offsets_y.clone()
        else:
            # New best - update best loss and offsets (before anchoring)
            best_loss = current_loss
            best_offsets_x = tilt_series.tilt_axis_offset_x.clone()
            best_offsets_y = tilt_series.tilt_axis_offset_y.clone()
            all_loss_values.extend(loss_values)
            if _DEBUG:
                print(f"  -> New best loss: {best_loss:.4f}")

        # ALWAYS apply anchoring restoration to ensure unreliable tilts
        # maintain their structure relative to the current iteration's reliable set
        # Restore unreliable tilts with chain-like relative offsets. Propagation
        # is sequential (each tilt depends on its restored neighbor), computed
        # with cumulative offsets from the boundary outward.
        # Negative side: boundary (first reliable tilt, sorted index
        # n_unreliable_per_side) holds the restored reference; moving outward
        # from it, tilt i (sorted idx n-1-i... i.e. the i-th from the boundary)
        # gets boundary + sum(rel for tilts between it and the boundary).
        # rel[i] = offset[sorted[i]] - offset[sorted[i+1]] with sorted[0] the
        # outermost tilt, so cumsum from the end gives outward propagation.
        if n > 0:
            boundary_x = tilt_series.tilt_axis_offset_x[sorted_indices[n]]
            boundary_y = tilt_series.tilt_axis_offset_y[sorted_indices[n]]
            # out_cum[i] = sum of rel[i:] = offset[sorted[i]] - offset[boundary]
            out_cum_x = torch.flip(
                torch.cumsum(torch.flip(neg_relative_offsets_x, [0]), 0), [0]
            )
            out_cum_y = torch.flip(
                torch.cumsum(torch.flip(neg_relative_offsets_y, [0]), 0), [0]
            )
            tilt_series.tilt_axis_offset_x[neg_sorted] = boundary_x + out_cum_x
            tilt_series.tilt_axis_offset_y[neg_sorted] = boundary_y + out_cum_y

        # Positive side: boundary (last reliable tilt, sorted idx
        # pos_boundary_sorted_idx); tilt i gets boundary + sum(rel[:i+1]).
        if len(pos_relative_offsets_x) > 0:
            boundary_sorted_idx = sorted_indices[pos_boundary_sorted_idx]
            boundary_x = tilt_series.tilt_axis_offset_x[boundary_sorted_idx]
            boundary_y = tilt_series.tilt_axis_offset_y[boundary_sorted_idx]
            in_cum_x = torch.cumsum(pos_relative_offsets_x, dim=0)
            in_cum_y = torch.cumsum(pos_relative_offsets_y, dim=0)
            tilt_series.tilt_axis_offset_x[pos_sorted] = boundary_x + in_cum_x
            tilt_series.tilt_axis_offset_y[pos_sorted] = boundary_y + in_cum_y

        # Expand reliable set by 1 tilt per side
        n_unreliable_per_side -= 1

    # Final optimization with all tilts reliable
    tilt_series, loss_values = optimize_fn(tilt_series)
    final_loss = loss_values[-1]
    if _DEBUG:
        print(f"Last one: loss went from {loss_values[0]} to {final_loss}")

    # Check for worse than best
    if final_loss >= best_loss:
        if _DEBUG:
            print(f"Final worse ({final_loss:.4f} >= {best_loss:.4f}), restoring best")
        tilt_series.tilt_axis_offset_x = best_offsets_x
        tilt_series.tilt_axis_offset_y = best_offsets_y
    else:
        all_loss_values.extend(loss_values)
        if _DEBUG:
            print(f"Final optimization achieved best loss: {final_loss:.4f}")

    return tilt_series, all_loss_values


def optimize_shifts_iterative(
    model: MissAlignment,
    tilt_series: TiltSeries,
    images: torch.Tensor,
    pixel_size: float,
    positions: torch.Tensor,
    patch_size: int = 96,
    batch_size: int = 16,
    apply_ctf: bool = True,
    device: str | torch.device = "cpu",
    initial_reliable_fraction: float = 1 / 2,
    position_subsample_fraction: float = 1.0,
    n_final_full_passes: int = 2,
    oversampling: float = 2.0,
    optimizer: str = "lbfgs",
    optimizer_kwargs: dict | None = None,
) -> tuple[TiltSeries, list[float]]:
    """Iterative anchored optimization for global shifts.

    Convenience wrapper around run_iterative_anchoring that uses
    optimize_shifts as the optimizer function.

    Parameters
    ----------
    model : MissAlignment
        Trained model for scoring reconstructions.
    tilt_series : TiltSeries
        Tilt series to optimize.
    images : torch.Tensor
        Preprocessed tilt images.
    pixel_size : float
        Pixel size in Angstroms.
    positions : torch.Tensor
        3D positions to reconstruct and evaluate.
    patch_size : int
        Size of reconstruction patches.
    batch_size : int
        Batch size for reconstruction.
    apply_ctf : bool
        Whether to apply CTF correction.
    device : str | torch.device
        Device to run optimization on.
    initial_reliable_fraction : float
        Initial fraction of tilts to consider reliable (0 to 1). Default 1/2.
    position_subsample_fraction : float, default 1.0
        Fraction of reconstruction positions evaluated in the early (cheap)
        anchoring passes; a fixed random subset is drawn once and reused so
        loss values stay comparable across passes. The last
        ``n_final_full_passes`` passes of the schedule and the final full
        optimization always use all positions. 1.0 disables subsampling.
    n_final_full_passes : int, default 2
        Number of passes at the end of the anchoring schedule that use the
        full position set (including the final all-tilts optimization).
    oversampling : float, default 2.0
        Reconstruction oversampling factor, passed through to
        ``optimize_shifts``.
    optimizer : str, default "lbfgs"
        Gradient optimizer for the alignment parameters ("lbfgs" or "adam"),
        passed through to ``optimize_shifts``.
    optimizer_kwargs : dict | None
        Extra optimizer settings, passed through to ``optimize_shifts``.

    Returns
    -------
    tuple[TiltSeries, list[float]]
        Optimized tilt series and loss values from all iterations.
    """
    if not 0.0 < position_subsample_fraction <= 1.0:
        raise ValueError(
            "position_subsample_fraction must be in (0, 1], got "
            f"{position_subsample_fraction}"
        )
    if n_final_full_passes < 0:
        raise ValueError("n_final_full_passes must be >= 0")

    # Keep model, tilt series, and images resident on device for the whole
    # anchoring run: every pass would otherwise pay device transfers both
    # ways. optimize_shifts skips its own transfers via device_override.
    device = torch.device(device)
    tilt_series.to(device)
    model.to(device)

    # Fixed random subset so early passes are comparable to each other (same
    # positions → comparable loss values). Drawn once, reused every pass.
    n_positions = positions.shape[0]
    if position_subsample_fraction < 1.0:
        n_subset = max(1, round(n_positions * position_subsample_fraction))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(torch.initial_seed())
        subset_indices = torch.randperm(n_positions, generator=generator)[:n_subset]
        subset_positions = positions[subset_indices.to(positions.device)]
    else:
        n_subset = n_positions
        subset_positions = positions

    # Mirror run_iterative_anchoring's loop counter to know when the
    # refinement tail begins: it runs one pass per decrement of
    # n_unreliable_per_side plus one final all-tilts pass. The last
    # n_final_full_passes of those use the full position set.
    n_tilts = tilt_series.n_tilts
    n_reliable = max(1, min(int(n_tilts * initial_reliable_fraction), n_tilts))
    remaining = (n_tilts - n_reliable) // 2 + 1  # +1 for the final pass

    def optimize_fn(ts: TiltSeries) -> tuple[TiltSeries, list[float]]:
        nonlocal remaining
        use_full = n_subset >= n_positions or remaining <= n_final_full_passes
        pass_positions = positions if use_full else subset_positions
        remaining -= 1

        return optimize_shifts(
            model=model,
            tilt_series=ts,
            images=images,
            pixel_size=pixel_size,
            positions=pass_positions,
            setting="global",
            patch_size=patch_size,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            device=device,
            oversampling=oversampling,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            device_override=device,
        )

    try:
        return run_iterative_anchoring(
            tilt_series=tilt_series,
            optimize_fn=optimize_fn,
            initial_reliable_fraction=initial_reliable_fraction,
        )
    finally:
        # Caller of evaluate_tilt_series expects CPU-resident metadata.
        tilt_series.to("cpu")
        model.to("cpu")
