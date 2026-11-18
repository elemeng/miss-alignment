"""Stack preparation utilities for tilt series.

This module provides functionality to load raw tilt images and create
preprocessed tilt stacks ready for training.
"""

from pathlib import Path

import mrcfile
from warpylib import TiltSeries
from warpylib.movie import Movie

from .distributed.manager import run_distributed


def _get_original_pixel_size(tilt_series: TiltSeries) -> float:
    """Get original pixel size from the first tilt image's MRC header.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series object containing movie paths and metadata.

    Returns
    -------
    float
        Original pixel size in Angstroms.

    Raises
    ------
    FileNotFoundError
        If the first tilt image is not found.
    ValueError
        If pixel size cannot be determined from the header.
    """
    if len(tilt_series.tilt_movie_paths) == 0:
        raise ValueError("Tilt series has no movie paths")

    first_tilt_path = tilt_series.tilt_movie_paths[0]
    full_path = str(
        Path(tilt_series.data_directory_name or tilt_series.processing_directory_name)
        / first_tilt_path
    )
    movie = Movie(path=full_path)
    average_path = movie.average_path

    if not Path(average_path).exists():
        raise FileNotFoundError(f"Average image not found: {average_path}")

    with mrcfile.open(average_path, mode="r", permissive=True) as mrc:
        voxel_size = mrc.voxel_size
        # voxel_size is a numpy recarray with x, y, z fields
        pixel_size = float(voxel_size.x)

    if pixel_size <= 0:
        raise ValueError(
            f"Invalid pixel size {pixel_size} in MRC header for {average_path}"
        )

    return pixel_size


def _stack_is_reusable(xml_path: Path, desired_pixel_size: float) -> bool:
    """Check whether an already-prepared tilt stack can be reused.

    A stack is reusable when the .st file, angle file, and thumbnail directory
    all exist and the voxel size stored in the .st header matches the desired
    pixel size. The pixel size is written by ``stack_tilts`` at preparation
    time, so a header match guarantees the stack was prepared at exactly the
    requested pixel size.

    Parameters
    ----------
    xml_path : Path
        Path to the tilt series XML metadata file.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stack.

    Returns
    -------
    bool
        True when the existing stack matches the requested pixel size.
    """
    import mrcfile

    ts = TiltSeries(str(xml_path))

    stack_path = Path(ts.tilt_stack_path)
    angle_file_path = Path(ts.angle_file_path)
    thumbnail_dir = Path(ts.tilt_stack_thumbnail_dir)

    if not stack_path.exists():
        return False
    if not angle_file_path.exists():
        return False
    if not thumbnail_dir.is_dir():
        return False

    try:
        with mrcfile.open(
            stack_path, mode="r", permissive=True, header_only=True
        ) as mrc:
            stored_pixel_size = float(mrc.voxel_size.x)
    except (ValueError, OSError):
        return False

    return stored_pixel_size == desired_pixel_size


def _prepare_single_tilt_series(
    xml_path: Path, desired_pixel_size: float, device: int | None
) -> None:
    """Load raw tilt images and create tilt stack for a single tilt series.

    Parameters
    ----------
    xml_path : Path
        Path to the tilt series XML metadata file.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stack.
    device : int | None
        CUDA device index to use for GPU operations. If None, uses CPU.

    Raises
    ------
    FileNotFoundError
        If required image files are not found.
    ValueError
        If pixel size cannot be determined or other validation fails.
    """
    import torch

    if device is not None and torch.cuda.is_available():
        torch.cuda.set_device(device)

    ts = TiltSeries(xml_path)
    original_pixel_size = _get_original_pixel_size(ts)

    images, _, _ = ts.load_images(
        original_pixel_size=original_pixel_size,
        desired_pixel_size=desired_pixel_size,
        use_denoised=False,
        load_averages=True,
        load_half_averages=False,
    )

    ts.stack_tilts(
        tilt_data=images,
        pixel_size=desired_pixel_size,
        create_thumbnails=True,
    )


def prepare_stacks_parallel(
    training_directory: Path,
    desired_pixel_size: float,
    devices: list[int] | None = None,
    n_cluster_workers: int | None = None,
    reuse_prepared_stacks: bool = False,
) -> None:
    """Prepare tilt stacks for all tilt series in the training directory.

    This function loads raw tilt images for each tilt series, rescales them
    to the desired pixel size, and creates aligned tilt stacks with thumbnails.
    With ``reuse_prepared_stacks``, series whose existing stack already matches
    the desired pixel size (checked via the .st MRC header) are skipped.

    Parameters
    ----------
    training_directory : Path
        Directory containing tilt series XML files.
    desired_pixel_size : float
        Desired pixel size in Angstroms for the output stacks.
    devices : list[int] | None
        CUDA device indices to distribute work across (one worker process per
        unique device). If None, a single default-device worker is used.
    n_cluster_workers : int | None
        Number of cluster jobs to submit. When set, activates cluster mode;
        requires MISS_CLUSTER_CONFIG and MISS_CLUSTER_SCRIPT to be set.
    reuse_prepared_stacks : bool, default False
        Skip tilt series whose prepared stack already matches the desired
        pixel size.

    Raises
    ------
    FileNotFoundError
        If no XML files are found in the training directory.
    RuntimeError
        If any tilt series fails to process.
    """
    xml_files = list(training_directory.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No XML files found in training directory: {training_directory}"
        )

    if reuse_prepared_stacks:
        reusable = [
            xml for xml in xml_files if _stack_is_reusable(xml, desired_pixel_size)
        ]
        reusable_set = set(reusable)
        xml_files = [xml for xml in xml_files if xml not in reusable_set]
        if reusable:
            print(
                f"Reusing {len(reusable)} already-prepared stack(s) "
                f"at {desired_pixel_size} Å"
            )

    if not xml_files:
        print(f"All stacks already prepared at {desired_pixel_size} Å; nothing to do")
        return

    print(
        f"Preparing stacks for {len(xml_files)} tilt series at {desired_pixel_size} Å"
    )

    queue_root = training_directory / "tasks"
    run_distributed(
        tilt_series_list=xml_files,
        model_checkpoint=Path(""),
        output_directory=training_directory,
        setting="",
        patch_size=0,
        patch_overlap=0.0,
        batch_size=0,
        apply_ctf=False,
        downsample=1,
        devices=devices or [],
        n_cluster_workers=n_cluster_workers,
        queue_root=queue_root,
        task_type="prepare_stacks",
        desired_pixel_size=desired_pixel_size,
    )

    print(f"Successfully prepared stacks for {len(xml_files)} tilt series")
