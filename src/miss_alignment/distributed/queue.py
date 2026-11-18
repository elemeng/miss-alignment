"""Filesystem queue: task JSON files + atomic-rename claim protocol.

Directory layout under <root>/:
    pending/                  one JSON per queued task
    running/<worker_id>/      claimed task + heartbeat ticks
    done/                     completed task JSONs (result fields appended)
    failed/                   failed task JSONs (error field appended)
    manager/                  manager heartbeat ticks
    cluster/                  rendered cluster submission scripts
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

# Number of subdirectories the pending queue is split into. Keeps a claim
# scan proportional to (queue / shards) instead of the whole queue;
# 16 is far more than the worker counts this tool targets.
_PENDING_SHARDS = 16


@dataclass
class QueueLayout:
    root: Path

    @property
    def pending(self) -> Path:
        return self.root / "pending"

    @property
    def running(self) -> Path:
        return self.root / "running"

    @property
    def done(self) -> Path:
        return self.root / "done"

    @property
    def failed(self) -> Path:
        return self.root / "failed"

    @property
    def manager_hb(self) -> Path:
        return self.root / "manager"

    @property
    def cluster(self) -> Path:
        return self.root / "cluster"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def stop_marker(self) -> Path:
        """Sentinel telling persistent workers to exit for good.

        Used only in persistent mode (workers outliving a single phase).
        Written by the owning process when the whole run finishes, so idle
        workers can distinguish "between phases" from "run over".
        """
        return self.root / "STOP"

    def ensure_directories(self) -> None:
        for d in (
            self.pending,
            self.running,
            self.done,
            self.failed,
            self.manager_hb,
            self.cluster,
            self.logs,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def worker_dir(self, worker_id: str) -> Path:
        return self.running / worker_id


@dataclass
class TaskSpec:
    task_id: str
    model_checkpoint_path: str
    tilt_series_path: str
    output_directory: str
    setting: str | list
    patch_size: int
    patch_overlap: float
    batch_size: int
    apply_ctf: bool
    downsample: int
    init_fingerprint: str
    # Task type and optional parameters for non-alignment tasks.
    task_type: str = "alignment"
    desired_pixel_size: float | None = None
    lowpass_cutoff: float | None = None
    # Alignment performance knobs (see evaluate_tilt_series).
    oversampling: float | None = None
    optimizer: str | None = None
    optimizer_kwargs: dict | None = None
    position_subsample_fraction: float | None = None


def _atomic_write(path: Path, data: dict) -> None:
    """Write JSON atomically via a temp file + rename."""
    tmp = path.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _read_spec(path: Path) -> TaskSpec:
    data = json.loads(path.read_text())
    return TaskSpec(
        **{k: v for k, v in data.items() if k in TaskSpec.__dataclass_fields__}
    )


def compute_fingerprint(
    model_checkpoint_path: str,
    setting: str | list,
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    oversampling: float | None = None,
    optimizer: str | None = None,
    optimizer_kwargs: dict | None = None,
    position_subsample_fraction: float | None = None,
) -> str:
    """SHA-256 over the fields that require reloading the model/settings.

    The alignment performance knobs are included even though they do not
    themselves require a model reload: the worker keys its cached state on
    this fingerprint alone, so omitting them would let two tasks with
    different reconstruction/optimization settings silently share cached
    state instead of reloading.
    """
    payload = json.dumps(
        {
            "model_checkpoint_path": model_checkpoint_path,
            "setting": setting,
            "patch_size": patch_size,
            "patch_overlap": patch_overlap,
            "batch_size": batch_size,
            "apply_ctf": apply_ctf,
            "downsample": downsample,
            "oversampling": oversampling,
            "optimizer": optimizer,
            "optimizer_kwargs": optimizer_kwargs,
            "position_subsample_fraction": position_subsample_fraction,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_pending(layout: QueueLayout, spec: TaskSpec) -> None:
    """Queue a task in the shard its id maps to."""
    _pending_shards(layout)  # ensure shards exist
    shard = layout.pending / f"shard_{_shard_for(spec):02d}"
    _atomic_write(shard / f"{spec.task_id}.json", asdict(spec))


def _pending_shards(layout: QueueLayout) -> list[Path]:
    """Return the shard directories under pending/, creating them on demand.

    Sharding keeps each claim proportional to a fraction of the queue rather
    than the whole queue: with N workers and M remaining tasks, unsharded
    claiming costs O(N*M) directory scans per phase, which becomes noticeable
    for datasets with many tilt series.
    """
    shards = []
    for ordinal in range(_PENDING_SHARDS):
        shard = layout.pending / f"shard_{ordinal:02d}"
        shard.mkdir(parents=True, exist_ok=True)
        shards.append(shard)
    return shards


def _orphan_shard(task_file: Path) -> int:
    """Shard for an orphaned task file, from its "<task_id>.json" name."""
    digest = hashlib.sha256(task_file.stem.encode()).digest()
    return digest[0] % _PENDING_SHARDS


def _shard_for(spec: TaskSpec) -> int:
    """Deterministically map a task id to a shard (also used for orphans)."""
    digest = hashlib.sha256(spec.task_id.encode()).digest()
    return digest[0] % _PENDING_SHARDS


def claim_one(layout: QueueLayout, worker_id: str) -> TaskSpec | None:
    """Attempt to claim a pending task via atomic rename.

    Scans one shard at a time and stops at the first successful claim, so the
    cost is bounded by the shard size rather than the whole queue. Shard
    order is randomised per claim to avoid all workers racing on shard 0.

    Returns the claimed TaskSpec, or None if the queue is empty.
    """
    worker_dir = layout.worker_dir(worker_id)
    worker_dir.mkdir(parents=True, exist_ok=True)

    shards = _pending_shards(layout)
    random.shuffle(shards)

    for shard in shards:
        candidates = list(shard.glob("*.json"))
        if not candidates:
            continue
        random.shuffle(candidates)

        for candidate in candidates:
            dest = worker_dir / candidate.name
            try:
                os.rename(candidate, dest)
                return _read_spec(dest)
            except FileNotFoundError:
                # another worker claimed it first
                continue

    return None


def mark_done(
    layout: QueueLayout,
    worker_id: str,
    spec: TaskSpec,
    final_loss: float,
    device: str,
) -> None:
    """Write result to done/, then remove from running/ (publish-before-delete)."""
    data = asdict(spec)
    data["final_loss"] = final_loss
    data["device"] = device
    _atomic_write(layout.done / f"{spec.task_id}.json", data)
    (layout.worker_dir(worker_id) / f"{spec.task_id}.json").unlink(missing_ok=True)


def mark_failed(
    layout: QueueLayout,
    worker_id: str,
    spec: TaskSpec,
    error: str,
) -> None:
    """Write error to failed/, then remove from running/ (publish-before-delete)."""
    data = asdict(spec)
    data["error"] = error
    data["worker_id"] = worker_id
    _atomic_write(layout.failed / f"{spec.task_id}.json", data)
    (layout.worker_dir(worker_id) / f"{spec.task_id}.json").unlink(missing_ok=True)


def clear_queue(layout: QueueLayout) -> None:
    """Delete stale queue state from a prior run; recover running orphans to pending.

    Order matters: wipe pending/done/failed first, THEN recover orphans into
    the now-empty pending/ so they are not immediately re-deleted.
    """
    # Wipe pending shards (and any stray flat files from before sharding).
    for pattern_root in (layout.pending, layout.done, layout.failed):
        for f in pattern_root.glob("*.json"):
            f.unlink(missing_ok=True)
        for shard in pattern_root.glob("shard_*"):
            if shard.is_dir():
                for f in shard.glob("*.json"):
                    f.unlink(missing_ok=True)

    _pending_shards(layout)  # ensure shards exist before recovering orphans

    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        for task_file in worker_dir.glob("*.json"):
            # Route the recovered orphan into its shard, matching write_pending
            shard_name = f"shard_{_orphan_shard(task_file):02d}"
            dest = layout.pending / shard_name / task_file.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(task_file, dest)
            except FileNotFoundError:
                pass
        for hb in worker_dir.glob("hb-*"):
            hb.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass  # not empty yet; scheduler will sweep it


def write_stop(layout: QueueLayout) -> None:
    """Signal persistent workers to exit once the queue is drained."""
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.stop_marker.write_text("stop\n")


def stop_requested(layout: QueueLayout) -> bool:
    return layout.stop_marker.exists()


def clear_stop(layout: QueueLayout) -> None:
    layout.stop_marker.unlink(missing_ok=True)
