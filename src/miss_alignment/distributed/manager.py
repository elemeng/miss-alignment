"""Head-node coordinator: writes tasks, starts provisioner, blocks until done."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import tqdm

from .config import load_cluster_config
from .provisioner import (
    ClusterProvisioner,
    CompositeProvisioner,
    LocalProvisioner,
    WorkerProvisioner,
)
from .queue import (
    QueueLayout,
    TaskSpec,
    _orphan_shard,
    clear_queue,
    clear_stop,
    compute_fingerprint,
    write_pending,
    write_stop,
)

_POLL_INTERVAL_S = 0.5
_SCHEDULER_INTERVAL_S = 10.0
_MANAGER_HB_INTERVAL_S = 5.0
_WORKER_STALL_TIMEOUT_S = 120.0


def _format_task_id(index: int, tilt_series_path: Path) -> str:
    return f"{index:07d}-{tilt_series_path.stem}"


def _write_manager_hb(layout: QueueLayout, seq: int) -> None:
    new_hb = layout.manager_hb / f"hb-{seq}"
    new_hb.write_text("")
    if seq > 0:
        (layout.manager_hb / f"hb-{seq - 1}").unlink(missing_ok=True)


def _sweep_stalled_workers(layout: QueueLayout) -> None:
    for worker_dir in layout.running.iterdir():
        if not worker_dir.is_dir():
            continue
        ticks = list(worker_dir.glob("hb-*"))
        latest_mtime = None
        for tick in ticks:
            try:
                mtime = tick.stat().st_mtime
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
            except FileNotFoundError:
                pass  # heartbeat thread replaced this tick; ignore
        if latest_mtime is not None:
            age = time.time() - latest_mtime
        else:
            age = time.time() - worker_dir.stat().st_mtime

        if age <= _WORKER_STALL_TIMEOUT_S:
            continue

        for task_file in worker_dir.glob("*.json"):
            # Skip tasks that already completed while the worker was being swept.
            if (layout.done / task_file.name).exists():
                task_file.unlink(missing_ok=True)
                continue
            if (layout.failed / task_file.name).exists():
                task_file.unlink(missing_ok=True)
                continue
            # Route back into the shard this task id maps to
            dest = layout.pending / f"shard_{_orphan_shard(task_file):02d}"
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
            pass


def _format_worker_counts(counts: dict[str, int]) -> str:
    return " ".join(f"{label}={n}" for label, n in sorted(counts.items()))


def _scheduler_thread(
    layout: QueueLayout,
    provisioner: WorkerProvisioner,
    n_workers: int,
    stop_event: threading.Event,
    error_box: list,
    worker_counts: dict,
) -> None:
    hb_seq = 1  # seq 0 written before thread starts
    last_hb = time.time()
    # Start from now so the first sweep is delayed by _SCHEDULER_INTERVAL_S;
    # run_distributed calls ensure_workers() explicitly at startup instead.
    last_sweep = time.time()

    try:
        while not stop_event.is_set():
            now = time.time()

            if now - last_hb >= _MANAGER_HB_INTERVAL_S:
                _write_manager_hb(layout, hb_seq)
                hb_seq += 1
                last_hb = now

            if now - last_sweep >= _SCHEDULER_INTERVAL_S:
                _sweep_stalled_workers(layout)
                # Count across shards (flat files from older layouts included)
                n_pending = sum(
                    1
                    for pat in ("*.json", "shard_*/*.json")
                    for _ in layout.pending.glob(pat)
                )
                if n_pending > 0:
                    # Cap at n_pending: no point submitting more workers than tasks.
                    # This also prevents resubmission when workers exit cleanly on
                    # an empty queue — only re-fill if the sweep re-pended orphans.
                    provisioner.ensure_workers(min(n_workers, n_pending))
                worker_counts.clear()
                worker_counts.update(provisioner.worker_counts_by_type())
                last_sweep = now

            stop_event.wait(timeout=1.0)
    except Exception as exc:
        error_box.append(exc)
        stop_event.set()  # wake the poll loop so it notices immediately


class PersistentSession:
    """Long-lived worker pool shared by the alignment phases of one run.

    A normal ``run_distributed`` call tears down its workers and queue when the
    phase ends. Across a multi-iteration training run that means every
    macro-iteration pays a cold start: fresh processes, torch/warpylib
    imports, checkpoint loads, and new CUDA contexts.

    A session keeps the queue directory, the provisioner, and the workers
    alive between phases. Workers run with ``persistent=True``, so they idle
    (cheaply) when the queue drains and pick up the next phase's tasks. The
    owning process closes the session once at the end of the run, which writes
    the STOP marker and shuts the pool down.

    Usage::

        with PersistentSession(queue_root, devices, n_cluster_workers) as session:
            for it in iterations:
                run_distributed(..., session=session)
    """

    def __init__(
        self,
        queue_root: Path,
        devices: list[int],
        n_cluster_workers: int | None = None,
    ) -> None:
        self.layout = QueueLayout(queue_root)
        self.layout.ensure_directories()
        self.devices = devices
        self.n_cluster_workers = n_cluster_workers

        self._stop_event = threading.Event()
        self._scheduler_errors: list = []
        self._worker_counts: dict = {}
        self._scheduler: threading.Thread | None = None
        self._provisioner: WorkerProvisioner | None = None
        self._closed = False

    @property
    def queue_root(self) -> Path:
        return self.layout.root

    def __enter__(self) -> PersistentSession:  # noqa: PYI034
        self._provisioner = _build_provisioner(
            self.queue_root,
            self.devices,
            self.n_cluster_workers,
            persistent=True,
        )
        n_workers = _worker_count(self.devices, self.n_cluster_workers)

        # Clear any stale STOP marker from a previous run in this directory.
        clear_stop(self.layout)
        _write_manager_hb(self.layout, seq=0)
        self._start_scheduler(n_workers)
        return self

    def _start_scheduler(self, n_workers: int) -> None:
        assert self._provisioner is not None
        self._scheduler = threading.Thread(
            target=_scheduler_thread,
            args=(
                self.layout,
                self._provisioner,
                n_workers,
                self._stop_event,
                self._scheduler_errors,
                self._worker_counts,
            ),
            daemon=True,
        )
        self._scheduler.start()
        self._provisioner.ensure_workers(n_workers)

    def ensure_running(self) -> None:
        """Validate the pool is alive; raise if the scheduler died."""
        if self._closed:
            raise RuntimeError("PersistentSession is already closed")
        if self._scheduler_errors:
            raise RuntimeError("Scheduler thread crashed") from self._scheduler_errors[
                0
            ]

    @property
    def worker_counts(self) -> dict:
        return self._worker_counts

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Tell persistent workers to exit, then stop the scheduler and pool.
        try:
            write_stop(self.layout)
        except OSError as exc:  # queue dir may already be gone
            print(f"[session] could not write stop marker: {exc}", file=sys.stderr)
        self._stop_event.set()
        if self._scheduler is not None:
            self._scheduler.join(timeout=5.0)
        if self._provisioner is not None:
            self._provisioner.shutdown()
        shutil.rmtree(self.layout.root, ignore_errors=True)


def _worker_count(devices: list[int], n_cluster_workers: int | None) -> int:
    if n_cluster_workers is not None:
        return n_cluster_workers
    return len(devices)


def _build_provisioner(
    queue_root: Path,
    devices: list[int],
    n_cluster_workers: int | None,
    persistent: bool = False,
) -> WorkerProvisioner:
    """Create the provisioner matching the requested execution mode."""
    if n_cluster_workers is not None:
        cluster_config = load_cluster_config()
        cluster = ClusterProvisioner(queue_dir=queue_root, config=cluster_config)
        if devices:
            # Also use local GPUs — the queue is shared so local and cluster
            # workers pull from the same task list with no extra coordination.
            local = LocalProvisioner(
                queue_dir=queue_root, devices=devices, persistent=persistent
            )
            return CompositeProvisioner([cluster, local])
        return cluster
    return LocalProvisioner(
        queue_dir=queue_root, devices=devices, persistent=persistent
    )


def run_distributed(
    tilt_series_list: list[Path],
    model_checkpoint: Path,
    output_directory: Path,
    setting: str | tuple,
    patch_size: int,
    patch_overlap: float,
    batch_size: int,
    apply_ctf: bool,
    downsample: int,
    devices: list[int],
    n_cluster_workers: int | None,
    queue_root: Path,
    task_type: str = "alignment",
    desired_pixel_size: float | None = None,
    lowpass_cutoff: float | None = None,
    oversampling: float | None = None,
    optimizer: str | None = None,
    optimizer_kwargs: dict | None = None,
    position_subsample_fraction: float | None = None,
    session: PersistentSession | None = None,
) -> dict[str, float]:
    """Write tasks, provision workers, block until all tasks are terminal.

    Returns dict[series_name → final_loss]. Raises RuntimeError listing all
    failed series if any task ends in failed/.

    Without a ``session`` the queue and workers are created and destroyed
    around this call. Pass a :class:`PersistentSession` to reuse one warm
    worker pool across several phases; in that case this function only adds
    this phase's tasks and waits for them, leaving the pool running.

    task_type selects the worker dispatch path:
      "alignment"          — evaluate_tilt_series (default)
      "prepare_stacks"     — _prepare_single_tilt_series
      "cross_correlation"  — _run_cross_correlation_single
    """
    if session is not None:
        # Shared pool: reuse its layout, only reset this phase's task state.
        layout = session.layout
        clear_queue(layout)
    else:
        layout = QueueLayout(queue_root)
        layout.ensure_directories()
        clear_queue(layout)

    # Fingerprint only meaningful for alignment tasks (amortizes model load).
    if task_type == "alignment":
        fingerprint = compute_fingerprint(
            model_checkpoint_path=str(model_checkpoint),
            setting=setting if isinstance(setting, str) else list(setting),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
            oversampling=oversampling,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            position_subsample_fraction=position_subsample_fraction,
        )
    else:
        fingerprint = ""

    task_ids = []
    for i, ts_path in enumerate(tilt_series_list):
        task_id = _format_task_id(i, ts_path)
        task_ids.append(task_id)
        spec = TaskSpec(
            task_id=task_id,
            model_checkpoint_path=(
                str(model_checkpoint) if task_type == "alignment" else ""
            ),
            tilt_series_path=str(ts_path),
            output_directory=str(output_directory),
            setting=setting if isinstance(setting, str) else list(setting),
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            batch_size=batch_size,
            apply_ctf=apply_ctf,
            downsample=downsample,
            init_fingerprint=fingerprint,
            task_type=task_type,
            desired_pixel_size=desired_pixel_size,
            lowpass_cutoff=lowpass_cutoff,
            oversampling=oversampling,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            position_subsample_fraction=position_subsample_fraction,
        )
        write_pending(layout, spec)

    if session is not None:
        session.ensure_running()
        provisioner = None
        scheduler = None
        stop_event = None
        scheduler_errors = session._scheduler_errors
        worker_counts = session.worker_counts
    else:
        # Write first heartbeat before starting workers so workers never see a
        # missing heartbeat on startup (would cause immediate exit).
        _write_manager_hb(layout, seq=0)

        provisioner = _build_provisioner(queue_root, devices, n_cluster_workers)
        n_workers = _worker_count(devices, n_cluster_workers)

        stop_event = threading.Event()
        scheduler_errors: list = []
        worker_counts: dict = {}
        scheduler = threading.Thread(
            target=_scheduler_thread,
            args=(
                layout,
                provisioner,
                n_workers,
                stop_event,
                scheduler_errors,
                worker_counts,
            ),
            daemon=True,
        )
        scheduler.start()
        provisioner.ensure_workers(n_workers)

    pending_ids = set(task_ids)
    losses: dict[str, float] = {}
    failed_series: list[str] = []

    _desc = {
        "alignment": "Tilt series alignment",
        "prepare_stacks": "Preparing stacks",
        "cross_correlation": "Cross-correlation alignment",
    }.get(task_type, task_type)
    pbar = tqdm.tqdm(total=len(task_ids), desc=_desc, file=sys.stdout)
    try:
        while pending_ids:
            time.sleep(_POLL_INTERVAL_S)

            if scheduler_errors:
                raise RuntimeError("Scheduler thread crashed") from scheduler_errors[0]

            pbar.set_postfix_str(_format_worker_counts(worker_counts))

            for done_file in layout.done.glob("*.json"):
                data = json.loads(done_file.read_text())
                tid = data["task_id"]
                if tid in pending_ids:
                    ts_name = Path(data["tilt_series_path"]).stem
                    losses[ts_name] = data.get("final_loss", float("nan"))
                    pending_ids.discard(tid)
                    pbar.update(1)

            for fail_file in layout.failed.glob("*.json"):
                data = json.loads(fail_file.read_text())
                tid = data["task_id"]
                if tid in pending_ids:
                    ts_name = Path(data["tilt_series_path"]).stem
                    failed_series.append(ts_name)
                    pending_ids.discard(tid)
                    pbar.update(1)
                    print(
                        f"[manager] FAILED {ts_name}: {data.get('error', '')}",
                        file=sys.stderr,
                    )
    finally:
        pbar.close()
        if session is None:
            # One-shot mode: tear the pool down with this phase.
            stop_event.set()
            scheduler.join(timeout=5.0)
            provisioner.shutdown()
            shutil.rmtree(queue_root, ignore_errors=True)
        # Shared pool: leave workers running; the session owns teardown.

    if failed_series:
        raise RuntimeError(
            f"Alignment failed for {len(failed_series)} tilt series: "
            + ", ".join(failed_series)
        )

    return losses
