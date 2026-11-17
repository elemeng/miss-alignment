from pathlib import Path

import mrcfile
import pytest
import torch
from warpylib import TiltSeries

from miss_alignment.prepare_stacks import (
    _stack_is_reusable,
    prepare_stacks_parallel,
)


def _make_xml_with_prepared_stack(
    tmp_path: Path,
    pixel_size: float,
    n_tilts: int = 3,
    with_thumbnails: bool = True,
) -> Path:
    """Create a training directory with one XML and a prepared stack."""
    xml_path = tmp_path / "series.xml"
    ts = TiltSeries(path=xml_path, n_tilts=n_tilts)
    ts.save_meta(xml_path)

    stack_dir = Path(ts.tilt_stack_dir)
    stack_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(ts.tilt_stack_path, overwrite=True) as mrc:
        mrc.set_data(torch.randn(n_tilts, 32, 32).numpy().astype("float32"))
        mrc.voxel_size = (pixel_size, pixel_size, pixel_size)
    Path(ts.angle_file_path).write_text("\n".join(f"{a:.2f}" for a in range(n_tilts)))

    thumbnail_dir = Path(ts.tilt_stack_thumbnail_dir)
    if with_thumbnails:
        thumbnail_dir.mkdir(parents=True, exist_ok=True)

    return xml_path


class TestStackIsReusable:
    def test_reusable_when_all_parts_exist_with_matching_pixel_size(self, tmp_path):
        xml_path = _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)
        assert _stack_is_reusable(xml_path, desired_pixel_size=5.0) is True

    def test_not_reusable_when_pixel_size_differs(self, tmp_path):
        xml_path = _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)
        assert _stack_is_reusable(xml_path, desired_pixel_size=4.5) is False

    def test_not_reusable_when_stack_missing(self, tmp_path):
        xml_path = _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)
        ts = TiltSeries(str(xml_path))
        Path(ts.tilt_stack_path).unlink()
        assert _stack_is_reusable(xml_path, desired_pixel_size=5.0) is False

    def test_not_reusable_when_angle_file_missing(self, tmp_path):
        xml_path = _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)
        ts = TiltSeries(str(xml_path))
        Path(ts.angle_file_path).unlink()
        assert _stack_is_reusable(xml_path, desired_pixel_size=5.0) is False

    def test_not_reusable_when_thumbnail_dir_missing(self, tmp_path):
        xml_path = _make_xml_with_prepared_stack(
            tmp_path, pixel_size=5.0, with_thumbnails=False
        )
        assert _stack_is_reusable(xml_path, desired_pixel_size=5.0) is False

    def test_not_reusable_when_no_xml(self, tmp_path):
        assert _stack_is_reusable(tmp_path / "missing.xml", 5.0) is False


class TestPrepareStacksParallelReuse:
    def test_all_reusable_skips_distribution(self, tmp_path, monkeypatch):
        _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)

        calls = []
        monkeypatch.setattr(
            "miss_alignment.prepare_stacks.run_distributed",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        prepare_stacks_parallel(
            training_directory=tmp_path,
            desired_pixel_size=5.0,
            reuse_prepared_stacks=True,
        )

        # Nothing was enqueued
        assert calls == []

    def test_not_reusable_enqueues_task(self, tmp_path, monkeypatch):
        _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)

        calls = []
        monkeypatch.setattr(
            "miss_alignment.prepare_stacks.run_distributed",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        # Different pixel size → stack must be re-prepared
        prepare_stacks_parallel(
            training_directory=tmp_path,
            desired_pixel_size=4.0,
            reuse_prepared_stacks=True,
        )

        assert len(calls) == 1
        assert calls[0]["tilt_series_list"] == [tmp_path / "series.xml"]
        assert calls[0]["desired_pixel_size"] == 4.0

    def test_reuse_disabled_always_enqueues(self, tmp_path, monkeypatch):
        _make_xml_with_prepared_stack(tmp_path, pixel_size=5.0)

        calls = []
        monkeypatch.setattr(
            "miss_alignment.prepare_stacks.run_distributed",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        prepare_stacks_parallel(
            training_directory=tmp_path,
            desired_pixel_size=5.0,
            reuse_prepared_stacks=False,
        )

        assert len(calls) == 1
        assert calls[0]["tilt_series_list"] == [tmp_path / "series.xml"]

    def test_no_xml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            prepare_stacks_parallel(
                training_directory=tmp_path,
                desired_pixel_size=5.0,
            )
