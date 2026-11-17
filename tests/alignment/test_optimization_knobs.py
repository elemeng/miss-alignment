"""Tests for alignment optimization performance knobs.

Covers the oversampling, optimizer (LBFGS/Adam), and position subsampling
options added for alignment speed control. These knobs change optimization
behavior by design, so the tests pin the plumbing: parameter pass-through,
subsample schedule (which passes see full vs subset positions), and
validation errors.
"""

import pytest
import torch
from unittest.mock import patch
from warpylib import TiltSeries

from miss_alignment.alignment.optimize_global import optimize_shifts
from miss_alignment.alignment.optimize_iterative import (
    optimize_shifts_iterative,
    run_iterative_anchoring,
)


class MockModel:
    def to(self, device):
        return self

    def freeze(self):
        pass

    def eval(self):
        pass

    def __call__(self, x):
        return (
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([0.0], requires_grad=True),
        )


def _make_tilt_series(n_tilts=3):
    ts = TiltSeries(n_tilts=n_tilts)
    ts.angles = torch.tensor([-30.0, 0.0, 30.0])
    ts.tilt_axis_offset_x = torch.zeros(n_tilts)
    ts.tilt_axis_offset_y = torch.zeros(n_tilts)
    return ts


class TestOptimizerOption:
    def test_unknown_optimizer_raises(self):
        ts = _make_tilt_series()
        with pytest.raises(ValueError, match="Unknown optimizer"):
            optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(3, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting="global",
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
                optimizer="rmsprop",
            )

    def test_adam_uses_n_steps(self):
        """Adam runs optimizer_kwargs['n_steps'] closure evaluations."""
        calls = [0]
        lrs = []
        original_adam_init = torch.optim.Adam.__init__

        def fake_step(optimizer_self, closure, *args, **kwargs):
            calls[0] += 1
            closure()

        def fake_init(optimizer_self, params, lr=1e-3, **kwargs):
            # Delegate to the real constructor so optimizer internals
            # (defaults, state) are set up; just capture the lr.
            lrs.append(lr)
            original_adam_init(optimizer_self, params, lr=lr, **kwargs)

        ts = _make_tilt_series()
        original_reconstruct = ts.reconstruct_subvolumes_single
        ts.reconstruct_subvolumes_single = lambda **kwargs: torch.randn(
            1, 32, 32, 32, requires_grad=True
        )

        try:
            with (
                patch("torch.optim.Adam.step", new=fake_step),
                patch("torch.optim.Adam.__init__", new=fake_init),
            ):
                _, losses = optimize_shifts(
                    model=MockModel(),
                    tilt_series=ts,
                    images=torch.randn(3, 128, 128),
                    pixel_size=10.0,
                    positions=torch.tensor([[100.0, 100.0, 100.0]]),
                    setting="global",
                    patch_size=32,
                    batch_size=1,
                    apply_ctf=False,
                    device="cpu",
                    optimizer="adam",
                    optimizer_kwargs={"n_steps": 3, "lr": 1e-2},
                )
        finally:
            ts.reconstruct_subvolumes_single = original_reconstruct

        assert calls[0] == 3
        assert len(losses) == 3
        # lr threaded through to the Adam constructor
        assert lrs == [1e-2]

    def test_lbfgs_default_single_step(self):
        """Default (LBFGS) path is unchanged: one .step call."""
        step_calls = [0]

        def mock_step(self, closure):
            step_calls[0] += 1
            # Return a clean loss so no retry loop is triggered
            closure()
            return None

        ts = _make_tilt_series()
        original_reconstruct = ts.reconstruct_subvolumes_single
        ts.reconstruct_subvolumes_single = lambda **kwargs: torch.randn(1, 32, 32, 32)

        try:
            with patch("torch.optim.LBFGS.step", mock_step):
                optimize_shifts(
                    model=MockModel(),
                    tilt_series=ts,
                    images=torch.randn(3, 128, 128),
                    pixel_size=10.0,
                    positions=torch.tensor([[100.0, 100.0, 100.0]]),
                    setting="global",
                    patch_size=32,
                    batch_size=1,
                    apply_ctf=False,
                    device="cpu",
                )
        finally:
            ts.reconstruct_subvolumes_single = original_reconstruct

        assert step_calls[0] == 1


class TestOversamplingOption:
    def test_oversampling_passed_to_reconstruction(self):
        """oversampling reaches reconstruct_subvolumes_single."""
        captured = {}

        ts = _make_tilt_series()

        def fake_reconstruct(**kwargs):
            captured.update(kwargs)
            return torch.randn(1, 32, 32, 32)

        original_reconstruct = ts.reconstruct_subvolumes_single
        ts.reconstruct_subvolumes_single = fake_reconstruct

        try:
            optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(3, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting="global",
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
                oversampling=1.0,
            )
        finally:
            ts.reconstruct_subvolumes_single = original_reconstruct

        assert captured["oversampling"] == 1.0

    def test_default_oversampling_is_two(self):
        captured = {}
        ts = _make_tilt_series()

        def fake_reconstruct(**kwargs):
            captured.update(kwargs)
            return torch.randn(1, 32, 32, 32)

        original_reconstruct = ts.reconstruct_subvolumes_single
        ts.reconstruct_subvolumes_single = fake_reconstruct

        try:
            optimize_shifts(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(3, 128, 128),
                pixel_size=10.0,
                positions=torch.tensor([[100.0, 100.0, 100.0]]),
                setting="global",
                patch_size=32,
                batch_size=1,
                apply_ctf=False,
                device="cpu",
            )
        finally:
            ts.reconstruct_subvolumes_single = original_reconstruct

        assert captured["oversampling"] == 2.0


class TestPositionSubsampling:
    def _run_anchoring(self, positions, subsample_fraction, n_final_full_passes=2):
        """Run optimize_shifts_iterative with mocked optimize_shifts, and
        record the positions each pass received."""
        received_positions = []
        call_counter = [0]

        def fake_optimize_shifts(*args, **kwargs):
            # optimize_shifts is called with tilt_series among the positional
            # args; positions is the 5th positional (model, ts, images,
            # pixel_size, positions) or passed by keyword.
            if len(args) >= 5:
                received_positions.append(args[4].clone())
            else:
                received_positions.append(kwargs["positions"].clone())
            call_counter[0] += 1
            # Improving losses so all iterations are kept
            return kwargs.get("tilt_series", args[1] if len(args) > 1 else None), [
                1.0 / call_counter[0]
            ]

        ts = TiltSeries(n_tilts=11)
        ts.angles = torch.linspace(-50, 50, 11)
        ts.tilt_axis_offset_x = torch.zeros(11)
        ts.tilt_axis_offset_y = torch.zeros(11)

        with patch(
            "miss_alignment.alignment.optimize_iterative.optimize_shifts",
            side_effect=fake_optimize_shifts,
        ):
            _, _ = optimize_shifts_iterative(
                model=MockModel(),
                tilt_series=ts,
                images=torch.randn(11, 128, 128),
                pixel_size=10.0,
                positions=positions,
                device="cpu",
                position_subsample_fraction=subsample_fraction,
                n_final_full_passes=n_final_full_passes,
            )
        return received_positions

    def test_full_fraction_uses_all_positions_every_pass(self):
        positions = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        received = self._run_anchoring(positions, 1.0)
        # 11 tilts, fraction 0.5 → n_reliable=5, per_side=3 → 3 passes + final
        assert len(received) == 4
        for pass_positions in received:
            assert pass_positions.shape[0] == 10

    def test_early_passes_use_subset_and_final_uses_full(self):
        positions = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        received = self._run_anchoring(positions, 0.5, n_final_full_passes=2)

        assert len(received) == 4
        # Early passes see ~half of positions
        assert received[0].shape[0] == 5
        assert received[1].shape[0] == 5
        # Refinement tail sees all positions
        assert received[2].shape[0] == 10
        assert received[3].shape[0] == 10

    def test_subset_is_identical_across_early_passes(self):
        positions = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        received = self._run_anchoring(positions, 0.5)

        torch.testing.assert_close(received[0], received[1])

    def test_invalid_fraction_raises(self):
        positions = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        with pytest.raises(ValueError, match="position_subsample_fraction"):
            optimize_shifts_iterative(
                model=MockModel(),
                tilt_series=TiltSeries(n_tilts=11),
                images=torch.randn(11, 128, 128),
                pixel_size=10.0,
                positions=positions,
                device="cpu",
                position_subsample_fraction=0.0,
            )

        with pytest.raises(ValueError, match="position_subsample_fraction"):
            optimize_shifts_iterative(
                model=MockModel(),
                tilt_series=TiltSeries(n_tilts=11),
                images=torch.randn(11, 128, 128),
                pixel_size=10.0,
                positions=positions,
                device="cpu",
                position_subsample_fraction=1.5,
            )

    def test_invalid_n_final_full_passes_raises(self):
        positions = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        with pytest.raises(ValueError, match="n_final_full_passes"):
            optimize_shifts_iterative(
                model=MockModel(),
                tilt_series=TiltSeries(n_tilts=11),
                images=torch.randn(11, 128, 128),
                pixel_size=10.0,
                positions=positions,
                device="cpu",
                n_final_full_passes=-1,
            )

    def test_run_iterative_anchoring_unchanged_by_knobs(self):
        """run_iterative_anchoring itself stays knob-free; the wrapper owns
        the schedule. Verify pass count is still 4 for 11 tilts @ 0.5."""
        ts = TiltSeries(n_tilts=11)
        ts.angles = torch.linspace(-50, 50, 11)
        ts.tilt_axis_offset_x = torch.zeros(11)
        ts.tilt_axis_offset_y = torch.zeros(11)

        call_count = [0]

        def mock_optimizer(tilt_series):
            call_count[0] += 1
            tilt_series.tilt_axis_offset_x = tilt_series.tilt_axis_offset_x + 1.0
            return tilt_series, [1.0 / call_count[0]]

        _, losses = run_iterative_anchoring(
            tilt_series=ts,
            optimize_fn=mock_optimizer,
            initial_reliable_fraction=0.5,
        )
        assert call_count[0] == 4
        assert len(losses) == 4
