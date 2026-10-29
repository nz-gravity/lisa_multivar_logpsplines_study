"""Check the saved fixed-orbit records, including FFT normalization and CSD sign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from generate_data import AET, CASES, read_fixed_delays


def validate(directory: Path) -> dict:
    _, orbit_checks = read_fixed_delays(directory / "fixed_orbits.h5")
    report = {"orbit_checks": orbit_checks, "cases": {}}
    for case in CASES:
        with h5py.File(directory / f"{case}.h5") as source:
            dt = float(source.attrs["dt"])
            n = int(source.attrs["size"])
            frequencies = np.fft.rfftfreq(n, dt)
            np.testing.assert_array_equal(frequencies, source["truth_frequency"][:])
            selected = (frequencies >= 1e-4) & (frequencies <= 0.02)
            count = int(selected.sum())
            tolerance = 5 / np.sqrt(count)
            summaries = {}
            xyz = np.column_stack([source[name][:] for name in ("X2", "Y2", "Z2")])
            aet = np.column_stack([source[name][:] for name in ("A2", "E2", "T2")])
            transform_error = float(
                np.max(np.abs(aet - xyz @ AET.T)) / np.max(np.abs(xyz))
            )
            assert transform_error < 2e-14
            for basis, raw in (("xyz", xyz), ("aet", aet)):
                # E[(sqrt(2 dt/N) D) (sqrt(2 dt/N) D)^H] = S, for interior bins.
                dft = np.fft.rfft(raw, axis=0) * np.sqrt(2 * dt / n)
                truth = source[f"truth_{basis}"][:][selected]
                np.testing.assert_allclose(
                    truth,
                    truth.conj().swapaxes(-1, -2),
                    rtol=0,
                    atol=1e-14 * np.max(np.abs(truth)),
                )
                white = np.linalg.solve(
                    np.linalg.cholesky(truth), dft[selected, :, None]
                )[..., 0]
                covariance = np.einsum("fi,fj->ij", white, white.conj()) / count
                error = float(np.max(np.abs(covariance - np.eye(3))))
                assert error < tolerance
                endpoint_ratio = float(
                    np.max(np.abs(dft[[0, -1]])) / np.max(np.abs(dft))
                )
                assert endpoint_ratio < 2e-14
                summaries[basis] = {
                    "whitened_covariance_real": covariance.real.tolist(),
                    "whitened_covariance_imag": covariance.imag.tolist(),
                    "whitened_covariance_max_error": error,
                    "dc_nyquist_amplitude_relative_to_max": endpoint_ratio,
                }
            report["cases"][case] = {
                "aet_transform_max_error_relative_to_xyz_max": transform_error,
                "frequency_band_hz": [1e-4, 0.02],
                "frequency_bins": count,
                "whitening_tolerance": tolerance,
                **summaries,
            }
    report["passed"] = True
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    args = parser.parse_args()
    result = validate(args.data)
    text = json.dumps(result, indent=2) + "\n"
    (args.data / "generation_validation.json").write_text(text)
    print(text, end="")
