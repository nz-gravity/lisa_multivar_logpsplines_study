"""Paired stationary LISA instrumental-noise synthesis with exact fixed TDI.

Uses official PyTDI X2/Y2/Z2 delay polynomials and static lisaorbits geometry.
This is continuous-PSD, band-limited Gaussian Fourier synthesis of independent
MOSA noises, not an end-to-end LISAInstrument digital-filter simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np
from lisaconstants import c
from lisaconstants.indexing import LINKS
from lisaorbits.orbits import StaticConstellation
from pytdi.michelson import X2_ETA, Y2_ETA, Z2_ETA

LINKS = tuple(int(link) for link in LINKS)
TAU = 8.3
AET = np.array(
    [[2, -1, -1], [0, -np.sqrt(3), np.sqrt(3)], [np.sqrt(2), np.sqrt(2), np.sqrt(2)]]
) / np.sqrt(6)
BASE_TM = np.full(6, 3e-15)
BASE_OMS = np.full(6, 15e-12)
CASES = {
    "symmetric": (BASE_TM, BASE_OMS),
    "asymmetric": (
        BASE_TM * np.array([0.70, 1.30, 0.85, 1.40, 0.90, 1.15]),
        BASE_OMS * np.array([0.80, 1.20, 0.65, 1.40, 1.10, 0.90]),
    ),
}


def create_fixed_orbits(path: Path, duration: float) -> None:
    orbit = StaticConstellation.from_armlengths(c * TAU, c * TAU, c * TAU)
    orbit.write(str(path), t0=-1000.0, dt=(duration + 2000.0) / 8, size=9)
    with h5py.File(path, "r+") as f:
        # Canonicalize sub-ULP geometric roundoff to the exact declared model.
        # The positions remain the unmodified official static constellation.
        for name in ("tcb/ltt", "tcb/ppr", "tps/ppr"):
            np.testing.assert_allclose(f[name][:], TAU, rtol=0, atol=1e-13)
            f[name][...] = TAU
        f.attrs["fixed_light_travel_time_s"] = TAU
        f.attrs["model"] = "StaticConstellation; exactly constant positions and delays"
        f.attrs["delay_roundoff_canonicalized"] = True


def read_fixed_delays(path: Path) -> tuple[dict[int, float], dict]:
    """Read the actual saved orbit file and reject evolving geometry."""
    with h5py.File(path, "r") as f:
        x, v, a, ltt = [f[name][:] for name in ("tcb/x", "tcb/v", "tcb/a", "tcb/ltt")]
        assert np.array_equal(x, np.broadcast_to(x[0], x.shape))
        assert np.array_equal(v, np.zeros_like(v))
        assert np.array_equal(a, np.zeros_like(a))
        assert np.array_equal(ltt, np.full(ltt.shape, TAU))
        for name in ("tcb/d_ltt", "tcb/d_ppr", "tps/d_ppr"):
            assert np.array_equal(f[name][:], np.zeros_like(f[name][:]))
        lengths = np.array(
            [
                np.linalg.norm(x[0, link // 10 - 1] - x[0, link % 10 - 1])
                for link in LINKS
            ]
        )
        np.testing.assert_allclose(lengths / c, ltt[0], rtol=0, atol=1e-13)
        return dict(zip(LINKS, ltt[0], strict=True)), {
            "fixed_positions": True,
            "zero_velocities": True,
            "zero_accelerations": True,
            "all_directed_delays_fixed": True,
            "orbit_samples": len(ltt),
            "light_travel_time_s": TAU,
            "arms_m": lengths.tolist(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def transfer_matrices(frequency: np.ndarray, delays: dict[int, float]) -> np.ndarray:
    """Transfer 6 velocity/c TM and 6 OMS noises into X2/Y2/Z2."""
    index = {link: i for i, link in enumerate(LINKS)}
    # Each eta_ij has local and delayed remote proof mass contributions.
    # m_ij=v_ij/c: eta_ij=m_ij+D_ij*m_ji+o_ij. No extra factor two.
    eta = np.zeros((len(frequency), 6, 12), complex)
    for link, row in index.items():
        reverse = 10 * (link % 10) + link // 10
        eta[:, row, row] = 1
        eta[:, row, index[reverse]] = np.exp(-2j * np.pi * frequency * delays[link])
        eta[:, row, row + 6] = 1
    xyz = np.zeros((len(frequency), 3, 6), complex)
    for row, combination in enumerate((X2_ETA, Y2_ETA, Z2_ETA)):
        for measurement, terms in combination.components.items():
            column = index[int(measurement.split("_")[-1])]
            for coefficient, delay_operators in terms:
                total = sum(
                    delays[int(label.split("_")[-1])] for label in delay_operators
                )
                xyz[:, row, column] += coefficient * np.exp(
                    -2j * np.pi * frequency * total
                )
    return xyz @ eta


def noise_psds(
    frequency: np.ndarray, tm_asds: np.ndarray, oms_asds: np.ndarray
) -> np.ndarray:
    """One-sided PSDs of independent fractional-frequency source processes."""
    if np.any(frequency <= 0):
        raise ValueError("Source PSDs are evaluated only at positive frequencies")
    f = frequency[:, None]
    tm = (
        (tm_asds[None, :] / (2 * np.pi * f * c)) ** 2
        * (1 + (0.4e-3 / f) ** 2)
        * (1 + (f / 8e-3) ** 4)
    )
    oms = (2 * np.pi * f * oms_asds[None, :] / c) ** 2 * (1 + (2e-3 / f) ** 4)
    return np.concatenate((tm, oms), axis=1)


def covariance(frequency, delays, tm_asds, oms_asds):
    h = transfer_matrices(np.asarray(frequency), delays)
    p = noise_psds(np.asarray(frequency), tm_asds, oms_asds)
    return (h * p[:, None, :]) @ h.conj().swapaxes(-1, -2)


def verify_equal_arm_limit(delays):
    frequency = np.geomspace(1e-4, 0.02, 200)
    p = noise_psds(frequency, *CASES["symmetric"])
    tm, oms = p[:, 0], p[:, 6]
    x = 2 * np.pi * frequency * TAU
    common = 16 * np.sin(x) ** 2 * np.sin(2 * x) ** 2
    diagonal = 4 * common * (oms + (3 + np.cos(2 * x)) * tm)
    cross = -2 * common * np.cos(x) * (oms + 4 * tm)
    expected = np.broadcast_to(cross[:, None, None], (len(x), 3, 3)).copy()
    expected[:, np.arange(3), np.arange(3)] = diagonal[:, None]
    actual = covariance(frequency, delays, *CASES["symmetric"])
    relative_error = np.max(np.abs(actual - expected) / diagonal[:, None, None])
    np.testing.assert_allclose(
        actual / diagonal[:, None, None],
        expected / diagonal[:, None, None],
        atol=2e-10,
        rtol=2e-10,
    )
    h_aet = AET @ transfer_matrices(frequency, delays)
    aett = (h_aet * p[:, None, :]) @ h_aet.conj().swapaxes(-1, -2)
    # Algebraically stable T expression avoids subtracting d+2o near zero.
    one_minus_cos = 2 * np.sin(x / 2) ** 2
    true_t = 4 * common * (one_minus_cos * oms + 2 * one_minus_cos**2 * tm)
    np.testing.assert_allclose(aett[:, 2, 2].real, true_t, rtol=2e-9, atol=0)
    return float(relative_error)


def generate(outdir: Path, days: float, dt: float, seed: int, chunk_size: int):
    if days <= 0 or dt <= 0 or chunk_size < 1:
        raise ValueError(
            "Require a positive duration, sampling interval and chunk size"
        )
    n = round(days * 86400 / dt)
    if n % 2:
        raise ValueError("Select an even sample count")
    if n * dt < 100:
        raise ValueError("Use at least 100 seconds for the 0.1–20 mHz validation band")
    outdir.mkdir(parents=True, exist_ok=True)
    duration, fs = n * dt, 1 / dt
    orbit_path = outdir / "fixed_orbits.h5"
    create_fixed_orbits(orbit_path, duration)
    delays, orbit_checks = read_fixed_delays(orbit_path)
    equal_error = verify_equal_arm_limit(delays)
    frequencies = np.fft.rfftfreq(n, dt)
    files, fourier, fourier_aet, whitened, count, normalized_diag, offdiag_coherence = (
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    )
    metadata = {
        "generator": "Stationary physical-MOSA Fourier synthesis through exact PyTDI X2_ETA/Y2_ETA/Z2_ETA",
        "not_end_to_end_instrument": True,
        "days": days,
        "dt": dt,
        "fs": fs,
        "size": n,
        "duration": duration,
        "seed": seed,
        "chunk_size": chunk_size,
        "tau": TAU,
        "links": LINKS,
        "noise_process_order": [f"tm_{m}" for m in LINKS] + [f"oms_{m}" for m in LINKS],
        "tdi_units": "fractional frequency",
        "psd_units": "1/Hz (one-sided)",
        "tm_asd_units": "m/s^2/sqrt(Hz)",
        "oms_asd_units": "m/sqrt(Hz)",
        "tm_fknee_hz": 0.4e-3,
        "tm_fbreak_hz": 8e-3,
        "oms_fknee_hz": 2e-3,
        "dc_and_nyquist": "zeroed; omit from analysis",
        "finite_record": "periodic stationary Gaussian Fourier ensemble, band limited at Nyquist",
        "same_standard_normal_fourier_draws": True,
        "aet_synthesis": "AET @ source transfer matrix before inverse FFT",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "orbit_file": orbit_path.name,
        "orbit_checks": orbit_checks,
        "equal_arm_closed_form_max_relative_error": equal_error,
        "AET_rows": AET.tolist(),
        "AET_convention": "A=(2X-Y-Z)/sqrt(6), E=(Z-Y)/sqrt(2), T=(X+Y+Z)/sqrt(3)",
        "sources": [
            "https://lisa-simulation.pages.in2p3.fr/instrument/noises.html",
            "https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html",
            "https://lisa-simulation.pages.in2p3.fr/pytdi/",
        ],
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pytdi", "lisaorbits", "lisaconstants", "h5py")
        },
    }
    for name, (tm, oms) in CASES.items():
        f = h5py.File(outdir / f"{name}.h5", "x")
        files[name] = f
        for key in ("dt", "fs", "size", "duration", "seed", "tau"):
            f.attrs[key] = metadata[key]
        f.attrs["description"] = metadata["generator"]
        f.attrs["tdi_units"] = metadata["tdi_units"]
        f.attrs["psd_units"] = metadata["psd_units"]
        f.attrs["links"] = LINKS
        f.attrs["tm_asds"] = tm
        f.attrs["oms_asds"] = oms
        f.attrs["orbit_sha256"] = orbit_checks["sha256"]
        f.create_dataset("t", data=np.arange(n) * dt, compression="gzip")
        f.create_dataset("truth_frequency", data=frequencies, compression="gzip")
        f.create_dataset(
            "truth_xyz",
            shape=(len(frequencies), 3, 3),
            dtype=np.complex128,
            chunks=(min(chunk_size, len(frequencies)), 3, 3),
            compression="gzip",
        )
        f.create_dataset(
            "truth_aet",
            shape=(len(frequencies), 3, 3),
            dtype=np.complex128,
            chunks=(min(chunk_size, len(frequencies)), 3, 3),
            compression="gzip",
        )
        fourier[name] = np.zeros((len(frequencies), 3), np.complex128)
        fourier_aet[name] = np.zeros_like(fourier[name])
        whitened[name] = np.zeros((3, 3), complex)
        count[name] = 0
        normalized_diag[name] = np.zeros(3)
        offdiag_coherence[name] = []
    rng = np.random.default_rng(seed)
    for first in range(1, len(frequencies) - 1, chunk_size):
        end = min(first + chunk_size, len(frequencies) - 1)
        freq = frequencies[first:end]
        h = transfer_matrices(freq, delays)
        ha = AET @ h
        z = (
            rng.normal(size=(len(freq), 12)) + 1j * rng.normal(size=(len(freq), 12))
        ) / np.sqrt(2)
        for name, (tm, oms) in CASES.items():
            p = noise_psds(freq, tm, oms)
            output = np.einsum("fij,fj->fi", h, np.sqrt(p) * z)
            s = (h * p[:, None, :]) @ h.conj().swapaxes(-1, -2)
            sa = (ha * p[:, None, :]) @ ha.conj().swapaxes(-1, -2)
            fourier[name][first:end] = np.sqrt(n * fs / 2) * output
            # Form quiet T before summing physical channels in the time domain.
            fourier_aet[name][first:end] = np.sqrt(n * fs / 2) * np.einsum(
                "fij,fj->fi", ha, np.sqrt(p) * z
            )
            files[name]["truth_xyz"][first:end] = s
            files[name]["truth_aet"][first:end] = sa
            band = (freq >= 1e-4) & (freq <= 0.02)
            if band.any():
                # Use Cholesky whitening as an independent multivariate draw check.
                w = np.linalg.solve(np.linalg.cholesky(s[band]), output[band, :, None])[
                    ..., 0
                ]
                whitened[name] += np.einsum("fi,fj->ij", w, w.conj())
                normalized_diag[name] += np.sum(
                    np.abs(output[band]) ** 2
                    / np.diagonal(s[band], axis1=1, axis2=2).real,
                    axis=0,
                )
                count[name] += band.sum()
                diag = np.diagonal(sa[band], axis1=1, axis2=2).real
                corr = sa[band] / np.sqrt(diag[:, :, None] * diag[:, None, :])
                offdiag_coherence[name].append(corr[:, [0, 0, 1], [1, 2, 2]])
    checks = {}
    for name in CASES:
        xyz = np.fft.irfft(fourier[name], n=n, axis=0)
        aet = np.fft.irfft(fourier_aet[name], n=n, axis=0)
        # AET is the same linear transform, evaluated without time-domain cancellation.
        np.testing.assert_allclose(
            aet, xyz @ AET.T, rtol=0, atol=2e-14 * np.max(np.abs(xyz))
        )
        assert np.isfinite(xyz).all()
        for label, array in zip(
            ("X2", "Y2", "Z2", "A2", "E2", "T2"),
            np.concatenate((xyz, aet), axis=1).T,
            strict=True,
        ):
            files[name].create_dataset(
                label, data=array, compression="gzip", shuffle=True
            )
        files[name].attrs["metadata_json"] = json.dumps(
            {
                **metadata,
                "case": name,
                "tm_asds": CASES[name][0].tolist(),
                "oms_asds": CASES[name][1].tolist(),
            }
        )
        files[name].close()
        white = whitened[name] / count[name]
        correlations = np.concatenate(offdiag_coherence[name])
        checks[name] = {
            "tm_asds": CASES[name][0].tolist(),
            "oms_asds": CASES[name][1].tolist(),
            "whitened_covariance_real": white.real.tolist(),
            "whitened_covariance_imag": white.imag.tolist(),
            "whitened_covariance_max_error": float(np.max(np.abs(white - np.eye(3)))),
            "power_to_truth_mean_by_xyz": (
                normalized_diag[name] / count[name]
            ).tolist(),
            "validation_frequency_band_hz": [1e-4, 0.02],
            "validation_frequency_bins": int(count[name]),
            "aet_pair_order": ["AE", "AT", "ET"],
            "median_aet_magnitude_squared_coherence": np.median(
                np.abs(correlations) ** 2, axis=0
            ).tolist(),
            "max_aet_imaginary_normalized_csd": np.max(
                np.abs(correlations.imag), axis=0
            ).tolist(),
        }
        checks[name]["whitening_check_tolerance"] = 5 / np.sqrt(count[name])
        assert (
            checks[name]["whitened_covariance_max_error"]
            < checks[name]["whitening_check_tolerance"]
        )
    assert max(checks["symmetric"]["median_aet_magnitude_squared_coherence"]) < 1e-20
    assert min(checks["asymmetric"]["median_aet_magnitude_squared_coherence"]) > 1e-5
    assert max(checks["asymmetric"]["max_aet_imaginary_normalized_csd"]) > 1e-4
    report = {**metadata, "checks": checks, "passed": True}
    (outdir / "generation_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    p.add_argument("--days", type=float, default=10)
    p.add_argument("--dt", type=float, default=2)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--chunk-size", type=int, default=8192)
    a = p.parse_args()
    generate(a.outdir, a.days, a.dt, a.seed, a.chunk_size)
