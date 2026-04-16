#!/usr/bin/env python3
"""phys 6260 hw2 — m–sigma fit. pip install numpy emcee matplotlib. run: py -3 hw2_msigma.py"""

import csv
import math
import re
import urllib.request
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np

# paths (same folder as this script)
SCRIPT_FOLDER = Path(__file__).resolve().parent
PATH_RAW_ASCII = SCRIPT_FOLDER / "current_ascii.txt"
PATH_MSIGMA_CSV = SCRIPT_FOLDER / "msigma_clean.csv"
URL_BERKELEY_ASCII = "http://blackhole.berkeley.edu/wp-content/uploads/2016/09/current_ascii.txt"
PATH_FIG_CORNER_BASELINE = SCRIPT_FOLDER / "figure_msigma_corner_baseline.png"
PATH_FIG_DATAFIT_BASELINE = SCRIPT_FOLDER / "figure_msigma_datafit_baseline.png"
PATH_FIG_EPSILON_MARGIN = SCRIPT_FOLDER / "figure_msigma_epsilon_marginal.png"

SIGMA_REF_KMS = 200.0  # km/s
ALPHA_LIM, B_LIM = (5.0, 11.0), (0.0, 10.0)
EPSILON_PRIOR_LIM = (0.0, 1.0)  # dex
NWALKERS, NSTEPS, BURN_IN, RNG_SEED = 32, 4000, 1000, 42


def hpd(samples, mass=0.68):
    # shortest interval holding ~mass of sorted 1d samples (discrete hpd)
    xs = np.sort(np.asarray(samples).ravel())
    n = len(xs)
    k = max(1, int(round(mass * n)))
    if k >= n:
        return float(xs[0]), float(xs[-1])
    w = xs[k - 1 :] - xs[: n - k + 1]
    i = int(np.argmin(w))
    return float(xs[i]), float(xs[i + k - 1])


def A_msun(alpha, b, sigma_ref_kms=SIGMA_REF_KMS):
    # M = A * sigma^b, sigma in km/s; alpha = log10 M at sigma_ref
    return 10.0 ** (alpha - b * math.log10(sigma_ref_kms))


def ingest():
    # berkeley ascii -> msigma_clean.csv (this file format only)
    if not PATH_RAW_ASCII.is_file():
        print(f"Downloading {URL_BERKELEY_ASCII}")
        urllib.request.urlretrieve(URL_BERKELEY_ASCII, PATH_RAW_ASCII)

    text = PATH_RAW_ASCII.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        t = line.strip()
        if not t or t.startswith(("Col.", "This ASCII", "Please cite")):
            continue
        lines.append(line)

    n_bad, n_dup = 0, 0
    rows, seen = [], set()
    n_pre_dedupe = 0

    for line in lines:
        if not line.strip():
            continue
        tok = line.split()
        if len(tok) < 9:
            n_bad += 1
            continue
        try:
            name = tok[0]
            M = float(tok[2])  # msun
            Mlo, Mhi = float(tok[3]), float(tok[4])  # msun
            sig = float(tok[6])  # km/s
            slo, shi = float(tok[7]), float(tok[8])  # km/s
        except ValueError:
            n_bad += 1
            continue

        gid = re.sub(r"\s+", " ", re.sub(r"\^[a-zA-Z0-9]+", "", name.strip())).strip()
        if not gid:
            n_bad += 1
            continue

        Mel, Meh = M - Mlo, Mhi - M
        Sel, Seh = sig - slo, shi - sig
        if M <= 0 or sig <= 0 or min(Mel, Meh, Sel, Seh) < 0 or any(e != e for e in (Mel, Meh, Sel, Seh)):
            n_bad += 1
            continue

        n_pre_dedupe += 1
        if gid in seen:
            n_dup += 1
            continue
        seen.add(gid)

        rows.append(
            {
                "galaxy_name_raw": name.strip(),
                "galaxy_id": gid,
                "M_bh_Msun": M,
                "M_err_lo": Mel,
                "M_err_hi": Meh,
                "sigma_kms": sig,
                "sigma_err_lo": Sel,
                "sigma_err_hi": Seh,
                "log10_M_bh": math.log10(M),
                "log10_sigma": math.log10(sig),
            }
        )

    n_final = len(rows)
    print("Ingestion summary")
    print(f"  final N (galaxies in CSV): {n_final}")
    print(f"  duplicates removed: {n_dup}")
    print(f"  bad / skipped rows: {n_bad}")
    print(f"  (rows passing filters before dedupe: {n_pre_dedupe})")

    cols = [
        "galaxy_name_raw",
        "galaxy_id",
        "M_bh_Msun",
        "M_err_lo",
        "M_err_hi",
        "sigma_kms",
        "sigma_err_lo",
        "sigma_err_hi",
        "log10_M_bh",
        "log10_sigma",
    ]
    with PATH_MSIGMA_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {PATH_MSIGMA_CSV}")


def load_csv(path):
    # msigma_clean.csv -> x, y, dex errors, N galaxies
    M, mlo, mhi, sig, slo, shi, logchk = [], [], [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            M.append(float(row["M_bh_Msun"]))
            mlo.append(float(row["M_err_lo"]))
            mhi.append(float(row["M_err_hi"]))
            sig.append(float(row["sigma_kms"]))
            slo.append(float(row["sigma_err_lo"]))
            shi.append(float(row["sigma_err_hi"]))
            logchk.append(float(row["log10_M_bh"]))

    M = np.array(M)
    sig = np.array(sig)
    y = np.log10(M)
    if not np.allclose(y, logchk, atol=1e-9):
        raise ValueError("log10_M_bh mismatch")

    Mlo, Mhi = M - np.array(mlo), M + np.array(mhi)
    if np.any(Mlo <= 0):
        raise ValueError("mass lower bound")
    sigma_y_dex = (y - np.log10(Mlo) + (np.log10(Mhi) - y)) / 2.0  # dex

    slo_a, shi_a = sig - np.array(slo), sig + np.array(shi)
    if np.any(slo_a <= 0):
        raise ValueError("sigma lower bound")
    lx = np.log10(sig)
    sigma_x_dex = (lx - np.log10(slo_a) + (np.log10(shi_a) - lx)) / 2.0  # dex

    x = np.log10(sig / SIGMA_REF_KMS)
    return x, y, sigma_y_dex, sigma_x_dex, len(M)


def log_prob_baseline(theta, x, y, sigma_y_dex, sigma_x_dex):
    alpha, b = float(theta[0]), float(theta[1])
    if not (ALPHA_LIM[0] <= alpha <= ALPHA_LIM[1] and B_LIM[0] <= b <= B_LIM[1]):
        return -np.inf
    st = np.sqrt(sigma_y_dex ** 2 + (b * sigma_x_dex) ** 2)
    if np.any(st <= 0) or not np.all(np.isfinite(st)):
        return -np.inf
    r = y - alpha - b * x
    return -0.5 * np.sum((r / st) ** 2) - np.sum(np.log(st))


def log_prob_scatter(theta, x, y, sigma_y_dex, sigma_x_dex):
    alpha, b, eps = float(theta[0]), float(theta[1]), float(theta[2])
    if not (EPSILON_PRIOR_LIM[0] <= eps <= EPSILON_PRIOR_LIM[1]):
        return -np.inf
    if not (ALPHA_LIM[0] <= alpha <= ALPHA_LIM[1] and B_LIM[0] <= b <= B_LIM[1]):
        return -np.inf
    sm = np.sqrt(sigma_y_dex ** 2 + (b * sigma_x_dex) ** 2)
    st = np.sqrt(sm ** 2 + eps ** 2)
    if np.any(st <= 0) or not np.all(np.isfinite(st)):
        return -np.inf
    r = y - alpha - b * x
    return -0.5 * np.sum((r / st) ** 2) - np.sum(np.log(st))


def run_mcmc(extended, x, y, sy, sx):
    # one emcee run + figures. extended=False: baseline + corner + data plot; else scatter + eps plot
    rng = np.random.default_rng(RNG_SEED)
    ndim = 3 if extended else 2
    logp = log_prob_scatter if extended else log_prob_baseline

    if extended:
        print("Scatter MCMC (+ epsilon^2 in sigma_tot^2)")
    else:
        print("Baseline MCMC (y = alpha + b x; sigma_tot^2 = sigma_y^2 + b^2 sigma_x^2)")
    print(f"  N={len(x)}, walkers={NWALKERS}, steps={NSTEPS}, burn={BURN_IN}")

    b_ols, a_ols = np.polyfit(x, y, 1)
    if ndim == 2:
        p0 = np.array(
            [
                np.clip(a_ols, ALPHA_LIM[0] + 0.01, ALPHA_LIM[1] - 0.01),
                np.clip(b_ols, B_LIM[0] + 0.01, B_LIM[1] - 0.01),
            ]
        )
    else:
        e0 = float(np.clip(0.05, EPSILON_PRIOR_LIM[0] + 1e-6, EPSILON_PRIOR_LIM[1] - 1e-6))
        p0 = np.array(
            [
                np.clip(a_ols, ALPHA_LIM[0] + 0.01, ALPHA_LIM[1] - 0.01),
                np.clip(b_ols, B_LIM[0] + 0.01, B_LIM[1] - 0.01),
                e0,
            ]
        )

    pos = p0 + 1e-3 * rng.standard_normal((NWALKERS, ndim))
    pos[:, 0] = np.clip(pos[:, 0], ALPHA_LIM[0] + 1e-6, ALPHA_LIM[1] - 1e-6)
    pos[:, 1] = np.clip(pos[:, 1], B_LIM[0] + 1e-6, B_LIM[1] - 1e-6)
    if ndim == 3:
        pos[:, 2] = np.clip(pos[:, 2], EPSILON_PRIOR_LIM[0] + 1e-8, EPSILON_PRIOR_LIM[1] - 1e-8)

    def lnprob(t):
        return logp(t, x, y, sy, sx)

    sampler = emcee.EnsembleSampler(NWALKERS, ndim, lnprob)
    sampler.run_mcmc(pos, NSTEPS, progress=False)

    print(f"  mean acceptance fraction: {float(np.mean(sampler.acceptance_fraction)):.4f}")
    try:
        print(f"  autocorr time (per dim): {sampler.get_autocorr_time(tol=0)}")
    except Exception as e:
        print(f"  autocorr: not estimated ({e})")

    flat = sampler.get_chain(discard=BURN_IN, flat=True)

    if not extended:
        a_med = float(np.median(flat[:, 0]))
        b_med = float(np.median(flat[:, 1]))
        A_s = A_msun(flat[:, 0], flat[:, 1])

        fig, axes = plt.subplots(2, 2, figsize=(7, 7))
        axes[1, 0].scatter(A_s, flat[:, 1], s=1, alpha=0.15)
        axes[1, 0].set_xlabel(r"$A$ (M$_\odot$)")
        axes[1, 0].set_ylabel(r"$b$")
        axes[0, 0].hist(A_s, bins=40, density=True, color="C0", alpha=0.85)
        axes[0, 0].set_ylabel("density")
        axes[0, 0].set_title(r"$A$ (M$_\odot$)")
        axes[0, 1].axis("off")
        axes[1, 1].hist(flat[:, 1], bins=40, density=True, color="C1", alpha=0.85, orientation="horizontal")
        axes[1, 1].set_xlabel("density")
        axes[1, 1].set_title(r"$b$")
        fig.suptitle("Baseline posterior: A and b")
        fig.tight_layout()
        fig.savefig(PATH_FIG_CORNER_BASELINE, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(x, y, yerr=sy, fmt="o", ms=3, alpha=0.6, capsize=0, elinewidth=0.8)
        xg = np.linspace(x.min(), x.max(), 200)
        ax.plot(xg, a_med + b_med * xg, "r-", lw=2, label="posterior median fit")
        ax.set_xlabel(r"$x=\log_{10}(\sigma/200\,\mathrm{km\,s^{-1}})$")
        ax.set_ylabel(r"$y=\log_{10}(M_\mathrm{bh}/M_\odot)$")
        ax.legend()
        ax.set_title("Baseline (log space)")
        fig.tight_layout()
        fig.savefig(PATH_FIG_DATAFIT_BASELINE, dpi=150)
        plt.close(fig)

        print(f"  wrote {PATH_FIG_CORNER_BASELINE}, {PATH_FIG_DATAFIT_BASELINE}")
    else:
        eps = flat[:, 2]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(eps, bins=50, density=True, color="C2", alpha=0.85)
        ax.axvline(0.0, color="k", ls="--", lw=1)
        ann = ["epsilon=0 (dashed)"]
        for mass, c in zip((0.68, 0.95), ("C0", "C3")):
            lo, hi = hpd(eps, mass=mass)
            ax.axvline(lo, color=c, ls="-", lw=1.5, alpha=0.8)
            ax.axvline(hi, color=c, ls="-", lw=1.5, alpha=0.8)
            ann.append(f"{mass:.0%} HPD: [{lo:.4f}, {hi:.4f}]")
        ax.set_xlabel(r"$\varepsilon$ (dex)")
        ax.set_ylabel("density")
        ax.set_title(r"Intrinsic scatter $\varepsilon$ (dex in log space)")
        ax.text(0.02, 0.98, "\n".join(ann), transform=ax.transAxes, va="top", fontsize=8, family="monospace")
        fig.tight_layout()
        fig.savefig(PATH_FIG_EPSILON_MARGIN, dpi=150)
        plt.close(fig)

        print(f"  wrote {PATH_FIG_EPSILON_MARGIN}")

    return flat


def dump_terminal(flat_b, flat_s, x, y, sy, sx):
    def pl(name, s, mass):
        lo, hi = hpd(s, mass=mass)
        print(f"  {mass:.0%} HPD  {name}: [{lo:.6g}, {hi:.6g}]   median {float(np.median(s)):.6g}")

    b_b, b_s = flat_b[:, 1], flat_s[:, 1]
    A_b, A_s = A_msun(flat_b[:, 0], b_b), A_msun(flat_s[:, 0], b_s)
    eps_s = flat_s[:, 2]

    print("\nBaseline posterior (HPD)")
    for mass in (0.68, 0.95):
        pl("b", b_b, mass)
        pl("A (Msun)", A_b, mass)
        print()

    print("Scatter posterior (HPD)")
    for mass in (0.68, 0.95):
        pl("b", b_s, mass)
        pl("A (Msun)", A_s, mass)
        pl("epsilon (dex)", eps_s, mass)
        print()

    e_lo, e_hi = hpd(eps_s, mass=0.95)
    print("Epsilon vs zero (95% HPD on epsilon):")
    if e_lo <= 1e-3:
        print(
            "  Epsilon is consistent with 0 dex (95% HPD lower limit at or very near zero). "
            "Intrinsic scatter beyond measurement noise is not required at this level."
        )
    else:
        print(
            f"  95% HPD is [{e_lo:.4f}, {e_hi:.4f}] dex with lower limit above zero; "
            "favors extra intrinsic scatter in log10(M) on top of measurement errors."
        )
    if e_hi >= EPSILON_PRIOR_LIM[1] - 1e-4:
        print("  (Note: upper end of that HPD sits near the prior upper bound; interpret cautiously.)")

    print("\nPosterior medians")
    print(f"  baseline: median b = {float(np.median(b_b)):.6g}, median A = {float(np.median(A_b)):.6g} Msun")
    print(
        f"  scatter:  median b = {float(np.median(b_s)):.6g}, median A = {float(np.median(A_s)):.6g} Msun, "
        f"median epsilon = {float(np.median(eps_s)):.6g} dex"
    )

    bmed = float(np.median(b_s))
    sig_meas = np.sqrt(sy ** 2 + (bmed * sx) ** 2)
    print("\nTypical noise vs intrinsic scatter (scatter model)")
    print(
        f"  median sigma_meas over galaxies (dex, at median b) = {float(np.median(sig_meas)):.6g}; "
        f"median epsilon = {float(np.median(eps_s)):.6g} dex"
    )

    ab, bb = flat_b[:, 0:1], flat_b[:, 1:2]
    stb = np.sqrt(sy ** 2 + (bb * sx) ** 2)
    ll_b = -0.5 * np.sum(((y - ab - bb * x) / stb) ** 2, axis=1) - np.sum(np.log(stb), axis=1)

    aa, ba, ea = flat_s[:, 0:1], flat_s[:, 1:2], flat_s[:, 2:3]
    sm = np.sqrt(sy ** 2 + (ba * sx) ** 2)
    sts = np.sqrt(sm ** 2 + ea ** 2)
    ll_s = -0.5 * np.sum(((y - aa - ba * x) / sts) ** 2, axis=1) - np.sum(np.log(sts), axis=1)

    best_b, best_s = float(np.max(ll_b)), float(np.max(ll_s))
    print("\nBest log-likelihood (max over posterior samples)")
    print(f"  baseline: {best_b:.4f}")
    print(f"  scatter:  {best_s:.4f}")
    print(f"  delta (scatter - baseline): {best_s - best_b:.4f}")

    print(f"\nMedian b: baseline {float(np.median(b_b)):.6g} vs scatter {float(np.median(b_s)):.6g}")


def main():
    ingest()
    if not PATH_MSIGMA_CSV.is_file():
        print(f"Missing {PATH_MSIGMA_CSV}")
        return 1

    x, y, sy, sx, n_gal = load_csv(PATH_MSIGMA_CSV)
    print(f"Loaded N={n_gal} from {PATH_MSIGMA_CSV}")

    fb = run_mcmc(False, x, y, sy, sx)
    fs = run_mcmc(True, x, y, sy, sx)
    dump_terminal(fb, fs, x, y, sy, sx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
