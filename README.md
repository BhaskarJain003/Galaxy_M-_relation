# M–σ relation (PHYS 6260 HW2)

Small homework project for **computational physics**: fit the famous **M–σ** scaling law (supermassive black hole mass vs bulge stellar velocity dispersion) with MCMC.

If you landed here randomly: this is student code, not a polished package—but it runs end-to-end and the math matches what we did in class.

## What it does

- Pulls the McConnell & Ma compilation from Berkeley (or uses `current_ascii.txt` if you already have it).
- Cleans it into `msigma_clean.csv` (masses, sigmas, asymmetric errors, deduped galaxies).
- Fits **log₁₀ M = α + b log₁₀(σ / 200 km/s)** with measurement errors in both axes (baseline model).
- Runs a second model that adds **intrinsic scatter** ε (dex in log M) on top of the noise.
- Writes a few PNGs (corner-ish posteriors, data + median fit, ε marginal) and dumps HPD intervals + a quick read on whether ε is consistent with zero.

Stack: **NumPy**, **emcee**, **matplotlib**. See `requirements.txt`.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows — on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python hw2_msigma.py
```

On Windows you can also do `py -3 hw2_msigma.py` if that is how you usually run Python.

First run may download the ASCII catalog from Berkeley; after that you can work offline if `current_ascii.txt` is present.

## Outputs (same folder as the script)

| File | What |
|------|------|
| `msigma_clean.csv` | Parsed galaxy table used for the fit |
| `figure_msigma_corner_baseline.png` | Posterior samples for **A** (M⊙ at the reference σ) and **b** |
| `figure_msigma_datafit_baseline.png` | Data in log space + median line |
| `figure_msigma_epsilon_marginal.png` | ε posterior (scatter model) |

The terminal printout has 68%/95% HPDs, medians, and a short blurb comparing the two models.

## Citing the data

The underlying catalog is **McConnell & Ma 2013, ApJ, 764, 184**. If you use their numbers in a paper or report, cite that paper (the ASCII header says the same). This repo is just my fit on top of their public table.

## License

Code here is under **Apache License 2.0** — see `LICENSE`. Third-party data rights stay with the original authors.

## Disclaimer

Course assignment, provided as-is. No warranty; check the numbers yourself if something matters for research.
