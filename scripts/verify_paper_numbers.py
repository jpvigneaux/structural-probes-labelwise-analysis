#!/usr/bin/env python3
"""Recompute every number the paper reports, and diff it against the paper.

`experiments/paper_runs.yaml` lists the six probing runs behind the article
together with the values published for each. This script refits each regression
from the same per-relation ULAS files and corpus statistics the paper used, and
prints a line per claim with the published value, the recomputed value and how
that value rounds at the paper's precision.

The point is that a claim in the text and the code that produced it cannot drift
apart silently: if a probing run is redone, or a predictor is redefined, the
mismatch shows up here rather than in review.

A check passes when the recomputed value, rounded to the number of decimals the
paper prints, equals the printed value -- that is, when the code produces
something that would be typeset the same way. Comparing against a +/- tolerance
instead misfires exactly on the boundaries (a value of 0.7355 against a printed
0.736), which is where several of these land. Rounding is half-up, as done by
hand, not the half-to-even of Python's round().

`--slack` admits a further N units in the last printed digit, for checking a
re-run whose probes were trained with a different seed rather than the archived
one.

The regressions here are the ones implemented in scripts/regression/; this
script re-implements the *fitting* (a few lines of statsmodels) rather than
shelling out and parsing stdout, but reads its inputs exactly as those scripts
do, via the shared helpers in scripts/regression/_regdata.py.

Usage:
    python scripts/verify_paper_numbers.py
    python scripts/verify_paper_numbers.py --manifest experiments/paper_runs.yaml \\
        --only bertbase --verbose
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent / 'regression'))
import _manifest                                    # noqa: E402
from _regdata import load_regression_frame          # noqa: E402
from holdout_predictivity import heldout_frame, heldout_stats   # noqa: E402

BASE = ['head_sim_entropy', 'mean_log_length']
FULL = BASE + ['sd_log_length']


def fit(df, cols):
    X = sm.add_constant(df[cols].to_numpy())
    return sm.WLS(df['uuas'].to_numpy(), X, weights=df['total'].to_numpy()).fit()


def peak_ulas(results_dir):
    """Highest dev UUAS over all checkpoints, and the checkpoint attaining it."""
    best, best_ck = -1.0, None
    for p in sorted(Path(results_dir).glob('layer-*/dev.uuas')):
        try:
            v = float(p.read_text().strip())
        except ValueError:
            continue
        if v > best:
            best, best_ck = v, int(p.parent.name.split('-')[1])
    return best, best_ck


def reliability(df):
    """Share of the weighted between-relation variance of ULAS that is not
    binomial sampling noise; caps the attainable R^2. See ulas_reliability.py."""
    p = df['uuas'].to_numpy(float)
    n = df['total'].to_numpy(float)
    w = n / n.sum()
    mean = float(np.sum(w * p))
    var_obs = float(np.sum(w * (p - mean) ** 2))
    var_noise = float(np.sum(w * p * (1 - p) / np.maximum(n, 1.0)))
    return max(var_obs - var_noise, 0.0) / var_obs if var_obs > 0 else float('nan')


def round_half_up(x, decimals):
    """Round the way a person does, not the way IEEE-754 does.

    Python's round() breaks ties to even, so round(0.7355, 3) is 0.736 or 0.735
    depending on which side of the tie the binary representation actually falls
    on. Decimal with ROUND_HALF_UP reproduces what was typed into the table.
    """
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-decimals)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def decimals_of(x):
    """How many decimal places the manifest wrote a published value with."""
    s = repr(float(x))
    return len(s.split('.')[1].rstrip('0')) if '.' in s else 0


class Report:
    def __init__(self, slack=0):
        self.rows, self.slack = [], slack

    def check(self, model, claim, published, got):
        if published is None:
            return
        d = decimals_of(published)
        if got is None:
            status = 'FAIL'
        elif abs(round_half_up(got, d) - published) <= self.slack * 10 ** -d + 1e-12:
            status = 'ok'
        elif round_half_up(round_half_up(got, d + 1), d) == published:
            # The scripts in scripts/regression/ print four decimals; a value
            # taken from that printout and rounded again to the paper's three
            # can land one unit high. The code is right and the paper's last
            # digit is a double rounding -- a distinct condition from a
            # genuine mismatch, so it is named rather than lumped in with one.
            status = 'ROUNDED TWICE'
        else:
            status = 'FAIL'
        self.rows.append((model, claim, published, got, d, status))

    def print(self):
        w = max(len(c) for _, c, *_ in self.rows) + 1
        cur = None
        for model, claim, pub, got, d, status in self.rows:
            if model != cur:
                print(f'\n{model}\n{"-" * len(model)}')
                cur = model
            if got is None:
                print(f'  {claim:<{w}}  {pub:11.6f}     MISSING              FAIL')
                continue
            print(f'  {claim:<{w}}  {pub:11.6f}  {got:11.6f}  '
                  f'-> {round_half_up(got, d):<10.{d}f}  {status}')

        bad = [r for r in self.rows if r[5] == 'FAIL']
        twice = [r for r in self.rows if r[5] == 'ROUNDED TWICE']
        print(f'\n{"=" * 78}')
        print(f'{len(self.rows) - len(bad) - len(twice)} of {len(self.rows)} checks '
              f'reproduce the published value exactly as printed')
        if twice:
            print(f'\n{len(twice)} value(s) differ by one unit in the last printed '
                  f'digit, consistent with\nrounding the scripts\' 4-decimal output '
                  f'again to 3 decimals. The code is\ncorrect; the paper\'s digit '
                  f'is not what the exact value rounds to:')
            for model, claim, pub, got, d, _ in twice:
                print(f'  {model:<20s} {claim:<26s} paper {pub:.{d}f}  '
                      f'exact {got:.6f} -> {round_half_up(got, d):.{d}f}')
        for model, claim, pub, got, d, _ in bad:
            g = f'{round_half_up(got, d):.{d}f}' if got is not None else 'missing'
            print(f'  FAIL  {model:<20s} {claim:<26s} '
                  f'paper {pub:.{d}f}  code {g}')
        return len(bad)


def verify_run(key, run, roots, predictors, rep, verbose):
    label = run['label']
    results = run['results'].format(**roots)
    pub = run.get('published', {})
    ck = run['optimal_checkpoint']

    uuas_path = Path(results) / f'layer-{ck:02d}' / 'dev.uuas_by_relation'
    if not uuas_path.exists():
        rep.rows.append((label, 'per-relation ULAS file', 0.0, None, 1, 'FAIL'))
        return

    df = load_regression_frame(uuas_path, predictors['sim'], predictors['length'])

    # --- peak ULAS and the checkpoint attaining it --------------------------
    peak, peak_ck = peak_ulas(results)
    rep.check(label, 'peak dev ULAS', pub.get('peak_ulas'), peak)
    if peak_ck != ck:
        rep.rows.append((label, 'optimal checkpoint', float(ck),
                         float(peak_ck), 0, 'FAIL'))

    # --- the published three-predictor regression ---------------------------
    m = fit(df, FULL)
    rep.check(label, 'R^2 (3 predictors)', pub.get('r2'), m.rsquared)
    rep.check(label, 'adjusted R^2', pub.get('adj_r2'), m.rsquared_adj)
    rep.check(label, 'F statistic', pub.get('f_stat'), m.fvalue)
    rep.check(label, 'n relations', pub.get('n_relations'), float(len(df)))
    rep.check(label, 'intercept', pub.get('const'), m.params[0])
    for i, name in enumerate(FULL, start=1):
        rep.check(label, f'beta {name}', pub.get('coef', {}).get(name),
                  m.params[i])
    for i, name in enumerate(['const'] + FULL):
        rep.check(label, f'SE {name}', pub.get('se', {}).get(name), m.bse[i])

    # --- moment ladder (Table tab:moment-ladder) ----------------------------
    if 'adj_r2_ladder' in pub or 'p_sd' in pub:
        cols3 = FULL + ['skew_log_length']
        ladder = [fit(df, BASE), fit(df, FULL), fit(df, cols3)]
        for got, want, tag in zip(ladder, pub.get('adj_r2_ladder', [None] * 3),
                                  ['M1', 'M1+sd', 'M1+sd+skew']):
            rep.check(label, f'adj R^2 {tag}', want, got.rsquared_adj)
        rep.check(label, 'p(sd) in M1+sd', pub.get('p_sd'), ladder[1].pvalues[-1])
        rep.check(label, 'p(skew) in M3', pub.get('p_skew'), ladder[2].pvalues[-1])

    # --- mean(log n) vs log(mean n), two-predictor models -------------------
    if 'r2_meanlog' in pub:
        rep.check(label, 'R^2 mean(log n)', pub.get('r2_meanlog'),
                  fit(df, BASE).rsquared)
        alt = df.assign(mean_log_length=df['log_mean_length'])
        rep.check(label, 'R^2 log(mean n)', pub.get('r2_logmean'),
                  fit(alt, BASE).rsquared)

    # --- standard deviation vs variance -------------------------------------
    if 'r2_var' in pub:
        var = df.assign(sd_log_length=df['sd_log_length'] ** 2)
        rep.check(label, 'R^2 with variance', pub.get('r2_var'),
                  fit(var, FULL).rsquared)

    # --- held-out predictivity ------------------------------------------------
    if 'heldout' in pub:
        ho, dropped = heldout_frame(results, ck, predictors['sim'],
                                    predictors['length'])
        stats = heldout_stats(ho)
        want = pub['heldout']
        rep.check(label, 'R^2 dev, relations shared with test',
                  want.get('r2_dev_heldout'), stats['r2_dev'])
        rep.check(label, 'R^2 on held-out edges', want.get('r2_test'),
                  stats['r2_test'])
        rep.check(label, 'Q^2 leave-one-relation-out', want.get('q2_loo'),
                  stats['q2_loo'])
        # The two frames must differ only by the relations the test split lacks;
        # anything else means the joins have drifted apart.
        if len(ho) + len(dropped) != len(df):
            rep.rows.append((label, 'held-out join size', float(len(df)),
                             float(len(ho) + len(dropped)), 0, 'FAIL'))

    # --- reliability ceiling --------------------------------------------------
    rep.check(label, 'ULAS reliability', pub.get('reliability'), reliability(df))

    # --- median R^2 of the log-linear decay model ----------------------------
    if 'median_r2_loglinear' in pub:
        tbl = Path(run['r2_table'].format(**roots))
        got = None
        if tbl.exists():
            r2 = pd.read_csv(tbl)
            col = 'R2' if 'R2' in r2.columns else 'R^2'
            got = float(r2[col].median())
        rep.check(label, 'median R^2 log-linear', pub['median_r2_loglinear'],
                  got)

    if verbose:
        print(f'\n[{label}] checkpoint {ck}, {len(df)} relations')
        print(m.summary(xname=['const'] + FULL, yname='uuas'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--paths', default=None,
                    help='paths.yaml holding roots: and predictors:')
    ap.add_argument('--only', action='append', help='verify only these run keys')
    ap.add_argument('--slack', type=int, default=0,
                    help='admit N further units in the last printed digit')
    ap.add_argument('--verbose', action='store_true',
                    help='also print the full WLS summary for each run')
    args = ap.parse_args()

    man = _manifest.load(args.manifest, args.paths)
    roots, predictors = man['roots'], man['predictors']
    rep = Report(args.slack)

    for key, run in man['runs'].items():
        if args.only and key not in args.only:
            continue
        verify_run(key, run, roots, predictors, rep, args.verbose)

    n_bad = rep.print()
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
