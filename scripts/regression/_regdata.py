"""The join behind every regression in the paper, in one place.

All the scripts in this directory join the same three files on the dependency
relation label, and they must join them identically or their R^2 values stop
being comparable -- which matters, because the paper prints them side by side.
Each currently carries its own copy of that join, differing only in which columns
it builds; `scripts/verify_paper_numbers.py` uses this module instead, so if one
of those copies ever drifts, the verification pass is what notices.

    dev.uuas_by_relation   relation, uuas, correct, total   (per probing run)
    <sim>.tsv              deprel, head_sim_entropy_bits    (PTB train, once)
    dep-lengths-ptb.tsv    deprel, mean_length, mean_log_length,
                           stdev_log_length, skew_log_length  (PTB train, once)

No label mapping is performed. The three files must already use the same
relation scheme; if they do not, the join comes up empty rather than silently
pairing unrelated relations, which is the failure mode that mapping between
PTB and UD schemes invites.

`total` is the number of dev-set edges a relation's ULAS was estimated from, and
is the regression weight throughout: ULAS is a proportion, so its sampling
variance scales as 1/total, and relations attested a handful of times must not
carry the same weight as `prep` (n = 3783).

Rows with a missing value in any *requested* column are dropped, and only then,
so that adding an unused column (e.g. skewness) cannot change the sample a
comparison is fitted on.
"""

import numpy as np
import pandas as pd

# The columns every caller may ask for, and where each comes from.
DERIVED = ['uuas', 'total', 'head_sim_entropy', 'mean_log_length',
           'log_mean_length', 'sd_log_length', 'var_log_length',
           'skew_log_length']


def load_regression_frame(uuas_path, sim_path, len_path, columns=None):
    """Join the three inputs and return one frame indexed by relation.

    columns: iterable of column names to require non-missing. Defaults to
    everything in DERIVED, which is what a verification pass wants; a script
    fitting a specific model should pass the columns that model uses, so that
    an unrelated missing value does not shrink its sample.
    """
    uuas = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in uuas.columns and {'correct', 'uuas'}.issubset(uuas.columns):
        uuas['total'] = uuas['correct'] / uuas['uuas']
    uuas = uuas.set_index('relation')
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')

    missing = [c for c in ('mean_length', 'mean_log_length', 'stdev_log_length')
               if c not in length.columns]
    if missing:
        raise SystemExit(f'{len_path} lacks {missing}; re-run '
                         f'scripts/regression/arc_length_moments.py to regenerate it.')

    common = uuas.index.intersection(sim.index).intersection(length.index)
    if len(common) == 0:
        raise SystemExit(
            'No relations in common across the three files. Do they use the '
            'same dependency-relation label scheme?')

    sd = length.loc[common, 'stdev_log_length']
    df = pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'head_sim_entropy': sim.loc[common, 'head_sim_entropy_bits'],
        'mean_log_length': length.loc[common, 'mean_log_length'],
        'log_mean_length': np.log(length.loc[common, 'mean_length']),
        'sd_log_length': sd,
        'var_log_length': sd ** 2,
    })
    # Optional: only add_length_spread.py and the moment-ladder table use it.
    if 'skew_log_length' in length.columns:
        df['skew_log_length'] = length.loc[common, 'skew_log_length']

    cols = [c for c in (columns if columns is not None else DERIVED)
            if c in df.columns]
    return df.dropna(subset=cols)
