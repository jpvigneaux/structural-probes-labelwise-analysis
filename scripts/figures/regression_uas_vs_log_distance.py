#!/usr/bin/env python3
"""
For each (checkpoint, relation) pair, fit the linear model

    UAS_δ  ~  a  +  b · ln(δ)

using the raw observed data points (n, UAS_n) with |S_n| >= 5 (the same
filter applied when building the NPZ).  Reports R², slope b, intercept a,
number of points, p-value for the slope, standard error of the slope, and
RMSE.

Outputs
-------
  tables/regression_uas_vs_log_distance.csv   — full flat table
  tables/regression_uas_vs_log_distance.html  — same, sortable in browser
  figures/r2_heatmap_uas_vs_log_distance.png  — R² heatmap (relation × checkpoint)

Usage (login node):
    python regression_uas_vs_log_distance.py \
        [--curves figures/uuas_mean_curves_by_checkpoint.npz]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).parent
    p.add_argument('--curves',
                   default=str(script_dir / 'figures'
                               / 'uuas_mean_curves_by_checkpoint.npz'))
    p.add_argument('--min-points', type=int, default=3,
                   help='Min observed distances required to fit (default: 3)')
    p.add_argument('--relations', nargs='+', default=None,
                   help='Draw only these relations, in decreasing mean R^2 as '
                        'usual. Use to restrict the figure to the relations the '
                        'text actually discusses; the CSV is unaffected and '
                        'still covers every relation.')
    p.add_argument('--out', default=None,
                   help='Output PNG (default: figures/r2_heatmap_uas_vs_log_distance.png)')
    return p.parse_args()


def main():
    args      = parse_args()
    npz       = np.load(args.curves, allow_pickle=False)
    script_dir = Path(args.curves).parent.parent

    (script_dir / 'tables').mkdir(exist_ok=True)
    (script_dir / 'figures').mkdir(exist_ok=True)

    checkpoints = npz['checkpoints'].tolist()
    relations   = npz['rel_names'].astype(str).tolist()
    keep = None
    if args.relations:
        missing = [r for r in args.relations if r not in relations]
        if missing:
            raise SystemExit(f'not in the curves file: {missing}\n'
                             f'available: {sorted(relations)}')
        keep = set(args.relations)
    n_ck  = len(checkpoints)
    n_rel = len(relations)

    # ── Run regressions ───────────────────────────────────────────────────────
    records = []
    for ci, ck in enumerate(checkpoints):
        for ri, rel in enumerate(relations):
            ns  = npz[f'rel_ns_{ci}_{ri}'].astype(float)
            uas = npz[f'rel_uas_{ci}_{ri}']

            if len(ns) < args.min_points:
                records.append(dict(
                    checkpoint=ck, relation=rel,
                    n_points=len(ns),
                    slope=np.nan, intercept=np.nan,
                    R2=np.nan, p_value=np.nan,
                    slope_se=np.nan, rmse=np.nan,
                ))
                continue

            x = np.log(ns)
            result = stats.linregress(x, uas)

            fitted = result.slope * x + result.intercept
            rmse   = float(np.sqrt(np.mean((uas - fitted) ** 2)))

            records.append(dict(
                checkpoint  = ck,
                relation    = rel,
                n_points    = len(ns),
                slope       = round(float(result.slope),     4),
                intercept   = round(float(result.intercept), 4),
                R2          = round(float(result.rvalue**2), 4),
                p_value     = float(result.pvalue),
                slope_se    = round(float(result.stderr),    4),
                rmse        = round(rmse,                    4),
            ))

    df = pd.DataFrame(records)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = script_dir / 'tables' / 'regression_uas_vs_log_distance.csv'
    df.to_csv(csv_path, index=False)
    print(f'CSV  → {csv_path}')

    # ── Sortable HTML table ───────────────────────────────────────────────────
    html_path = script_dir / 'tables' / 'regression_uas_vs_log_distance.html'

    # Format p-value in scientific notation for readability
    df_disp = df.copy()
    df_disp['p_value'] = df_disp['p_value'].apply(
        lambda v: f'{v:.2e}' if pd.notna(v) else ''
    )

    table_html = df_disp.to_html(index=False, na_rep='—', border=0,
                                  classes='tbl', float_format='{:.4f}'.format)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UAS ~ a + b·ln(δ)  regression results</title>
<style>
  body {{ font-family: sans-serif; font-size: 13px; padding: 16px; }}
  h2   {{ margin-bottom: 6px; }}
  .tbl {{ border-collapse: collapse; width: 100%; }}
  .tbl th, .tbl td {{ border: 1px solid #ccc; padding: 4px 10px;
                      text-align: right; white-space: nowrap; }}
  .tbl th {{ background: #f0f0f0; cursor: pointer; user-select: none; }}
  .tbl th:hover {{ background: #ddd; }}
  .tbl tr:nth-child(even) {{ background: #fafafa; }}
  .tbl td:nth-child(1), .tbl td:nth-child(2) {{ text-align: left; }}
</style>
</head>
<body>
<h2>Linear regression: mean UASL ~ <i>a</i> + <i>b</i>·ln(<i>&delta;</i>)</h2>
<p>Each row is one (checkpoint, relation) pair. Only pairs with at least
{args.min_points} observed distances (|S_n| ≥ 5) are fitted;
others show —.</p>
{table_html}
<script>
// Click a column header to sort the table by that column.
document.querySelectorAll('.tbl th').forEach((th, ci) => {{
  let asc = true;
  th.addEventListener('click', () => {{
    const tbody = th.closest('table').querySelector('tbody');
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const va = a.cells[ci].textContent.trim();
      const vb = b.cells[ci].textContent.trim();
      const na = parseFloat(va), nb = parseFloat(vb);
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }});
    rows.forEach(r => tbody.appendChild(r));
    asc = !asc;
  }});
}});
</script>
</body>
</html>
"""
    html_path.write_text(html, encoding='utf-8')
    print(f'HTML → {html_path}')

    # ── R² heatmap ────────────────────────────────────────────────────────────
    r2_wide = df.pivot(index='relation', columns='checkpoint', values='R2')
    if keep is not None:
        # Restrict the drawn rows only; the CSV above still covers everything,
        # so the median R^2 the paper quotes is unaffected by this filter.
        r2_wide = r2_wide.loc[[r for r in r2_wide.index if r in keep]]
    # Sort relations by mean R² descending
    r2_wide = r2_wide.loc[r2_wide.mean(axis=1).sort_values(ascending=False).index]
    # Drop relations that have no valid R² across all checkpoints
    r2_wide = r2_wide[~r2_wide.isna().all(axis=1)]

    # Both dimensions scale with the matrix so that a cell keeps the same printed
    # size whatever is being drawn. Width: 12 columns (BERT-base, DeBERTa-v3-base,
    # GPT-2) reproduce the original 5.0in, while ModernBERT-base (22) and GPT-J
    # (28) widen instead of compressing their annotations. Height likewise, since
    # --relations can cut the 41 rows down to the handful the text discusses;
    # leaving it fixed would stretch eight rows over the height of forty-one.
    n_col, n_row = len(r2_wide.columns), len(r2_wide)
    fig_w = 1.04 + 0.33 * n_col
    fig_h = max(2.2, 0.95 + 0.175 * n_row)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad(color='#e0e0e0')   # grey for NaN

    im = ax.imshow(r2_wide.values.astype(float), aspect='auto',
                   vmin=0, vmax=1, cmap=cmap)

    ax.set_xticks(range(len(checkpoints)))
    ax.set_xticklabels([str(c) for c in r2_wide.columns], fontsize=9)
    ax.set_yticks(range(len(r2_wide)))
    ax.set_yticklabels(r2_wide.index.tolist(), fontsize=7)
    ax.set_xlabel('Residual stream checkpoint', fontsize=10)
    ax.set_ylabel('Relation', fontsize=10)
    ax.set_title(r'$R^2$: mean UASL ~ $a + b\,\ln \delta$', fontsize=10)

    # Annotate cells with R² value
    for ri in range(len(r2_wide)):
        for ci in range(len(r2_wide.columns)):
            v = r2_wide.values[ri, ci]
            if np.isnan(v):
                continue
            color = 'black' if 0.2 < v < 0.85 else 'white'
            ax.text(ci, ri, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, shrink=0.9)
    cbar.set_label('$R^2$', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    plt.tight_layout()
    heatmap_path = script_dir / 'figures' / 'r2_heatmap_uas_vs_log_distance.png'
    if args.out:
        heatmap_path = Path(args.out)
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f'Heatmap → {heatmap_path}')

    # ── Quick summary to stdout ───────────────────────────────────────────────
    fitted = df.dropna(subset=['R2'])
    print(f'\nFitted pairs : {len(fitted)} / {len(df)}')
    print(f'Median R²    : {fitted["R2"].median():.3f}')
    print(f'Mean   R²    : {fitted["R2"].mean():.3f}')
    print(f'R² ≥ 0.90    : {(fitted["R2"] >= 0.90).sum()} pairs')
    print(f'R² ≥ 0.95    : {(fitted["R2"] >= 0.95).sum()} pairs')
    print(f'\nTop-10 by R² (averaged over checkpoints):')
    top = (fitted.groupby('relation')['R2']
                 .mean()
                 .sort_values(ascending=False)
                 .head(10))
    for rel, val in top.items():
        print(f'  {rel:<15}  {val:.3f}')
    print(f'\nBottom-10 by R²:')
    bot = (fitted.groupby('relation')['R2']
                 .mean()
                 .sort_values()
                 .head(10))
    for rel, val in bot.items():
        print(f'  {rel:<15}  {val:.3f}')


if __name__ == '__main__':
    main()
