#!/usr/bin/env python3
"""Query experiments/paper_runs.yaml from a shell script.

The driver scripts take a run key (bertbase, deberta, modernbert, gpt2, gptj,
shufflen1) and need that run's model id, layout, paths and checkpoint list. Those
live in one place -- the manifest -- so that a driver and the verification script
cannot disagree about which run they are talking about.

    $ manifest.py gptj extraction.hf_name
    EleutherAI/gpt-j-6B
    $ manifest.py gptj post_block_checkpoints        # lists print space-separated
    1 2 3 ... 28
    $ manifest.py --path gptj results                # {root} placeholders expanded
    /your/scratch/lnconv/results-gptj-convA
    $ manifest.py --keys                             # every run key
    bertbase deberta modernbert gpt2 gptj shufflen1
    $ manifest.py --root embeddings

Missing keys exit non-zero with a message, so `set -e` in a driver stops rather
than proceeding with an empty variable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'scripts'))
import _manifest                                              # noqa: E402

MANIFEST = _manifest.MANIFEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?')
    ap.add_argument('field', nargs='?',
                    help='dotted path within the run, e.g. extraction.hf_name')
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--paths', default=None,
                    help='paths.yaml holding roots: and predictors: '
                         '(default: the repository root)')
    ap.add_argument('--path', action='store_true',
                    help='expand {root} placeholders against the roots: block')
    ap.add_argument('--keys', action='store_true', help='list every run key')
    ap.add_argument('--root', help='print one entry of the roots: block')
    ap.add_argument('--predictor', help='print one entry of the predictors: block')
    ap.add_argument('--default', help='print this instead of failing if absent')
    args = ap.parse_args()

    man = _manifest.load(args.manifest, args.paths)

    if args.keys:
        print(' '.join(man['runs']))
        return
    if args.root:
        print(man['roots'][args.root])
        return
    if args.predictor:
        print(man['predictors'][args.predictor])
        return

    if not args.run or not args.field:
        sys.exit('usage: manifest.py RUN FIELD  (or --keys / --root R / --predictor P)')
    if args.run not in man['runs']:
        sys.exit(f"unknown run '{args.run}'; known: {' '.join(man['runs'])}")

    node = man['runs'][args.run]
    for part in args.field.split('.'):
        if not isinstance(node, dict) or part not in node:
            if args.default is not None:
                print(args.default)
                return
            sys.exit(f"'{args.field}' not set for run '{args.run}'")
        node = node[part]

    if isinstance(node, list):
        print(' '.join(str(x) for x in node))
    elif isinstance(node, bool):
        print('1' if node else '0')
    else:
        s = str(node)
        print(s.format(**man['roots']) if args.path else s)


if __name__ == '__main__':
    main()
