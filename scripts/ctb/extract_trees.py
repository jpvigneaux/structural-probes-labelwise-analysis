#!/usr/bin/env python3
"""Chinese Treebank 9.0 -> bare Penn-style trees, one per line, split by file.

Input is the `data/bracketed/` directory of CTB 9.0 (LDC2016T13). Each file wraps
its trees in SGML (<DOC>, <S ID=...>, <seg>, ...). Every SGML tag starts its
line, while tree text never does, so dropping lines whose first non-blank
character is '<' removes the markup and keeps punctuation leaves such as `(PU <)`.

Per tree:
  * `-NONE-` leaves (empty categories: *pro*, *PRO*, *OP*, *T*-1, ...) are removed,
    and so are the constituents they leave empty, so that the dependency
    converter sees only the overt words the model reads;
  * a gapping index on a part-of-speech tag (`VV-2`) is removed;
  * a sentence containing a `URL`, `X` (foreign symbol), `NOI` (noise) or `EM`
    (emoticon) token is dropped whole: no relation should be scored on those
    tokens, and deleting only the token would change the text the model reads.

Split: CTB ships no train/dev/test split, so one is drawn over whole files,
stratified by genre: within each genre the files are shuffled with a fixed seed
and assigned to dev, then test, until each part holds that genre's share of the
requested token counts; the remaining files are train. The file lists are
written to --split-dir and reused on later runs, so the split is fixed once
drawn (pass --redraw to replace it).

    python scripts/ctb/extract_trees.py --ctb-root /path/to/ctb9.0 \
        --out-dir OUT --split-dir data_splits/ctb9-nmw
"""
import argparse
import collections
import glob
import os
import random
import re
import sys

DROP_TAGS = {'URL', 'X', 'NOI', 'EM'}
POS_INDEX = re.compile(r'^([A-Z]+)-\d+$')
SPLITS = ('train', 'dev', 'test')


def read_trees(path):
    """Yield each top-level tree of one CTB file as a nested list."""
    lines = [l for l in open(path, encoding='utf-8') if not l.lstrip().startswith('<')]
    toks = re.findall(r'\(|\)|[^\s()]+', ' '.join(lines))
    stack, i = [], 0
    for tok in toks:
        if tok == '(':
            stack.append([])
        elif tok == ')':
            node = stack.pop()
            if stack:
                stack[-1].append(node)
            else:
                yield node
        else:
            if not stack:
                sys.exit(f'{path}: text outside a tree: {tok!r}')
            stack[-1].append(tok)
    if stack:
        sys.exit(f'{path}: unbalanced brackets')


def is_preterminal(node):
    return len(node) == 2 and isinstance(node[0], str) and isinstance(node[1], str)


def prune(node):
    """Remove -NONE- leaves and emptied constituents; strip POS gapping indices.
    Returns the pruned node, or None when nothing overt is left under it."""
    if is_preterminal(node):
        if node[0] == '-NONE-':
            return None
        m = POS_INDEX.match(node[0])
        return [m.group(1) if m else node[0], node[1]]
    label = node[0] if node and isinstance(node[0], str) else None
    kids = [prune(k) for k in node[1 if label is not None else 0:]]
    kids = [k for k in kids if k is not None]
    if not kids:
        return None
    return ([label] if label is not None else []) + kids


def leaves(node):
    if is_preterminal(node):
        return [node]
    return [l for k in node if isinstance(k, list) for l in leaves(k)]


def to_str(node):
    if is_preterminal(node):
        return f'({node[0]} {node[1]})'
    label = node[0] if isinstance(node[0], str) else ''
    kids = ' '.join(to_str(k) for k in node if isinstance(k, list))
    return f'({label} {kids})' if label else f'( {kids})'


def draw_split(files_by_genre, tokens_of, dev_tokens, test_tokens, seed):
    total = sum(tokens_of[f] for fs in files_by_genre.values() for f in fs)
    parts = {s: [] for s in SPLITS}
    rng = random.Random(seed)
    for genre in sorted(files_by_genre):
        files = sorted(files_by_genre[genre])
        rng.shuffle(files)
        share = sum(tokens_of[f] for f in files) / total
        quota = {'dev': dev_tokens * share, 'test': test_tokens * share}
        got = {'dev': 0, 'test': 0}
        for f in files:
            part = next((p for p in ('dev', 'test') if got[p] < quota[p]), 'train')
            if part != 'train':
                got[part] += tokens_of[f]
            parts[part].append(f)
    return {s: sorted(v) for s, v in parts.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ctb-root', required=True)
    ap.add_argument('--genres', nargs='+', default=['nw', 'mz', 'wb'])
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--stem', default='ctb9-nmw')
    ap.add_argument('--split-dir', required=True, help='where the file-id lists live')
    ap.add_argument('--dev-tokens', type=int, default=40000)
    ap.add_argument('--test-tokens', type=int, default=57000)
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--redraw', action='store_true')
    args = ap.parse_args()

    files_by_genre = collections.defaultdict(list)
    for g in args.genres:
        for p in sorted(glob.glob(os.path.join(args.ctb_root, 'data', 'bracketed', f'*.{g}'))):
            files_by_genre[g].append(os.path.basename(p))

    kept, tokens_of = {}, {}
    stats = collections.Counter()
    for g, files in files_by_genre.items():
        for f in files:
            out = []
            for tree in read_trees(os.path.join(args.ctb_root, 'data', 'bracketed', f)):
                stats[g, 'trees'] += 1
                t = prune(tree)
                if t is None:
                    stats[g, 'empty'] += 1
                    continue
                if any(tag in DROP_TAGS for tag, _ in leaves(t)):
                    stats[g, 'dropped'] += 1
                    continue
                out.append(t)
            kept[f] = out
            tokens_of[f] = sum(len(leaves(t)) for t in out)

    os.makedirs(args.split_dir, exist_ok=True)
    lists = {s: os.path.join(args.split_dir, f'{s}.txt') for s in SPLITS}
    if args.redraw or not all(os.path.exists(p) for p in lists.values()):
        split = draw_split(files_by_genre, tokens_of, args.dev_tokens, args.test_tokens,
                           args.seed)
        for s, p in lists.items():
            with open(p, 'w') as fh:
                fh.write(f'# CTB 9.0 {args.stem} {s} files; seed {args.seed}, '
                         f'dev {args.dev_tokens} / test {args.test_tokens} tokens, '
                         f'stratified by genre\n')
                fh.write(''.join(f + '\n' for f in split[s]))
        print('drew a new split ->', args.split_dir)
    split = {s: [l.strip() for l in open(p) if l.strip() and not l.startswith('#')]
             for s, p in lists.items()}
    assigned = [f for s in SPLITS for f in split[s]]
    if sorted(assigned) != sorted(kept) or len(set(assigned)) != len(assigned):
        sys.exit('split lists do not partition the selected files; use --redraw')

    os.makedirs(args.out_dir, exist_ok=True)
    genre_of = {f: f.rsplit('.', 1)[1] for f in kept}
    table = collections.Counter()
    for s in SPLITS:
        with open(os.path.join(args.out_dir, f'{args.stem}-{s}.trees'), 'w',
                  encoding='utf-8') as fh:
            for f in split[s]:
                for t in kept[f]:
                    fh.write(to_str(t) + '\n')
                    table[s, genre_of[f], 'sentences'] += 1
                    table[s, genre_of[f], 'tokens'] += len(leaves(t))
                table[s, genre_of[f], 'files'] += 1

    print(f"{'genre':6}{'trees':>8}{'empty':>7}{'dropped':>9}")
    for g in args.genres:
        print(f"{g:6}{stats[g, 'trees']:8}{stats[g, 'empty']:7}{stats[g, 'dropped']:9}")
    print(f"\n{'split':6}{'genre':6}{'files':>6}{'sentences':>10}{'tokens':>9}")
    for s in SPLITS:
        for g in args.genres:
            print(f"{s:6}{g:6}{table[s, g, 'files']:6}{table[s, g, 'sentences']:10}"
                  f"{table[s, g, 'tokens']:9}")
        print(f"{s:6}{'all':6}{sum(table[s, g, 'files'] for g in args.genres):6}"
              f"{sum(table[s, g, 'sentences'] for g in args.genres):10}"
              f"{sum(table[s, g, 'tokens'] for g in args.genres):9}")


if __name__ == '__main__':
    main()
