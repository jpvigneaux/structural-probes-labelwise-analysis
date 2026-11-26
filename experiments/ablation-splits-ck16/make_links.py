#!/usr/bin/env python3
"""Per-split embedding and alignment files for the re-split ablation at checkpoint 16.

The 20 stratified re-splits of the PTB (generate_splits.py, split_metadata.json) are
those of the earlier checkpoint-13 ablation. Each split's HDF5 is a small file of
ExternalLinks into the paper's own embeddings (the -convB files) and alignments, so the
probe reads exactly the representations of the published BERT-base run and nothing
is copied. The alignment files' root attributes (special-token counts, read by
data.BERTDataset) are copied onto the link files.

    python make_links.py OUT_ROOT EMB_DIR ALIGN_DIR
"""
import json
import sys
from pathlib import Path

import h5py

STEM = 'bertbase-natural-layers'


def link(out_path, provenance, sources):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as fh:
        with h5py.File(sources['dev'], 'r') as src:     # attrs are identical across splits
            for k, v in src.attrs.items():
                fh.attrs[k] = v
        for new_idx, (orig_split, orig_idx) in enumerate(provenance):
            fh[str(new_idx)] = h5py.ExternalLink(sources[orig_split], str(orig_idx))


def main():
    out_root, emb_dir, align_dir = map(Path, sys.argv[1:4])
    emb = {s: str(emb_dir / f'raw.{s}.{STEM}.hdf5') for s in ('train', 'dev', 'test')}
    aln = {s: str(align_dir / f'alignments.{s}.hdf5') for s in ('train', 'dev', 'test')}
    for p in list(emb.values()) + list(aln.values()):
        if not Path(p).is_file():
            sys.exit(f'missing {p}')
    meta = json.load(open(Path(__file__).parent / 'split_metadata.json'))
    splits = meta['splits'] if 'splits' in meta else meta
    for key in sorted(splits):
        for part in ('train', 'dev', 'test'):
            prov = splits[key][part]
            link(out_root / 'embeddings' / key / f'{part}.hdf5', prov, emb)
            link(out_root / 'alignments' / key / f'{part}.hdf5', prov, aln)
        print(key, {p: len(splits[key][p]) for p in ('train', 'dev', 'test')})


if __name__ == '__main__':
    main()
