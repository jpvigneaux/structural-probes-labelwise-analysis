#!/usr/bin/env python3
"""Rewrite matplotlib PNGs as plain 8-bit RGB, for portable pdflatex builds.

Matplotlib's Agg backend always writes RGBA PNGs, and it attaches ancillary
chunks (tEXt, pHYs). pdfTeX reads a PNG with an alpha channel through its
soft-mask path, which is the code path in which `pdfTeX error (pdflatex):
libpng: internal error` is reported; the failure depends on the libpng that a
particular TeX distribution was built against, so a document can compile on one
installation and fail on another with no change to the file.

This script removes the variable: every image becomes an opaque 8-bit RGB PNG
with no ancillary chunks, which is the most conservative form pdfTeX accepts.

Alpha is composited onto white, matching the figure background that the plotting
scripts set explicitly. Any file containing genuinely transparent pixels is
reported rather than silently altered, since for those the flattening is a
visible change and not merely a re-encoding.

Files whose names contain characters that are awkward in a LaTeX argument or in
a build system -- parentheses, brackets, spaces -- are reported too, with the
name they should be given; pass --rename to apply that, and update the
\\includegraphics paths to match.

Usage:
    python flatten_png_for_pdflatex.py imgs/*.png imgs/appendix/*.png
    python flatten_png_for_pdflatex.py --dry-run --rename imgs/*.png
"""

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image

UNSAFE = re.compile(r'[()\[\]{} ]')


def safe_name(name):
    """A filename with no characters that need escaping in a LaTeX argument."""
    stem, suffix = Path(name).stem, Path(name).suffix
    stem = re.sub(r'\((\d+)\)$', r'-\1', stem.strip())   # 'foo(1)' -> 'foo-1'
    stem = UNSAFE.sub('_', stem)
    return stem + suffix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', type=Path)
    ap.add_argument('--backup-dir', type=Path, default=None,
                    help='Copy each original here before rewriting it')
    ap.add_argument('--rename', action='store_true',
                    help='Also rename files whose names contain characters that '
                         'are awkward in LaTeX or in a build system')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if args.backup_dir:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    renames, transparent, converted, skipped = [], [], 0, 0
    before = after = 0

    for p in args.paths:
        if not p.is_file():
            print(f'missing: {p}')
            continue
        im = Image.open(p)
        im.load()
        size_before = p.stat().st_size
        needs = im.mode != 'RGB'

        if im.mode in ('RGBA', 'LA'):
            alpha = im.getchannel('A')
            lo = alpha.getextrema()[0]
            if lo < 255:
                transparent.append((p, lo))
            flat = Image.new('RGB', im.size, (255, 255, 255))
            flat.paste(im.convert('RGBA'), mask=im.convert('RGBA').getchannel('A'))
        elif im.mode == 'RGB':
            flat = im
        else:
            flat = im.convert('RGB')

        target = p
        if args.rename and UNSAFE.search(p.name):
            target = p.with_name(safe_name(p.name))
            renames.append((p, target))

        if not needs and target == p:
            skipped += 1
            before += size_before
            after += size_before
            continue

        if not args.dry_run:
            if args.backup_dir:
                shutil.copy2(p, args.backup_dir / p.name)
            # pnginfo is omitted, so no tEXt/pHYs/iCCP/eXIf survives.
            flat.save(target, format='PNG', optimize=True)
            if target != p:
                p.unlink()
        converted += 1
        before += size_before
        after += target.stat().st_size if (not args.dry_run and target.exists()) \
            else size_before

    print(f'rewrote {converted} file(s) as 8-bit RGB; {skipped} already RGB')
    if before:
        print(f'total size {before/1e6:.1f} MB -> {after/1e6:.1f} MB '
              f'({100*(before-after)/before:+.0f}%)')
    if transparent:
        print('\nfiles that contained genuinely transparent pixels (flattening '
              'onto white is a visible change for these -- check them):')
        for p, lo in transparent:
            print(f'  {p}  min alpha {lo}')
    if renames:
        verb = 'would rename' if args.dry_run else 'renamed'
        print(f'\n{verb} (update \\includegraphics accordingly):')
        for a, b in renames:
            print(f'  {a.name}  ->  {b.name}')


if __name__ == '__main__':
    main()
