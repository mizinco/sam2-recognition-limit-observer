"""
全ケースを一括解析するスクリプト。

使い方:
    python src/run_all.py results/week2
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_case import analyze


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'base_dir',
        help='ベースディレクトリ。配下の masks.npz を全部処理',
    )
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f'ERROR: directory not found: {base}')
        sys.exit(1)

    npz_paths = sorted(base.rglob('masks.npz'))
    if not npz_paths:
        print(f'ERROR: no masks.npz found under {base}')
        sys.exit(1)

    print(f'Found {len(npz_paths)} cases')
    for npz in npz_paths:
        print()
        print('=' * 60)
        analyze(str(npz), output_dir=str(npz.parent))

    print()
    print('=' * 60)
    print(f'Done. {len(npz_paths)} cases processed.')


if __name__ == '__main__':
    main()
