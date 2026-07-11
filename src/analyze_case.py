"""
1ケース分の masks.npz を読み、指標を計算し、JSON と可視化 PNG を出力するスクリプト。

使い方:
    python src/analyze_case.py results/week2/parked_car_001/masks.npz
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 同ディレクトリの metrics をインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_all_metrics


def _try_set_jp_font():
    """日本語フォントを使えれば設定。失敗しても英語にフォールバック"""
    candidates = [
        'Hiragino Sans', 'Hiragino Sans GB', 'Yu Gothic',
        'Noto Sans CJK JP', 'IPAGothic', 'TakaoGothic',
        'DejaVu Sans',
    ]
    import matplotlib
    for name in candidates:
        matplotlib.rcParams['font.family'] = name
        try:
            from matplotlib.font_manager import findfont, FontProperties
            findfont(FontProperties(family=name), fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def analyze(npz_path: str, output_dir: str = None) -> dict:
    """
    masks.npz を解析、metrics.json と metrics.png を保存。

    Args:
        npz_path: masks.npz のパス
        output_dir: 出力先（省略時は npz と同じディレクトリ）

    Returns:
        metrics dict
    """
    data = np.load(npz_path, allow_pickle=True)
    masks = data['masks']

    # ディレクトリ名を case_name の正とする（npz 内の値が誤っていることがあるため）
    case_name = Path(npz_path).parent.name
    if 'case_name' in data:
        embedded = str(data['case_name'].item() if data['case_name'].ndim == 0 else data['case_name'][0])
        if embedded != case_name:
            print(f'  [WARN] case_name mismatch: dir="{case_name}" vs npz="{embedded}", using dir name')

    click_xy = data['click_xy'].tolist() if 'click_xy' in data else None

    print(f'[{case_name}] loaded: masks shape={masks.shape}, dtype={masks.dtype}')

    metrics = compute_all_metrics(masks)
    metrics['case_name'] = case_name
    metrics['click_xy'] = click_xy

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(npz_path))
    os.makedirs(output_dir, exist_ok=True)

    # --- JSON ---
    json_path = os.path.join(output_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f'[{case_name}] saved: {json_path}')

    # --- PNG (5-panel unified format, English labels) ---
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    frames = np.arange(len(masks))

    # 0. Mask area + threshold
    axes[0].plot(frames, metrics['areas'], color='steelblue', linewidth=1.5)
    axes[0].axhline(
        metrics['disappearance_threshold'],
        color='red', linestyle='--',
        label=f"Disappearance threshold ({metrics['disappearance_threshold']:.0f}px)",
    )
    axes[0].set_ylabel('Mask area (px)')
    axes[0].set_title(f'{case_name} - mask area over time')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    # 1. Area change rate
    rates = np.array(metrics['area_change_rate'], dtype=float)
    rates_disp = np.clip(rates, -2.0, 2.0)
    axes[1].plot(frames, rates_disp, color='darkorange', linewidth=1.5)
    axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='+-50%')
    axes[1].axhline(-0.5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Area change rate')
    axes[1].set_title('Indicator 1: Mask area change rate (clipped to +-2.0)')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    # 2. Connected components
    axes[2].plot(
        frames, metrics['connected_components'],
        color='seagreen', linewidth=1.5, marker='o', markersize=3,
    )
    axes[2].set_ylabel('Component count')
    axes[2].set_title(
        f'Indicator 2: Connected components (>={metrics["min_component_size"]}px only)'
    )
    axes[2].grid(alpha=0.3)

    # 3. Disappearance (Indicator 3)
    is_dis = np.array(metrics['is_disappeared'])
    cons = np.array(metrics['consecutive_disappearance'])
    axes[3].fill_between(
        frames, 0, is_dis.astype(int),
        alpha=0.4, color='crimson', step='post', label='Disappeared',
    )
    if cons.max() > 0:
        axes[3].plot(
            frames, cons / max(cons.max(), 1),
            color='crimson', linewidth=1.5, label='Consecutive (normalized)',
        )
    axes[3].set_ylim(-0.05, 1.1)
    axes[3].set_ylabel('Disappearance')
    axes[3].set_title(
        f'Indicator 3: Mask disappearance (max consecutive: {int(cons.max())} frames)'
    )
    axes[3].legend(loc='upper left')
    axes[3].grid(alpha=0.3)

    # 4. Shape IoU (Indicator 4)
    siou = np.array(metrics.get('shape_iou_temporal', []), dtype=float)
    if len(siou) > 0:
        axes[4].plot(frames, siou, color='purple', linewidth=1.5)
        axes[4].axhline(0.8, color='red', linestyle='--', alpha=0.5, label='IoU 0.8')
        axes[4].set_ylim(0, 1.05)
        min_idx = int(np.argmin(siou))
        axes[4].set_title(
            f'Indicator 4: Shape stability (centroid-aligned IoU, min {float(siou.min()):.3f} at frame {min_idx})'
        )
    else:
        axes[4].set_title('Indicator 4: Shape IoU (no data)')
    axes[4].set_ylabel('Shape IoU')
    axes[4].set_xlabel('Frame')
    axes[4].legend(loc='lower right')
    axes[4].grid(alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(output_dir, 'metrics.png')
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[{case_name}] saved: {png_path}')

    # --- サマリ表示 ---
    print(f'[{case_name}] summary:')
    print(f'  - frames: {metrics["frame_count"]}')
    print(f'  - reference area: {metrics["reference_area"]:.0f}px')
    print(f'  - disappearance threshold: {metrics["disappearance_threshold"]:.0f}px')
    print(f'  - frames with disappearance: {int(is_dis.sum())} / {len(masks)}')
    print(f'  - max consecutive disappearance: {int(cons.max())}')
    print(f'  - max components: {int(max(metrics["connected_components"]))}')
    abrupt = (np.abs(rates) > 0.5).sum()
    print(f'  - frames with |area change rate| > 50%: {int(abrupt)}')
    if len(siou) > 0:
        print(f'  - min shape IoU: {float(siou.min()):.3f} at frame {int(np.argmin(siou))}')
        print(f'  - mean shape IoU (frames 1+): {float(siou[1:].mean()):.3f}')

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='Path to masks.npz')
    parser.add_argument('--out', default=None, help='Output directory')
    args = parser.parse_args()
    analyze(args.npz, args.out)


if __name__ == '__main__':
    main()
