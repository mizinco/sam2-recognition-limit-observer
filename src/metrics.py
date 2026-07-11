"""
SAM2 認識限界観測システム — 観測指標
============================================

3つの定量指標を計算する関数群:

1. mask_area_change_rate    — マスク面積急変率 (パターン2/4/5/6 対応)
2. connected_components_count — マスク連結成分数 (パターン3/5 対応)
3. mask_disappearance       — マスク消失検出 (パターン1 対応)

入力: bool 配列 masks (shape: T x H x W)
"""

from typing import Dict
import numpy as np
from scipy import ndimage


# --- 指標1: マスク面積急変率 -------------------------------------------------

def mask_area_change_rate(masks: np.ndarray) -> np.ndarray:
    """
    隣接フレーム間のマスク面積変化率を返す。

    change_rate[t] = (area[t] - area[t-1]) / max(area[t-1], 1)

    Args:
        masks: bool 配列 (T, H, W)

    Returns:
        change_rate: float32 配列 (T,)。change_rate[0] = 0
        前フレームが面積0の場合、現フレームも0なら0、そうでなければ +inf
    """
    areas = masks.sum(axis=(1, 2)).astype(np.float64)
    change_rate = np.zeros(len(areas), dtype=np.float32)
    for t in range(1, len(areas)):
        prev = areas[t - 1]
        curr = areas[t]
        if prev == 0:
            change_rate[t] = 0.0 if curr == 0 else float('inf')
        else:
            change_rate[t] = (curr - prev) / prev
    return change_rate


# --- 指標2: マスク連結成分数 -------------------------------------------------

def connected_components_count(masks: np.ndarray, min_size: int = 50) -> np.ndarray:
    """
    各フレームのマスクに含まれる連結成分数を返す。
    min_size 未満の小さな塊（ノイズ）は除外。

    Args:
        masks: bool 配列 (T, H, W)
        min_size: 成分とみなす最小ピクセル数

    Returns:
        counts: int32 配列 (T,)
    """
    counts = np.zeros(len(masks), dtype=np.int32)
    for t in range(len(masks)):
        if masks[t].sum() == 0:
            counts[t] = 0
            continue
        labeled, n = ndimage.label(masks[t])
        if n == 0:
            counts[t] = 0
        elif min_size > 0:
            sizes = ndimage.sum(masks[t], labeled, range(1, n + 1))
            counts[t] = int((sizes >= min_size).sum())
        else:
            counts[t] = n
    return counts


def largest_component_ratio(masks: np.ndarray) -> np.ndarray:
    """
    補助指標: 最大連結成分の面積 / 総マスク面積。
    1.0 = 単一成分、< 1.0 = 断片化。
    総面積 0 のフレームは 1.0 を返す（消失と区別したい場合は別指標と組み合わせる）。
    """
    ratios = np.ones(len(masks), dtype=np.float32)
    for t in range(len(masks)):
        total = masks[t].sum()
        if total == 0:
            ratios[t] = 1.0
            continue
        labeled, n = ndimage.label(masks[t])
        if n == 0:
            ratios[t] = 1.0
            continue
        sizes = ndimage.sum(masks[t], labeled, range(1, n + 1))
        ratios[t] = float(sizes.max() / total)
    return ratios


# --- 指標4: 形状の時間的安定性（重心揃え後の連続フレーム IoU）---------------

def shape_iou_temporal(masks: np.ndarray) -> np.ndarray:
    """
    隣接フレーム間で重心を揃えた後の IoU を返す。
    位置の自然な変化を打ち消し、形状の変化のみを評価する。

    Args:
        masks: bool 配列 (T, H, W)

    Returns:
        iou: float32 配列 (T,)。iou[0] = 1.0（前フレーム無し）
             どちらかのマスクが空のフレームは 1.0（未定義扱い、消失検出と二重計上しない）
    """
    T = len(masks)
    iou = np.ones(T, dtype=np.float32)
    if T == 0:
        return iou

    H, W = masks[0].shape

    def _centroid(m: np.ndarray):
        ys, xs = np.where(m)
        if len(ys) == 0:
            return None
        return float(ys.mean()), float(xs.mean())

    for t in range(1, T):
        m_prev = masks[t - 1]
        m_curr = masks[t]
        if m_prev.sum() == 0 or m_curr.sum() == 0:
            iou[t] = 1.0
            continue

        c_prev = _centroid(m_prev)
        c_curr = _centroid(m_curr)
        dy = int(round(c_prev[0] - c_curr[0]))
        dx = int(round(c_prev[1] - c_curr[1]))

        # m_curr を (dy, dx) だけシフト
        shifted = np.zeros_like(m_curr)
        y_dst_start = max(0, dy)
        y_dst_end = min(H, H + dy)
        x_dst_start = max(0, dx)
        x_dst_end = min(W, W + dx)
        if y_dst_start < y_dst_end and x_dst_start < x_dst_end:
            shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = \
                m_curr[y_dst_start - dy:y_dst_end - dy,
                       x_dst_start - dx:x_dst_end - dx]

        inter = np.logical_and(m_prev, shifted).sum()
        union = np.logical_or(m_prev, shifted).sum()
        iou[t] = float(inter / union) if union > 0 else 1.0

    return iou


# --- 指標3: マスク消失検出 ---------------------------------------------------

def mask_disappearance(
    masks: np.ndarray,
    threshold_abs: int = 50,
    threshold_rel: float = 0.01,
    ref_frames: int = 5,
) -> Dict[str, np.ndarray]:
    """
    マスク消失を検出する。

    閾値は以下の最大値を採用:
      - 絶対閾値: threshold_abs (例: 50px 未満で消失)
      - 相対閾値: 最初の ref_frames フレーム平均 × threshold_rel

    Args:
        masks: bool 配列 (T, H, W)
        threshold_abs: 絶対閾値（ピクセル数）
        threshold_rel: 相対閾値（基準面積に対する比率）
        ref_frames: 基準面積の計算に使う先頭フレーム数

    Returns:
        dict with:
            - areas: 各フレームの面積 (T,)
            - is_disappeared: 各フレームで消失中か (T,) bool
            - consecutive: 各フレームまでの連続消失数 (T,) int
            - threshold: 採用された絶対閾値 (float)
            - reference_area: 基準面積 (float)
    """
    areas = masks.sum(axis=(1, 2)).astype(np.float64)
    ref = areas[:max(ref_frames, 1)]
    ref_area = float(ref.mean()) if len(ref) > 0 else 1.0
    threshold = max(float(threshold_abs), ref_area * threshold_rel)

    is_disappeared = areas < threshold
    consecutive = np.zeros(len(areas), dtype=np.int32)
    for t in range(len(areas)):
        if is_disappeared[t]:
            consecutive[t] = (consecutive[t - 1] if t > 0 else 0) + 1
        else:
            consecutive[t] = 0

    return {
        'areas': areas,
        'is_disappeared': is_disappeared,
        'consecutive': consecutive,
        'threshold': threshold,
        'reference_area': ref_area,
    }


# --- 全指標を一括計算 --------------------------------------------------------

def compute_all_metrics(masks: np.ndarray, min_component_size: int = 50) -> Dict:
    """
    全4指標（+ 補助指標）を計算し、JSON シリアライズしやすい dict を返す。
    """
    area_change = mask_area_change_rate(masks)
    components = connected_components_count(masks, min_size=min_component_size)
    largest_ratio = largest_component_ratio(masks)
    disappear = mask_disappearance(masks)
    shape_iou = shape_iou_temporal(masks)

    return {
        'frame_count': int(len(masks)),
        'areas': disappear['areas'].astype(float).tolist(),
        'area_change_rate': area_change.tolist(),
        'connected_components': components.tolist(),
        'largest_component_ratio': largest_ratio.tolist(),
        'shape_iou_temporal': shape_iou.tolist(),
        'is_disappeared': disappear['is_disappeared'].tolist(),
        'consecutive_disappearance': disappear['consecutive'].tolist(),
        'reference_area': float(disappear['reference_area']),
        'disappearance_threshold': float(disappear['threshold']),
        'min_component_size': int(min_component_size),
    }
