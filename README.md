# sam2-recognition-limit-observer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![SAM2](https://img.shields.io/badge/Model-SAM2.1%20Hiera--Large-blue)](https://github.com/facebookresearch/sam2)

SAM2 (Segment Anything Model 2) の動画オブジェクト追跡が「いつ・どのように破綻するか」を定量指標で観測・可視化するツール。任意の動画クリップに対して追跡マスクの品質を 4 つの指標で監視し、消失・分裂・形状崩れ・別物体への乗り換えといった破綻パターンを検出する。

## 観測指標（4 つ）

| 指標 | 内容 | 検出する破綻 |
|---|---|---|
| 1. マスク面積急変率 | 前フレーム比の面積変動 | 急激な拡大・縮小 |
| 2. 連結成分数 | マスクの断片数（50px 以上のみ） | 分断・破れ |
| 3. 消失検出 | 閾値以下の連続フレーム数 | 完全消失 |
| 4. 形状の時間的 IoU | 重心揃え後のフレーム間 IoU | 形状の崩れ |

実装の詳細は [src/metrics.py](src/metrics.py) を参照。

指標には限界もある。たとえば追跡対象が滑らかに別の物体へ「乗り換える」ケースでは、面積・形状が連続的に変化するため 4 指標すべてが破綻を見逃すことがある。こうした指標の死角の観測もこのツールの目的の一つ。

## リポジトリ構成

```
.
├── README.md
├── requirements.txt       Python 依存
├── notebooks/
│   ├── sam2_video_predictor_starter.ipynb   SAM2 動画追跡の最小構成ノートブック
│   └── sam2_gradio.ipynb                    Gradio UI（クリック指定 → 追跡 → 指標可視化 → フォーカス動画生成）
├── src/
│   ├── metrics.py         4 指標 + 補助指標の実装
│   ├── analyze_case.py    masks.npz → metrics.json + metrics.png 解析
│   └── run_all.py         全ケース一括解析
├── data/                  入力クリップ置き場（gitignore）
└── results/               解析結果（metrics.json / metrics.png / masks.npz）
```

`results/` には車載映像（KITTI raw）と一般動画（動物・夜間街路など）に対する解析結果を収録している。動画ファイル自体は容量の都合でリポジトリには含まれない。

## セットアップ

```bash
git clone https://github.com/mizinco/sam2-recognition-limit-observer.git
cd sam2-recognition-limit-observer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

SAM2 本体とチェックポイントは別途取得する:

```bash
git clone https://github.com/facebookresearch/sam2.git vendor/sam2
cd vendor/sam2 && pip install -e . && cd ../..
# チェックポイント (sam2.1_hiera_large.pt) は公式リポジトリの手順でダウンロードし checkpoints/ に置く
```

Apple Silicon (MPS) で動作確認済み。CUDA 環境でも動作する。

## 使い方

1. `data/` に解析したい動画クリップを置く
2. `notebooks/sam2_gradio.ipynb` を開き、UI 上で対象をクリック指定して追跡を実行
3. 生成された `masks.npz` を `src/analyze_case.py` で解析すると `metrics.json` と `metrics.png` が出力される
4. 複数ケースをまとめて解析する場合は `src/run_all.py`

## ライセンス

本リポジトリのコード: MIT License（[LICENSE](LICENSE) 参照）
SAM2 本体: 公式リポジトリのライセンスに従う（Apache 2.0）
入力動画・データセット: 各出典のライセンスに従う

## 引用

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2024}
}
```
