# Computer-Vision-Study 

## 1. 動作環境

* **Hardware:** Apple M4 Chip
* **OS:** macOS Tahoe
* **Package Manager:** Miniforge (conda-forge)
* **Primary Libraries:** Python 3.11, OpenCV

## 2. セットアップ手順

### Step 1: Miniforge のインストール

Homebrew経由、または公式サイトのインストーラーを用いて Miniforge をインストールします。

```bash
brew install miniforge

```

### Step 2: 仮想環境の作成と起動

画像認識専用の環境を作成します。安定性を考慮し Python 3.11 を選択しています。

```bash
# 環境の作成
conda create -n vision-lab python=3.11

# 環境の有効化
conda activate vision-lab

```

### Step 3: ライブラリのインストール

OpenCV をインストールします。conda-forge チャンネルを使用することで、Apple Silicon に最適化されたバイナリが取得されます。

```bash
conda install opencv -c conda-forge

```

## 3. 動作確認

環境が正しく構築されたか確認するためのスクリプトです。

```python
import cv2
import sys

print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")

# カメラの起動テスト
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found")
else:
    print("Camera is ready")
    cap.release()

```

## 4. 技術選定の背景 (Why Miniforge?)

当初は標準の `venv` を検討しましたが、以下の理由により `Miniforge (conda)` へ方針転換しました。

* **SSL/TLS問題の回避**: OS標準のPythonとHomebrew経由のライブラリ（OpenSSL等）の競合を避けるため、自己完結型の環境を選択。
* **依存関係の分離**: OpenCVが必要とする GUIライブラリ（Qt等）を、システム側の管理と完全に切り離して管理。
* **Apple Silicon 最適化**: `conda-forge` により、M4チップの GPU/Neural Engine を活用しやすいバイナリを優先的に利用。

---

## Tips: よく使うコマンド

* 環境の停止: `conda deactivate`
* パッケージの確認: `conda list`

