# OwlViT 物体検出アプリ

OwlViTモデルを使用した、テキストプロンプトベースの物体検出Streamlitアプリケーションです。

## 機能

- 画像のアップロードと物体検出
- カンマ区切りで複数のプロンプトに対応
- プロンプトごとに異なる色でバウンディングボックスを描画
- 検出閾値、線の太さ、文字サイズの調整可能
- CUDA、MPS、CPUの自動デバイス選択

## 必要要件

- Python 3.10以上
- uv（Pythonパッケージマネージャー）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/ikomiki/owl-vit-streamlit.git
cd owl-vit-streamlit
```

### 2. 依存関係のインストール

このプロジェクトは`uv`を使用して依存関係を管理しています。

```bash
# uvがインストールされていない場合
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 3. アプリケーションの起動

```bash
uv run streamlit run main.py
```

ブラウザが自動的に開き、アプリケーションが表示されます（通常は `http://localhost:8501`）。

## 使い方

1. **画像のアップロード**: 「画像を選択してください...」ボタンから、PNG、JPG、JPEG形式の画像をアップロードします。

2. **プロンプトの入力**: テキストボックスに検出したい物体を入力します。
   - 単一プロンプト: `a dog`
   - 複数プロンプト: `a dog, a cat, a car`（カンマ区切り）

3. **設定の調整**（サイドバー）:
   - **検出閾値**: 検出の感度を調整（0.0〜1.0）
   - **線の太さ**: バウンディングボックスの線の太さ（1〜20）
   - **文字サイズ**: ラベルの文字サイズ（8〜128）

4. **推論実行**: 「推論実行」ボタンをクリックして物体検出を開始します。

5. **結果の確認**:
   - バウンディングボックスが描画された画像
   - 検出結果一覧（ラベル、スコア、座標、色）

## プロジェクト構成

```
owl-vit-streamlit/
├── main.py              # メインアプリケーション
├── utils.py             # バウンディングボックス描画ユーティリティ
├── pyproject.toml       # プロジェクト設定と依存関係
├── test_inference.py    # 推論テストスクリプト
└── README.md            # このファイル
```

## 技術スタック

- **Streamlit**: Webアプリケーションフレームワーク
- **Transformers**: OwlViTモデルの読み込みと推論
- **PyTorch**: 深層学習フレームワーク
- **Pillow**: 画像処理

## デバイスサポート

アプリケーションは利用可能なデバイスを自動的に検出します：

1. CUDA（NVIDIA GPU）
2. MPS（Apple Silicon）
3. CPU（フォールバック）

使用中のデバイスはサイドバーに表示されます。

## 開発について

このプロジェクトは、主に **Google Antigravity** と **Gemini** を使用して作成されました。AIアシスタントによるコード生成、デバッグ、ドキュメント作成により、効率的な開発を実現しています。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 謝辞

- [OwlViT](https://huggingface.co/google/owlvit-base-patch32) - Google Research
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
