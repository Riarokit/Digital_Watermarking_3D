# 研究：3次元点群データへの電子透かし埋め込み

このリポジトリは、主に、3次元点群データに対する電子透かし技術の実装・評価を目的としています。

---

## 🔧 使用技術

- Python 3.10
- NumPy, Open3D, pandas, sklearn etc.
- 必要なライブラリはrequirements.txtに記載されています。pip install -r requirements.txt　でインストールできます。


---

## 📁 ディレクトリ構成

modulesフォルダの中身は、過去研究室で作成された点群データ処理に役立つライブラリ群です。

## 点群の対応付け

`DW2_exp.py` の `PROPOSED_CORRESPONDENCE_MODE` と各mainの
`correspondence_mode` で切り替えます。

| 設定値 | 方式 |
|---|---|
| `hk` | HK判定＋疎最小費用割当て（デフォルト。従来の1to1を置換） |
| `auction` | HK判定＋近似最小費用のオークション法 |
| `bidir` | 双方向近傍の候補グラフ＋疎最小費用割当て |
| `coordinate` | 最近傍の多対一対応 |
| `auto` / `index` | 従来の順序判定付き対応 / インデックス対応 |

旧設定値 `1to1` は廃止しました。`hk` に変更してください。
新しい3方式はクラスタ内で1対1ですが、クラスタ間での受信点再利用は許可します。
頂点数の減少時や候補内で割当てが成立しない場合は、従来どおり `coordinate` に戻ります。
El Zeinの `ELZEIN_CORRESPONDENCE_MODE` でも同じ方式を選択できます。

`PROPOSED_ONE_TO_ONE_CANDIDATES` は `hk` / `auction` の候補段階
（既定8,16,32,64,128）、`bidir` では最大値が受信候補プールの近傍数です。
`bidir` の辺は正逆方向それぞれ8→16→32近傍で試します。
`PROPOSED_ONE_TO_ONE_WORKERS` はクラスタ処理のスレッド数（0は自動）です。
Numbaを使用するため、`pip install -r requirements.txt` が必要です。
初回の `hk` / `auction` ではJITコンパイルが発生し、以後はキャッシュを利用します。

