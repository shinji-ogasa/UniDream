# Alpha/DD 2026-09-05 evidence snapshot

このディレクトリは、BTCUSDT の alpha/DD 実験、事前固定した cross-asset 比較（ETHUSDT/BNBUSDT）、および完了済み ML25 の結果を、後から再照合できる小さな追跡可能スナップショットとして保存する。元の `codex_outputs/` JSON は変更していない。

## 収録範囲

- `btc/`、`eth/`、`bnb/`、`ml/` に、各 run の registration、selection lock、qualification（存在する場合）、historical/fresh の JSON を保存した。これらは元ファイルのバイト列をそのままコピーしている。
- development は大きいため、`development_compact.json` に元ファイルの SHA/サイズ、選択候補の全 fold 行、全候補の登録値・集約値・欠測理由を保存した。モデルファイルや `development_progress.json` は含めない。
- `cross_asset_lock.json` は confirmation 前に凍結された asset 選択の原本コピーである。ETH の confirmation は実施していないため、`eth/confirmation_status.json` にその状態を明記した。
- `manifest.json` は元ファイルと evidence ファイルの対応、SHA-256、データ provenance、取得範囲・最新 archive identity を列挙する。

## 数値とデータの意味

JSON の AlphaEx と ΔMaxDD は比率（fraction）で、レポートでは 100 倍して percentage points（pt）表示する。ΔMaxDD は負値が B&H より drawdown 改善を示す。historical/fresh の JSON には holdout と選択候補の全 metric rows および provenance を含めた。

データは公式 Spot monthly klines の 15m bar OPEN 時刻を使い、2018-01 から 2026-08 を要求した。全 asset で 2026-07 まで 103/104 月を取得し、2026-08 は公式 archive の HTTP 404 による `unavailable_tail` として明示した。historical gap は埋めず、latest archive の checksum は ledger に記録された値を使う。archive の published/collector/exchange timestamp は live market observation ではなく、`live_causal_eligible=false` である。

この evidence は raw parquet、raw ZIP、NPZ、joblib、checkpoint を追跡しない。結果は descriptive であり、fold の再利用を含むため selection-adjusted significance ではない。通常の unit/evidence 検証を金融上の妥当性、live causal archive、または HF deploy の証明と解釈してはならない。
