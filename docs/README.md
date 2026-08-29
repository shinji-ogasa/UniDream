# UniDream 実験結果・再現性ドキュメント

このディレクトリは、実験結果と「その結果をどう読むか」を残す場所です。学習途中の checkpoint は容量と誤利用を避けるため Git 管理せず、再現性を必要とする run の間だけ生成します。

## 現行の再現性ステータス

| 項目 | 現行仕様 |
|---|---|
| 学習 entrypoint | `uv run python -m unidream.cli.train` |
| pipeline | Transformer WM → BC → Imagination AC → validation selector → test report |
| デフォルト | `configs/trading.yaml`, seed `7`, device `auto` |
| データ | BTCUSDT / 15m / 現行 v3 feature-return cache |
| cache 契約 | include_*、feature columns、index、有限値を検証し、metadata sidecar を保存 |
| checkpoint 契約 | atomic save、WM/BC/AC metadata、Actor の全推論 runtime + selector 設定、semantic hash |
| 永続化方針 | 結果は本ドキュメント、checkpoint は run 中のみ。配信用 Space bundle は別管理 |

## 読む順番

1. [SPEC.md](../SPEC.md) — 学習・評価・リーク防止の固定契約
2. [plan011_v31_investor_evidence.md](plan011_v31_investor_evidence.md) — 開発 WFO と holdout の総合 evidence
3. [plan011_v31_holdout_2024_2026.md](plan011_v31_holdout_2024_2026.md) — 2024–2026 holdout の結果
4. [policy_family_holdout_comparison.md](policy_family_holdout_comparison.md) — policy family 比較
5. [risk_aware_rl_unidream_holdout_2026.md](risk_aware_rl_unidream_holdout_2026.md) — 別系統の risk-aware RL probe
6. [project_map.md](project_map.md) — research / Space / web の責務分担

図表は [fold0–12](figures/plan011_v31_folds0_12/README.md) と [holdout folds15–23](figures/plan011_v31_holdout_folds15_23/README.md) にまとまっています。図表の `checkpoint` 表記は、その時点の実験で使った生成物を示す履歴情報です。現在はローカル checkpoint を削除済みなので、再生成する場合は下記の学習コマンドを先に実行してください。

## 再現コマンド

開発 WFO（fold 0–12）:

```bash
uv run python -m unidream.cli.train
```

holdout（fold 15–23）:

```bash
uv run python -m unidream.cli.train \
  --config configs/plan011_overlay_actor_v31_holdout.yaml \
  --seed 7 \
  --device cuda
```

各 run は `logging.checkpoint_dir` に次を生成します。

- `resolved_config.yaml`: cost profile 解決後の config 全体
- `run_manifest.json`: run ID、config/source/data hash、Git、seed、環境、cache 契約、必須 artifact
- `fold_<n>/world_model.pt`: WM state と provenance
- `fold_<n>/bc_actor.pt`: BC actor と provenance
- `fold_<n>/ac.pt`: AC state、Actor の全推論 runtime snapshot、validation selector の最終 `adjust_rate_scale` / `advantage_level` と provenance

`run_manifest.json` が `status: completed` で、各必須 artifact の metadata が manifest と一致し、読み込み側の config hash も一致するときだけ replay CLI は checkpoint を読み込みます。不一致や旧形式の artifact は「条件不明」として拒否します。

## 結果の読み方

- `AlphaEx`: strategy の最終 total return − B&H の最終 total return。年率換算ではありません。
- `MaxDDDelta`: strategy の絶対 MaxDD − B&H の絶対 MaxDD。負値が drawdown 改善です。
- validation は selector / checkpoint 選択に使い、test は選択後の report-only です。
- 過去の数値は結果ドキュメントに保存しますが、現在の v3 cache と新 checkpoint で再学習した結果とは別の run として扱います。
