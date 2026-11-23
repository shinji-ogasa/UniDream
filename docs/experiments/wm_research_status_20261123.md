# 2026年11月の予測・WM・RL研究スナップショット

2026-11-23（JST）に、Macローカルの実装、設定、保存済み集計を確認した記録。
以下は11月9〜12日の研究結果の転記であり、今回の再学習・独立再計算ではない。
主系統の Plan011 v31 と、ここで扱う研究用 entrypoint は区別する。

## 実装と入口

研究用モジュールは `unidream/experiments/wm_*.py`、固定設定は
`configs/experiments/wm_*.yaml`、テストは `tests/test_wm_*.py` に保存する。
各実験の `--help` は、例えば次のように確認できる。

```bash
uv run python -m unidream.experiments.wm_technical_ablation --help
uv run python -m unidream.experiments.wm_model_comparison --help
uv run python -m unidream.experiments.wm_lstm_grid_benchmark --help
uv run python -m unidream.experiments.wm_seed_robustness_benchmark --help
uv run python -m unidream.experiments.wm_tail_transfer_benchmark --help
```

追加依存は次のファイルに固定されている。プロジェクトの `uv` 環境に追加して使う。

```bash
uv pip install -r configs/experiments/wm_model_comparison_requirements.txt
uv pip install -r configs/experiments/wm_volatility_classification_requirements.txt
```

比較実装には Ridge、Logistic Regression、LightGBM、EWMA、GARCH、
直接予測 Transformer、Transformer WM、LSTM がある。
分類・回帰・同時学習、特徴量 ablation、有意差検定、予測値とWM状態の
policyへの追加、保有率制約の解除、seed追加検証を別の実験として扱う。
主系統の学習条件やvalidation selectorを、この結果で変更していない。

## データ

11月の特徴量研究は BTCUSDT の15分足。分析期間は2020-09-01以上、
2026-11-01未満の216,192行で、warmup込み219,168行。
これは欠損を含む時刻グリッドの行数であり、全行が利用可能という意味ではない。

現物・無期限先物OHLCV、売買代金、取引件数、taker買い比率、basis、
mark/index/premium、funding、OI、long/short ratio等を取得している。
OHLCV5・市場14・テクニカル44を合わせたD条件は63チャネル。
過去の実際の公開時刻は不明で、source timestampに保守的な遅延を仮定している。
取得成功やarchiveの更新日時を、リアルタイム利用可能性の証拠としない。

詳細は [データ監査JSON](wm_research_evidence_20261123/data_summary.json)。

## 予測性能

[420行の比較表](wm_research_evidence_20261123/forecast_results_420.csv) は
4入力条件×5モデル×7タスク×3学習形式の集計。
線形モデル・LightGBMのjoint行には既存headの再利用があるため、
420個の独立した学習済みモデルとは解釈しない。
ニューラルモデルは3 seed、線形モデル・LightGBMは1 seed。
15四半期のhistorical Test、124,488個の時刻を対象とする探索的比較である。

D条件、4時間先予測の抜粋（回帰・分類は別の学習条件）:

| モデル | リターン回帰R² ↑ | 分散予測QLIKE ↓ | リターン3分類BA ↑ |
|---|---:|---:|---:|
| LightGBM | -0.051040 | 0.480507 | 43.6560% |
| Transformer WM | -0.006318 | 0.550399 | 41.9103% |
| 直接予測Transformer | -0.141664 | 0.555635 | 42.3717% |
| LSTM | -0.033207 | 0.541927 | 42.4474% |

回帰R²はゼロリターン予測を基準とし、D条件では全モデルが基準未達。
LightGBMの回帰R²はA〜Dのすべてで負だった。
WMにはC条件で小幅な正の値（約0.000971）もあり、全条件未達とは表現しない。

3分類は下落・横ばい・上昇。BAはクラスごとの再現率の平均であり、
横ばい検出による改善を含む。売買方向の的中率や収益率とは異なる。
この表の点推定だけでモデル間の統計的優位性を主張しない。

先行する11月9日の比較結果と検定も
[予測比較](wm_research_evidence_20261123/model_comparison.csv)・
[検定](wm_research_evidence_20261123/model_comparison_inference.csv) に保存した。
その設定ではWMのリターン予測誤差はLightGBMより小さい一方、
ゼロ予測への改善は有意でなく、分散予測ではLightGBMが優れていた。
異なる設定の結果を同じ実験の更新値として混ぜない。

## 予測値・WM状態から保有率を決める実験

`wm_policy_ablation` は以下を比較する。

| arm | 入力 |
|---|---|
| A | 市場特徴X＋保有状態 |
| B | X＋予測値F＋保有状態 |
| C | X＋凍結WM内部状態Z＋保有状態 |
| D | X＋F＋Z＋保有状態 |

FはLightGBM・直接予測Transformer・LSTM・WMの57 head、121次元。
LightGBMのheadは21個あり、LightGBM単独の追加効果を分離した実験ではない。
予測headは既に観測したforecast Testの結果から選んだため、
独立未使用holdoutの評価とは扱わない。
PPOは過去の実市場Train区間で学習し、主系統v31のimagination ACとは異なる。

## 保有率制約の段階解除と10 seed検証

BTCUSDT、2025年4月〜2026年9月。売買費用込みの区間連結評価。
RLはseedごとの最終リターン・最大DDの平均、ルール戦略は固定実行。
最大DDは下落幅の正値で示す。

| 条件 | 最終リターン | 最大DD |
|---|---:|---:|
| WMあり・段階解除 | +9.4092% | 15.9390% |
| WMなし・段階解除 | +6.1376% | 12.8030% |
| WMあり・制約維持 | +2.3658% | 35.4096% |
| 20日MA | +9.1343% | 34.7845% |
| 年率10%ボラターゲット | +6.3282% | 16.3186% |
| Buy & Hold | -5.3566% | 52.9554% |

元データ:
[seed別結果](wm_research_evidence_20261123/policy_per_seed.csv)、
[seed集計](wm_research_evidence_20261123/policy_seed_summary.csv)、
[ルール戦略](wm_research_evidence_20261123/policy_baselines.csv)、
[主要検定](wm_research_evidence_20261123/policy_primary_inference.csv)。

WMあり対WMなしの主要検定のfamily内Holm補正後p値は0.456754。
より広い14比較全体での補正後p値は1.0。
WMの収益追加効果は確認できていない。
seed追加は同じ市場期間の再学習であり、独立した市場サンプルの追加ではない。

評価条件の決定には既知Testを参照している。予定52,608本中50,038本を評価し、
欠落区間の価格変動を飛ばす。8回の区間リセットで保有率を100%に無償で戻すため、
18か月の連続運用収益とは表現しない。

## 未完了の研究と保存範囲

`wm_tail_transfer_20261112` は追加市場情報、価格帯別の板集計、疎な清算サンプル、
tail予測と保有率制御への転用を扱う。完全な板イベント履歴や注文執行最適化ではない。
2026-11-23の確認時点では対応する学習プロセスと最終評価出力がなく、
最終status更新は2026-11-14。未完了の研究としてコード・設定・テストのみ保存する。
多資産共分散を使ったポートフォリオ最適化の完成実装も今回確認していない。

`codex_outputs/` の市場データ、個別予測、checkpoint、ログはローカル保存のまま。
本書で使った軽量な集計だけを `docs/experiments/wm_research_evidence_20261123/` に転記した。
元path・SHA-256・byte数は [manifest](wm_research_evidence_20261123/manifest.json) に記録する。
この集計だけでは全モデルの学習・推論をreplayできない。
新環境での再実行には設定に対応するデータ取得・前段実験の生成物が必要であり、
既存の登録済みローカル出力を上書きして再学習しない。

## 保存時の検証

2026-11-23に `uv run python -m unittest discover -s tests -v` を実行し、
1,280 tests が109.413秒で成功した。`git diff --check`、
転記した8ファイルの元データ・byte数・SHA-256一致、追加案内のローカルリンクも確認した。
これは実装の検証であり、予測精度・収益性の再実験ではない。
