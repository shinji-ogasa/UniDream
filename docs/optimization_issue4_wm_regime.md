# Optimization Loop: Issue 4 WM に regime 補助目的を追加すべきか

## 狙い
Issue 4 では、world model の latent が regime を十分に表現できていないかを局所診断する。

ここでは full 再学習はせず、既存 checkpoint の latent から

- `current_regime`
- `next_regime`

の線形識別性能を測る。

## 診断スクリプト

- [audit_wm_regime.py](../audit_wm_regime.py)
- [wm_regime_audit.py](../unidream/experiments/wm_regime_audit.py)

## 真偽確認

対象は `medium_v2_fix` の fold 4。
feature family は checkpoint に合わせて raw-only の config を使う。

- config: [medium_ext_sources_rawonly.yaml](../configs/medium_ext_sources_rawonly.yaml)
- 出力:
  - [WM regime summary](../checkpoints/medium_v2_fix/wm_regime_audit/medium_ext_sources_rawonly_wm_regime_audit_summary.csv)

実行は軽く切るため、`val` の末尾 `4096` bars に限定した。

## 結果

### fold 4 / val

| task | n_samples | accuracy | balanced_accuracy | macro_f1 |
| --- | ---: | ---: | ---: | ---: |
| `current_regime` | 4096 | 0.683 | 0.690 | 0.702 |
| `next_regime` | 4095 | 0.679 | 0.687 | 0.699 |

## 判定

Issue 4 の真偽判定は次の通り。

- **WM が regime をほとんど持てていない**: false
- **WM の regime 表現はそこそこある**: true

完全に十分とは言えないが、`balanced_accuracy ~0.69` と `macro_f1 ~0.70` が出ているので、
少なくとも「WM が regime transition を全く持てていない」ことは主因ではない。

## 次の遷移

Issue 4 もここで一段閉じる。
Issue 2 と Issue 3 の結果を合わせると、いま濃いのは

- teacher / BC 出力設計の collapse
- support を強く制限した actor 更新

側である。

次は `issue5: AWR/AWAC or IQL/CQL 系へ寄せる` に進み、

- `KL budget`
- `support budget`
- `conservative AC`

の 2〜3 本を小さく切る。
