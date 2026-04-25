# bcplan_3 true routing BC 検証結果

作成日: 2026-04-25
対象: fold4 / BTCUSDT 15m / seed 7

## 結論

`bcplan_3.md` のレビュー方針は妥当。position label を直接真似るより、transition advantage を `neutral / de_risk / recovery / overweight` の route label に落として、no-trade margin を入れる設計の方が筋がいい。

ただし今回のfold4実験では、ACへ進む条件は満たさなかった。

理由は明確。

- route label自体は active 33-51% で作れている。
- test時の predicted route は neutral に寄りやすく、recovery route はほぼ 0%。
- activeを強めると turnover は許容内でも MaxDDΔ と AlphaEx が悪化する。
- 安定寄りにすると AlphaEx/MaxDDは良いが flat 100% 扱いになる。

つまり、実装経路は通ったが、現状の `z/h/regime` だけでは route を十分に分離できていない可能性が高い。ACはまだ早い。

## Web調査からの判断

今回参照した一次資料:

- IQL: https://arxiv.org/abs/2110.06169
- AWAC: https://arxiv.org/abs/2006.09359
- TD3+BC: https://arxiv.org/abs/2106.06860
- CQL: https://arxiv.org/abs/2006.04779
- Transaction-cost no-trade region: https://pubsonline.informs.org/doi/abs/10.1287/moor.15.4.676

設計に反映した点:

- IQL は policy extraction を advantage-weighted BC として扱うので、transition advantage をBC信号に使う方向は妥当。
- AWAC は offline/prior data から actor update を advantage で重み付ける流れなので、route soft target + advantage weight と相性がいい。
- TD3+BC は actor update にBC項を残して分布外逸脱を抑えるので、ACへ進むならBC priorを強く残す必要がある。
- CQL は distribution shift によるQ過大評価を抑える思想なので、壊れたBCをACで救う用途ではなく、安定BC後の保守的fine-tune向き。
- 取引コスト付きポートフォリオでは no-trade region が自然に出るため、`best_advantage - neutral_advantage < margin` を neutral に落とす設計は妥当。

## 実装内容

追加したもの:

- `transition_advantage.py`
  - fine-grained transition class を 4 route に集約。
  - route score = `max(adv_vs_neutral, adv_vs_current, 0)`。
  - margin未満の非neutral routeを落として no-trade化。
  - `softmax(route_adv / tau)` + label smoothing の soft route target を生成。

- `Actor`
  - `route_head`: `neutral / de_risk / recovery / overweight`。
  - `route_delta_head`: routeごとの小さい overlay adjustment。
  - route controllerでは target position を直接予測せず、現在inventoryから route別 small step target を混合。

- `BCPretrainer`
  - `route_target_coef` による route CE / soft CE。
  - `route_advantage_weight_coef` による positive route advantage weighting。
  - chunk training / per-step training / path dataset の route label受け渡し。

- `fold_inputs.py` / `train.py` / `bc_stage.py`
  - transition route labels を fold input からBCへ渡す経路を追加。

- `test_stage.py`
  - test時の predicted route distribution を出力。

## 実装・データ経路チェック

実行確認:

```powershell
uv run python -m py_compile unidream\experiments\transition_advantage.py unidream\actor_critic\actor.py unidream\actor_critic\bc_pretrain.py unidream\experiments\bc_setup.py unidream\experiments\bc_stage.py unidream\experiments\fold_inputs.py unidream\cli\train.py unidream\experiments\test_stage.py
```

結果: pass

```powershell
git diff --check
```

結果: 実質pass。LF/CRLF警告のみ。

データ経路確認:

- train側で `Transition route labels: ...` が出力される。
- BC学習が route loss込みで完走する。
- test側で `Route dist: ...` が出力される。
- したがって route label は `fold_inputs -> train.py -> bc_stage -> BCPretrainer -> Actor.route_head` まで渡っている。

## 実験結果

比較基準:

- 以前の best: `bc_transition_relabel_smooth_m0005_ac500_s007`
- AC500結果: AlphaEx `+3.57 pt/yr`, SharpeΔ `+0.082`, MaxDDΔ `-0.28 pt`, long 6% / flat 94%, turnover `8.42`
- 問題: turnoverが高く、bcplan_3の `turnover <= 4` 目標から外れる。

| 実験 | config | 内容 | AlphaEx | SharpeΔ | MaxDDΔ | Test dist | Turnover | Route dist | 判定 |
|---|---|---|---:|---:|---:|---|---:|---|---|
| K | `bc_true_routing_k_s007` | direct relabelなし / route labelのみ | -0.59pt | +0.048 | +1.30pt | long 16% / flat 84% | 2.14 | neutral 94% / ow 6% | activeは出るがDD悪化 |
| L | `bc_true_routing_l_notrade_s007` | margin強め / step 0.05 | -0.10pt | +0.014 | +0.23pt | flat 100% | 0.75 | neutral 100% | flat collapse |
| M | `bc_true_routing_m_softsmooth_s007` | soft route + smoothed weak position補助 | +0.17pt | +0.017 | -0.27pt | flat 100% | 3.26 | neutral 97% / de_risk 3% | 安定だがactive不足 |
| M2 | `bc_true_routing_m2_active_s007` | route loss強化 / step 0.15 | -0.04pt | +0.069 | +0.25pt | long 6% / flat 94% | 5.89 | neutral 95% / ow 4% | 最も妥協に近いがMaxDD超過 |
| M3 | `bc_true_routing_m3_balanced_s007` | MとM2の中間 | -0.11pt | +0.083 | +0.41pt | long 7% / flat 93% | 4.70 | neutral 91% / ow 7% | active化でDD悪化 |

## AC移行判定

bcplan_3の推奨ライン:

- AlphaEx >= +1.0 pt/yr
- SharpeΔ >= +0.02
- MaxDDΔ <= 0
- Turnover <= 4
- Flat 80-92%
- Active 8-20%
- collapse guard pass
- route diagnostics OK

今回このラインを満たすものはなし。

妥協ライン:

- AlphaEx >= 0 近辺
- SharpeΔ >= -0.01
- MaxDDΔ <= +0.2pt
- Turnover <= 6
- Flat <= 95%
- Active >= 3%
- collapse guard pass
- top-decile route advantage positive

最も近いのはM2だが、`AlphaEx -0.04pt` と `MaxDDΔ +0.25pt` で未達。Mは `AlphaEx +0.17pt`, `MaxDDΔ -0.27pt`, `turnover 3.26` と良いが、flat 100% / active不足で未達。

したがってACは実行しない。壊れてはいないが、BC priorとしてまだ弱い。

## 分かったこと

1. true routing BCはhigh-turnover collapseを抑える。

K/L/M/M2/M3はすべて turnover `0.75-5.89` に収まった。以前のdirect relabel系の `turnover 665` とは別物。

2. no-trade marginは効きすぎると即flat化する。

Lはtrain labelのactiveを33%まで落とし、test routeはneutral 100%。これは安定ではなく不活動。

3. weak smoothed position補助は必要。

Kはdirect relabelなしで動くが、testでoverweight寄りになってMaxDDが悪化した。Mのように弱いposition補助を残した方がリスクは良い。

4. route recoveryが学習されていない。

train labelには recovery が約1%しかなく、test predicted routeは全実験で recovery 0%。BCの戻り行動問題はまだ残っている。

5. 現状は「loss以前にroute分離が弱い」可能性が高い。

train route labelは active 45-51% でも、test predicted routeは neutral 91-100%。route headがrouteを十分に一般化できていない。TransformerWM predictive stateを直接positionに渡すのではなく、bcplan_3通り gate-limited route feature として使う次段が妥当。

## 採用・不採用

採用:

- true routing BCの実装。
- route label生成とdiagnostics。
- M / M2を次の比較基準として残す。

不採用:

- Lの強いno-trade設定。flat 100%になる。
- Kのroute-only設定。activeは出るがMaxDDが悪い。
- 現時点のAC移行。

暫定best:

- 安定性重視: `configs/bc_true_routing_m_softsmooth_s007.yaml`
- active条件に近い: `configs/bc_true_routing_m2_active_s007.yaml`

## 次の最短改善案

次はACではなく、route headの予測性能を直接測るべき。

実験案:

1. route classifier probe
   - train/valで route CE, accuracy, macro-F1, active recall, overweight recall, de_risk recall, recovery recall を出す。
   - route label自体が予測不能ならBC/AC以前の特徴問題。

2. predictive gate routing
   - predictive stateをposition headへ直結しない。
   - return予測は overweight gate、risk予測は de_risk gate、inventory age/current inventoryは recovery gate に限定。

3. recovery oversampling
   - recovery labelが1%なので、そのままでは学習されない。
   - recovery routeだけ class weight / sampler / synthetic current inventory rollout を入れる。

4. M2の軽い再調整
   - 目標は `AlphaEx >= 0`, `MaxDDΔ <= +0.2`, `flat <= 95`, `turnover <= 6`。
   - ただしこれを満たしてもACは500 stepsだけ。TD3+BC/CQL-liteの強い改善は期待しすぎない。

## 実行ログ

- `documents/logs/20260425_bc_true_routing_k_fold4.log`
- `documents/logs/20260425_bc_true_routing_l_notrade_fold4.log`
- `documents/logs/20260425_bc_true_routing_m_softsmooth_fold4.log`
- `documents/logs/20260425_bc_true_routing_m2_active_fold4.log`
- `documents/logs/20260425_bc_true_routing_m3_balanced_fold4.log`

## 最終判断

レビューの方向性は正しい。

ただし今回の結果では、`true routing BC + no-trade + soft route target` は turnover collapse を抑えた一方で、route予測がneutral寄りに潰れやすい。ACに進むにはまだ弱い。

次はACではなく、route予測精度の診断と predictive gate routing をやる。ここを通さずにACへ進むと、BCのneutral/overweight偏りをcriticで増幅するリスクが高い。
