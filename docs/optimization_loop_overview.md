# Optimization Loop Overview

## 順番

最適化ループは次の順で回す。

1. `issue1 teacher audit`
2. `issue2 BC prior audit`
3. `issue3 AC support audit`
4. `issue4 WM regime audit`
5. `issue5 conservative AC`
6. `issue6 external source evaluation`

## 目的

- `issue1`: teacher が regime に応じて de-risk しているか
- `issue2`: BC が teacher を再現できているか
- `issue3`: AC が BC prior から離れすぎていないか
- `issue4`: WM latent が regime / transition を持てているか
- `issue5`: 今の実装ノブで保守的な AC に寄せられるか
- `issue6`: 外部ソースが補助要因として効くか

## 実行順の考え方

- まず診断を先に回す
- そこで主因っぽい issue を確認する
- その issue の候補アルゴリズムを `L1 -> L2` で試す
- 片付いたら次の issue に進む

## 主要 runner

- [run_issue2_bc_prior_loop.ps1](../scripts/run_issue2_bc_prior_loop.ps1)
- [run_issue3_ac_support_loop.ps1](../scripts/run_issue3_ac_support_loop.ps1)
- [run_issue4_wm_regime_loop.ps1](../scripts/run_issue4_wm_regime_loop.ps1)
- [run_issue5_conservative_ac_loop.ps1](../scripts/run_issue5_conservative_ac_loop.ps1)
- [run_issue6_external_source_loop.ps1](../scripts/run_issue6_external_source_loop.ps1)

## 全体 wrapper

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_optimization_loop.ps1 -CheckUv
```
