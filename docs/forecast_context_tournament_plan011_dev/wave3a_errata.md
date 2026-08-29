# Wave3A frozen-result errata

Wave3A's original artifact is intentionally left unchanged. It is retained as a historical development screen only.

The original Wave3A comparison centered the dynamic path on B&H=1.0 while its constant comparator was selected independently on validation (for example, 0.5 or 1.12). Its reported timing increment therefore mixed temporal timing with an exposure-level difference. Its fold/gate robustness contract also predated the strict Wave3C checks.

Wave3C is the formal superseding screen: it uses the validation-selected constant exposure as the overlay baseline, fixed execution delay 1, a common right-side return window for dynamic/constant/lag/null paths, exact development folds `{0, 2, 8}`, and test results only for the report-only development tournament.

Frozen source artifact: `docs/forecast_tournament_plan011_dev/result.json`
Frozen source commit: `e0ab435ab6601ce49b4f6c28bdb15504d2c57315`
Corrected replay status: `complete`; rows=6; failures=0
Corrected replay median dynamic AlphaEx: `+18.9746pt`
Corrected replay median constant AlphaEx: `+20.1675pt`
Corrected replay median timing increment: `-1.1929pt`

The corrected replay is report-only and is not used to select a Wave3C candidate, threshold, horizon, or next-wave promotion.
