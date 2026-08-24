# Project map

This workspace has three separate repositories. They share the Plan011 v31
bundle contract, but they have different responsibilities.

```text
UniDream (research)
  data/oracle -> world_model -> bc -> ac -> selector/test -> bundle export
       |                                               |
       +---------------- experiment evidence ----------+
                                                       v
unidream-space (HF inference)
  bundle contract -> feature pipeline -> Plan011 predictor -> FastAPI /predict
                                                       |
                                                       v
unidream-demo-web (live demo)
  Binance candles -> Supabase Edge inference job -> paper-trading tables
                                              -> Next.js dashboard + Realtime
```

## Responsibility boundaries

- `UniDream/unidream/data`: data download, feature construction, dataset and
  teacher generation. It does not own model selection or deployment.
- `UniDream/unidream/world_model` and `actor_critic`: model definitions and
  training primitives. They do not own CLI orchestration or report formatting.
- `UniDream/unidream/experiments`: stage orchestration and run artifacts. The
  current training entry point is `experiments/fold_training.py`; the CLI is a
  thin argument parser in `cli/train.py`.
- `UniDream/unidream/eval`: reusable evaluation primitives. Policy statistics
  and selector decisions live in `eval/policy_stats.py` and `eval/selector.py`.
- `unidream-space/backend/predictors/plan011.py`: the only supported inference
  predictor. `backend/runtime.py` is only a bundle-type factory.
- `unidream-demo-web/supabase/functions/run-unidream-inference`: the Edge
  Function is an orchestration layer. Binance access, model calls, and pure
  paper-fill logic are separate modules.
- `unidream-demo-web/src/hooks/useLiveDashboard.ts`: browser Realtime state;
  `src/lib/server/dashboardRepository.ts`: initial server-side snapshot load.

## Current contract

- Symbol: `BTCUSDT`
- Timeframe: `15m`
- Current inference bundle: `plan011_v31_overlay_actor`
- Feature count: `17`
- Sequence length: `64`
- API input: candle arrays; legacy `returns` and `history_returns` inputs are
  intentionally unsupported.

## Historical material

Experiment reports, fold figures, and Plan013 probe documents remain under
`docs/`. They are evidence and provenance, not runtime entry points. Superseded
Space verification material is kept under `unidream-space/docs/legacy/`.
