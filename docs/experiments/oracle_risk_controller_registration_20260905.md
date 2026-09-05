# Risk controller v1: adaptive exploratory registration

This family is registered after observing oracle-feature-frontier-v1 and before
observing any risk-controller result. It is an adaptive development experiment,
not fresh evidence. No old locked test window is evaluated.

Motivation: technical HGB's 6h future volatility predictions improve mean MSE
skill against training climatology in all three start-of-quarter trend groups,
while return MSE remains weak. Test whether the improved risk forecast helps
allocation without replacing the directional backbone or increasing model size.

Fixed 12-candidate family: two directional backbones (the frozen robust overlay,
and technical Ridge h672 downside mapper) x two previously fitted HGB h24 risk
forecasts (base16 / technical) x strengths {.25,.5,1}. No models are refitted.

At a scheduled causal decision, ratio = clip(trailing vol7 annualized *
sqrt(24/35040) / max(predicted future RMS volatility, .001), .5, 1.5).
New target = clip(backbone_target * ratio**strength, .5, 1.12).
When no risk forecast or valid trailing vol is available, use the original
backbone target; absent backbone intent remains absent. This is a declared
causal fallback, not fabricated feature input. Costs, max-step, drift, delay,
borrow, 6h cadence, validation quarters, and regime definitions are unchanged.

Source results, forecasts, targets and registration are bound by SHA256 before
evaluation. All 12 candidates and both unchanged backbones are retained.
Ranking uses the previously fixed worst-regime score across base/2x costs;
no unseen-data or high-probability claim is available from reused validation.
