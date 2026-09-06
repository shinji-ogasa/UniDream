# Stage16: fixed sign/magnitude hindsight registration

Commit and push source, tests, configuration, this protocol, primary research
and data-only preflight before computing a new real quantile, component forecast,
loss or order. Stage15 is complete and rejected (report commit c49602d).
This experiment diagnoses the information sensitivity of the original Stage12
controllers. It does not choose Stage15 features or optimize model architecture.

## Scope and four cells

Use original reused validation5–12 (April2021–April2023), cutoff strictly before
2023-04-16T13:45Z. Original test(f)=validation(f+1); these are also old test4–11,
not independent confirmation. No additional-test15–24 modeling, labels, scores
or selection. The inherited Spot loader decodes its full parquet before slicing;
semantic exclusion is not a claim that later bytes were never decoded.

Bind the completed Stage12 source revision
`d3b25734a34915049a327256bd9f99cd9aea8336`, its original technical_half and
perp_delay0_half mean streams and the shared calibrated technical risk forecast.
Neither the best-looking old candidate nor the latest added features replace
these two parents. The previous 174 adaptive causal names remain unchanged.

For each parent's saved mean mu and saved realized h24 log return y:

| Cell | Value on original scored support O |
|---|---|
| base | mu |
| sign | sign(y) * abs(mu) |
| magnitude | sign(mu) * abs(y) |
| full | y, the saved Stage12 return-only Oracle |

Use np.sign(0)=0, without epsilon. For mu=0 both partial cells remain zero;
full may differ. For y=0 partial and full cells are zero. Actual non-return
columns and all unscored outcomes must be ignored by the new intervention.
All I\\O values retain their own learned mu, all ~I stay NaN. Risk and masks
remain unchanged. This is explicitly future information supplied on O only,
not a causal predictor, training teacher, conditional-mean truth or global bound.

Keep all28 Stage12 policies (12 causal controls,16 existing hindsight diagnostics,
including4 finite RL beams), append only8 new hindsight policies:2 streams ×
2 partial components ×2 missing rules. Total36×8=288 rows and576 base/stress
accounts. Do not rerun or retune the finite RL search. Its retained feasible
objectives are not globally optimal upper bounds on causal performance.

## What the decomposition can and cannot show

abs(mu) is the absolute conditional-mean estimate, NOT an estimate of E[abs(Y)].
sign(mu) is a mean's sign, not necessarily the most probable outcome sign.
A small mean can be appropriate when positive/negative returns cancel. The
magnitude cell reveals one future absolute endpoint return, not conditional
variance or realized intraperiod quadratic variation. Keep risk unchanged.

Sign replacement weakly improves scored squared error by construction. For
nonzero y the reduction is4*abs(mu*y) on opposite signs and0 otherwise;
y=0 removes mu². Full replacement has scored MSE0 mechanically. Magnitude
replacement can worsen error by expanding a wrong-sign mean to−y. These are
algebraic checks, not learned accuracy. Existing sign accuracy uses (>0), so
zero means must be separately counted; it is not three-class sign agreement.

Each cell rolls out its own inventory with nonlinear thresholds, deadband,
step limits, fees and borrowing. Correct signs with tiny magnitude may not
trigger useful orders, and an endpoint sign does not describe its intraperiod
path. A partial cell can beat full in this fixed controller. Do not claim that
adding information must improve AlphaEX or DD here.

Report all five fixed edges per stream:sign−base,magnitude−base,full−base,
full−sign,full−magnitude, plus the symmetric contrast full−sign−magnitude+base.
Apply these separately to each economic metric/cost and subgroup MSE. These
are scale-specific descriptive differences; they are not additive causal shares
or a percentage allocation of the Oracle gap. No winner selection follows.

## Fit-only magnitude threshold and descriptive score groups

Compute exactly8 fold thresholds AFTER freeze:np.quantile(abs(y_fit),0.9,
method='linear') on the original selected fit return rows. Both streams share
one fold threshold. These are descriptive distribution estimates, not model
fits; no E label selects the quantile/window. Preflight binds each fit vector,
fit mask and count without calculating a new quantile. Fit counts are
800/1034/1313/1500/1503/1634/1672/1794 with the original mature-label support.

Score each of8 means (2 bases,4 partial,2 old full) in3 groups:all O,
abs(y)>=fit_q90,abs(y)<fit_q90. Exactly24 records per fold,192 total. Equality belongs
to large; save equality/zero-threshold counts. Tail membership deliberately
uses future y and NEVER changes forecasting/ordering opportunities. It describes
errors and return energy, not attributed dollar PnL or an online tail detector.
A shifted volatility distribution need not place10% of E in this tail.

Empty groups have rows0 and null losses. Equal-quarter/regime means and paired
contrasts require every included quarter's relevant metric defined or return
null; never silently omit an empty or undefined-rank quarter. Save nonempty and
defined-rank counts. Separately label pooled-row MSE and its explicit denominator.
Save192 MSE/MAE/binary-sign/rank records,48 direction-group records (2 streams ×
3groups ×8), actual/forecast zeros, nonzero same/opposite sign counts, subgroup
squared-return share of all and opposite-sign return-energy share of the subgroup.
No MBB, p-value, confidence/probability or selective-significance claim.

## Unchanged execution and endpoints

I2586/O2574 leaves12 unscored learned decisions per stream. Preserve332 fallback
opportunities and2 missing-current-open scheduled decisions. Use original UTC6h
clock on a complete15m open grid; features through t−1; next bar's open fill;
y=log(close[t+24]/open[t+1]), maturity375minutes. Fit/S/I labels mature strictly
before segment end; E score labels no later than end. Current future-label
availability never gates inference or missing-input handling.

Initial B&H inventory, own cash/units, one-way fee0.00055, annual borrow0.10,
utility risk1/cost allowance2, target intents[.5,1.12], maxstep.08 and deadband.01.
Hold does not submit on unavailable means. Fallback submits target1 on known
current open with no prediction; it does not immediately reset actual exposure.
Missing current open cannot order; missing immediately-next open skips without
rollover. Continuous borrow across gaps. Replay identical base intents at twice
fee and borrowing; no second optimization. No label-driven window truncation.

Before accepting new results, reproduce64 endpoint paths:2 streams ×base/full ×
2 rules ×8. Their targets must exactly match Stage12;32 full decision traces
match the original trace fields under inherited numerical tolerance1e−12.
Full forecast O values equal y exactly; remaining mu/risk/support equal parent.
Replay all224 old control rows at both costs and compare original accounts.
These reproductions are not new candidates or independent confirmation.

## Provenance, outputs and validation

Verify25 source modules and all bound config/manifests. Stage12 prepare returns
1328 ancestors; also verify its completed400 artifacts and8fold manifests,
merging1728 distinct ancestors, rejecting conflicts/path aliases. Reconstruct
original fit values/masks through immutable parity preparation, not new sampling.
Preflight has no new quantile, forecast, loss, coefficient or order.

Save49 artifacts/fold:4 new forecast NPZ,36 target NPZ,8 new hindsight trace
JSON,1 fit-threshold JSON;392 total, plus8fold manifests/registration/results.
Every new trace explicitly overrides the inherited causal label and records
which decisions received future sign/magnitude. Save288 rows,192 scores,
48 direction diagnostics,8 thresholds and64 endpoint checks. All32 old full
trace bindings and all unchanged targets remain reachable in the ancestor chain.

A completed result refuses rerun. A terminal partial attempt may be retried by
full deterministic replay with exact existing-artifact comparisons; never restart
because an observation timed out. Recheck the same live handle. Report attempts.
No model refit, weight fit, production, live trading, paid compute or automation.

Run full `uv run python -m unittest discover -s tests -v` and `git diff --check`
before real diagnostics. Synthetic tests must cover zero/poison/mask invariants,
nonlinear factorial arithmetic, frozen config, fit-only thresholds, empty-tail
nulls, complete contrasts and paired regime/support rejection. Independent
source, substitution/threshold, own-state/accounts and score/summary audits follow.

## Decision boundary and sources

This diagnostic cannot promote a new causal model. Partial/full cells use the
future; original base cells remain causal controls without new independent confirmation. This study can identify which supplied component changes the
fixed controller and where its errors concentrate. Any later causal predictor
requires a new frozen protocol and chronological validation. Prior failures and
locks remain intact; 2bull/4bear/2sideways and repeatedly reused/overlapping
histories do not prove high-probability generalization or formal P1 success.

[Christoffersen–Diebold](https://pubsonline.informs.org/doi/10.1287/mnsc.1060.0520)
distinguish mean/sign/volatility dependence. [Anatolyev–Gospodinov](https://pages.nes.ru/sanatoly/Papers/Decomp.pdf)
model the joint sign/magnitude distribution; E[S*A] generally differs from
E[S]*E[A]. [Anatolyev–Gerko](https://pages.nes.ru/sanatoly/Papers/Profit.pdf)
distinguish sign agreement and magnitude-weighted trading outcomes. Their
markets, costs and inferential assumptions do not validate this BTC experiment.
See [primary research note](oracle_sign_magnitude_research_20260906.md).
