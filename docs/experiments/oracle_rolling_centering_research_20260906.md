# Stage14 research design: fixed rolling mean and forecast centering

Planning note, 2026-09-06. Not a registration, model run, coefficient estimate, policy selection, or independent confirmation. Only implementation/registration source and primary literature were read; no new real coefficients, forecasts, losses, additional-test15–24 data or later market outcomes were computed/read.

## Recommended bounded comparison

Retain the proposed six new names: technical/perpetual rolling-centered means × hold/fallback, plus one shared rolling-return anchor × hold/fallback. Preserve all16 existing policies, both cost settings, risk forecasts, step/deadband/execution accounting, original8 development quarters and their2bull/4bear/2side inventory. Freeze each stream's existing stage13 scale weight w_S; do not update slopes or try another window. No scientifically necessary seventh variant is required to test whether this **joint historical intercept update** improves the frozen procedure. The rolling anchor is necessary: without it, better results cannot distinguish use of the historical return mean from the centered Ridge component.

For origin t and each fold's already frozen Ridge forecast stream m_s, define a history by origin time:

    H_t = {s: UTC six-hour scheduled origin,
           t - DateOffset(months=3) <= s < t,
           s + 375 minutes <= t,
           original causal inference support held at s,
           both frozen raw streams were available at s,
           their paired return label is now observed and finite}.
    a_t = mean_{s in H_t}(y_s)
    mbar_{g,t} = mean_{s in H_t}(m_{g,s})
    mu_{g,t} = a_t + w_{g,S} * (m_{g,t} - mbar_{g,t})
    rolling_anchor_t = a_t.

Use precisely the same H_t for both means, both streams and the anchor; assert this in preflight. A three-calendar-month origin window is not the last90days, lastN usable rows, or a maturity-time window. Define UTC DateOffset and left inclusion explicitly, including month-end behavior. Missing days do not expand the window. The3months choice inherits the earlier calibration time scale and bounds this hypothesis; no cited paper establishes it as optimal for BTC or this horizon.

Fix a history minimum before outcomes (64 preserves the previous calibration sufficiency convention, not a statistical guarantee). Prefer a data-only preflight that requires the minimum at every existing inference origin and stops without a performance run if unmet. Do not silently discard new forecasts, compress time, widen the window, or use future score support to meet it. If operational fallback for insufficient history is desired instead, its exact frozen rule and ledger must be registered first. Never exclude a live inference/order merely because its future return is unavailable for scoring.

## Timing and source hazards

The existing target is y_s = log(close[s+24] / open[s+1]) on15m bars. The close of bar s+24 becomes available at s+25, hence375minutes. At a six-hour decision t the immediately preceding scheduled origin t−6h has maturity t+15min and is excluded; with no gaps, the latest usable origin is t−12h. The registered archival availability convention includes equality: maturity <= t. It permits use at the nominal completion timestamp but does not establish actual feed receipt or processing latency. At the current UTC six-hour schedule maturities occur at minute15, so equality cannot occur; generic synthetic equality cases should nevertheless verify the inclusive rule. The cited literature does not choose this archive-time convention. Historical archive completeness is not historical receipt evidence.

Process each clock event in order: admit only prior labels whose maturity is at or before the event under the registered archival convention, expire origins left of the fixed window, compute means and issue/store the current forecast and action from that state, and enqueue its pending label. The current forecast's outcome cannot enter its own update. Outcome finiteness may determine admission once its maturity has passed; the eventual evaluation score mask must never be consulted while forming earlier states. Poison tests should alter all not-yet-mature labels and leave every prior mean, mask and action unchanged.

Existing calibration `actual` arrays deliberately mask the tail of S/I at each segment boundary; these tails may mature during the following segment. Permanent reuse of `interval_mask`, `scale_mask`, or evaluation `score_support` as update-history support therefore introduces artificial boundary deletions. Use separately verified origin-time inference support and as-of-t label maturity. Saved raw calibration mu is generated on the causal prediction support independently of calibration label masks; source label reconstruction is needed only where the saved label array was intentionally blanked. Bind the exact source files and compare reconstruction on already populated labels. Do not refit or use a later fold's model to fill a prior fold's forecast history.

Each fold retains its own Ridge fit ending before S and its own frozen w_S. Its S/I/E raw forecasts must all come from that same fold model; do not concatenate overlapping fold forecasts with different models or count one origin twice. w_S is available only after S ends. Rolling I diagnostics can be sequential from I start; S scores using its completed w_S remain retrospective. E is a sequentially updated diagnostic: already matured earlier E outcomes can lawfully affect later E forecasts, so it must not be described as a quarter with no evaluation-label updates.

## What this comparison can identify

With w fixed, mu_t = w*m_t + b_t, where b_t = a_t − w*mbar_t. Thus it changes a time-varying intercept while freezing the Ridge slope and architecture. Comparing with stage13 reliability estimates the entire registered intercept-update procedure's observed effect, not separate causal effects of return-mean adaptation versus forecast-centroid adaptation. Those two mechanisms are not individually identified by these six names; a further factorial family is unnecessary unless that separate attribution becomes the objective.

Unlike static positive affine shrinkage, time-varying b_t can change whole-quarter ranks. w=0 is exactly the **time-varying** rolling anchor, not a constant forecast; its IC need not be undefined. The update incorporates historical state beyond the current raw scalar forecast, so the literature's statement that fixed recalibration adds no information cannot be copied without qualifying the information set.

Report equal-quarter MSE/MAE against zero, saved fit mean, saved scale anchor, own full/half/reliability and rolling anchor; preserve all quarters, masks and adverse differences. Pair each new policy with its own missing-rule controls for AlphaEx/MaxDDDelta, turnover and costs; keep forecast and economic improvement separate. Record history counts, origin age, exclusions and temporal state hashes. Three months of paired support may omit crisis intervals and span several local regimes. A regime at E's first decision is not the regime of every history/update and cannot condition earlier S/I forecasts. Reused quarters, serially dependent losses, overlapping rolling histories/calibration windows and2/4/2 regime counts remain exploratory evidence, not high-probability trend invariance or an IID significance sample.

## Primary sources read and exact applicability

1. A. P. Dawid (1984), *Statistical Theory: The Prequential Approach*, JRSS A147. [Publisher](https://academic.oup.com/jrsssa/article/147/2/278/7106293), [original paper PDF](https://people.csail.mit.edu/jrennie/trg/papers/dawid-prequential-84.pdf). The introduction describes forecasts issued sequentially, outcomes subsequently observed, and the accumulated experience used for later forecasts. This supports the predict/observe/update discipline. It does not establish our375minute timestamp contract, a three-month window, independence, BTC accuracy, or economic superiority.

2. M. H. Pesaran and A. Timmermann (2007), *Selection of estimation window in the presence of breaks*, Journal of Econometrics137,134–161. [Author-hosted January2006 version](https://rady.ucsd.edu/_files/faculty-research/timmermann/estimation-window.pdf), [published DOI](https://doi.org/10.1016/j.jeconom.2006.03.010). Abstract and Sections1–2 explain the bias/forecast-variance tradeoff when older observations precede a break: retaining some pre-break observations can improve MSFE; removing older data is not uniformly better. Break timing, size and exogeneity complicate window selection. This motivates a falsifiable fixed-window comparison and its uncertainty, not selection of3months or a claim that rolling correction must help.

3. T. Dimitriadis and M. Puke (2026), *Statistical Inference for Score Decompositions*, [original preprint v1](https://arxiv.org/html/2603.04275v1). Section3 separates recalibration, discrimination and uncertainty; its canonical no-added-information statement concerns a measurable transformation of the forecast alone. Assumption3.1 imposes strict stationarity, correct linear recalibration and an interior parameter; further inference requires mixing/moment and nondegeneracy conditions. Our rolling historical state and frozen boundary-clipped weights do not inherit these results. Score decomposition remains descriptive algebra; the paper does not guarantee forecast/economic gains for the proposed update.

Source crosscheck used current `oracle_mean_reliability_registration_20260906.md`, `oracle_mean_reliability_decisions.py`, `oracle_risk_calibration.py`, `oracle_derivative_delay.py`, and `oracle_frozen_procedure_parity.py`. The proposed history contract deliberately distinguishes inference, matured history, and retrospective scoring masks.
