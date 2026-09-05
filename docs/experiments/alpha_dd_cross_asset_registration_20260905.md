# ETH / BNB transfer experiment, registered before either asset's results

The user explicitly allows an asset other than BTC. BTC experiment v1 has failed
the joint mean targets and is retained as a failed trial. This new family tests
the identical 83 registered policies on ETHUSDT and BNBUSDT; the benchmark for each
is that asset's own buy-and-hold. No mixed benchmark or hindsight asset switching.

The two YAML files pin the same 13 development folds, nine historical confirmation
folds, one mostly-new confirmation fold, execution timing, cash/units account,
5.5-bps cost, 10% borrowing assumption, .08 adjustment cap, and maximum target1.12.
All parameters match BTC v1. The new core also binds the symbol to the data sidecar.

Run development for both assets first. For each asset, the runner locks its highest
`min(mean AlphaEx, -mean MaxDDDelta)` candidate. Select exactly one asset using
that score, then AlphaEx, DD delta, and lexicographic symbol as deterministic ties.
Persist a cross-asset lock with both development file hashes before computing any
confirmation metrics. Run historical and fresh confirmation only for that locked
asset/candidate; a failure cannot trigger selecting the other asset from its test.
Thresholds apply to the combined ten equal-weight quarters, with both subset
results disclosed. Archive availability remains retrospective, not live evidence.

This is a further research trial after observing BTC's failed confirmation.
The full search count (83 BTC plus 166 cross-asset candidates, including N/A
candidates) is disclosed. A positive empirical mean is not a selection-adjusted
significance claim or proof of future returns. The still-pending input-coverage
ML experiment is a separately registered family; its model selection also uses
development only.
