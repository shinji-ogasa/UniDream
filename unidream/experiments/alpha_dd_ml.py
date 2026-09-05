"""Preregistered gap-aware ML-only continuation; prior experiments stay immutable."""
from __future__ import annotations

import argparse
from pathlib import Path

from .alpha_dd_features import FEATURE_NAMES, make_features
from .alpha_dd_search import Candidate, candidate_universe, run


RECIPE_NAME = "gap_aware_ml_v1"


def registered_candidates() -> list[Candidate]:
    """B&H and the original 24 fixed ML candidates, with no new sizing search."""
    return [candidate for candidate in candidate_universe()
            if candidate.family in ("hold", "ridge", "hgb", "logistic")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("development", "historical", "fresh"), required=True)
    args = parser.parse_args()
    run(args.config, args.stage, candidates=registered_candidates(),
        feature_builder=make_features, feature_names=FEATURE_NAMES, recipe_name=RECIPE_NAME)


if __name__ == "__main__":
    main()
