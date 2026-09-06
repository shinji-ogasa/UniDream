import copy
import unittest

import numpy as np
import pandas as pd

from unidream.experiments.wm_rl_release import summarize_matrix, path_metrics


def matrix():
    rows = []
    for arm, alpha, dd in [('bc_only', .01, -.01), ('ac_test', .02, -.015)]:
        for fold in [6, 7, 9]:
            for cost in ['base', 'stress']:
                rows.append(dict(arm=arm, fold=fold, cost_case=cost, split='validation',
                    metrics=dict(alpha_ex=alpha, maxdd_delta=dd),
                    bc_intent_difference={'changed_rows': 10} if arm.startswith('ac_') else {}))
    return rows


class ReleaseSelectionTests(unittest.TestCase):
    def test_selects_only_active_rl_passing_both_cost_cases(self):
        self.assertEqual(summarize_matrix(matrix())['screen_selected'], 'ac_test')
        rows = matrix()
        for row in rows:
            if row['arm'] == 'ac_test' and row['cost_case'] == 'stress':
                row['metrics']['alpha_ex'] = -.01
        self.assertIsNone(summarize_matrix(rows)['screen_selected'])

    def test_nominal_rl_same_decisions_as_bc_is_not_selected(self):
        rows = matrix()
        for row in rows:
            row['bc_intent_difference'] = {'changed_rows': 0}
        self.assertIsNone(summarize_matrix(rows)['screen_selected'])

    def test_test_rows_duplicates_and_missing_cost_fold_rejected(self):
        rows = matrix()
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            summarize_matrix(rows)
        with self.assertRaises(ValueError):
            summarize_matrix(matrix() + [copy.deepcopy(matrix()[0])])
        with self.assertRaises(ValueError):
            summarize_matrix(matrix()[1:])

    def test_initial_capital_enters_drawdown_and_buy_hold_basis(self):
        bars = pd.DataFrame({'open': [100., 90., 95.], 'close': [90., 95., 92.]})
        result = path_metrics(bars, np.array([.85, .97, .96]), np.ones(3), np.ones(3), {})
        self.assertAlmostEqual(result['alpha_ex'], .04)
        self.assertAlmostEqual(result['maxdd'], .15)
        self.assertAlmostEqual(result['bh_maxdd'], .10)
        self.assertAlmostEqual(result['maxdd_delta'], .05)

    def test_missing_terminal_never_shifts_period(self):
        bars = pd.DataFrame({'open': [100., 90.], 'close': [90., np.nan]})
        with self.assertRaises(ValueError):
            path_metrics(bars, np.array([.9, np.nan]), np.ones(2), np.ones(2), {})


if __name__ == '__main__':
    unittest.main()
