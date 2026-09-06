import unittest

import numpy as np

from unidream.experiments.oracle_information_interventions import mark_hindsight_trace, substitute_information


class InformationInterventionTests(unittest.TestCase):
    def setUp(self):
        self.mu = np.array([.01, np.nan, -.02, .03])
        self.var = np.array([.001, np.nan, .002, .003])
        self.inf = np.array([True, False, True, True])
        self.score = np.array([True, False, True, False])
        self.actual = np.array([[.2, .1, .3], [np.nan]*3, [-.1, .3, 0.], [np.inf]*3])

    def run_swap(self, selected_swap, **changes):
        args = dict(mu=self.mu, variance=self.var, inference_mask=self.inf,
                    score_support=self.score, actual=self.actual, swap=selected_swap)
        args.update(changes)
        return substitute_information(**args)

    def test_substitution_changes_only_existing_scored_slots_without_support_mutation(self):
        for swap in ('return', 'realized_risk', 'both'):
            x = self.run_swap(swap)
            np.testing.assert_array_equal(x['inference_mask'], self.inf)
            np.testing.assert_array_equal(x['score_support'], self.score)
            np.testing.assert_array_equal(x['mu'][~self.score], self.mu[~self.score])
            np.testing.assert_array_equal(x['variance'][~self.score], self.var[~self.score])
            np.testing.assert_array_equal(x['mu'][self.score], self.actual[self.score,0] if swap != 'realized_risk' else self.mu[self.score])
            np.testing.assert_array_equal(x['variance'][self.score], self.actual[self.score,2]**2 if swap != 'return' else self.var[self.score])
            self.assertEqual(x['metadata']['learned_remainder_rows'], 1)
            self.assertFalse(x['metadata']['deployable'])
        np.testing.assert_array_equal(self.mu, [.01, np.nan, -.02, .03])

    def test_future_tail_values_cannot_gate_or_change_unchanged_inference(self):
        expected = self.run_swap('both')
        changed = self.actual.copy(); changed[~self.score] = -1e200
        actual = self.run_swap('both', actual=changed)
        np.testing.assert_array_equal(expected['mu'], actual['mu'])
        np.testing.assert_array_equal(expected['variance'], actual['variance'])

    def test_invalid_support_and_scored_nonfinite_values_fail(self):
        for changes in ({'score_support': self.inf.astype(int)}, {'score_support': np.ones(4,bool)},
                        {'actual': self.actual.astype(complex)}, {'mu': self.mu.astype(complex)},
                        {'swap': 'new'}, {'score_support': np.zeros(4,bool)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):self.run_swap('both', **changes)
        bad = self.actual.copy();bad[0,2] = 1e308
        with self.assertRaises(ValueError):self.run_swap('both', actual=bad)
        bad[0,2] = -.1
        with self.assertRaises(ValueError):self.run_swap('both', actual=bad)

    def test_causal_trace_labels_are_overridden_without_mutating_source(self):
        source = {'diagnostic_kind':'causal', 'hindsight_only':False,
            'future_information_used_for_decisions':False,
            'decision_trace':{'bar_indices':[0,2,3], 'reasons':['learned']*3}}
        changed = mark_hindsight_trace(source, swap='both', score_support=self.score)
        self.assertTrue(changed['hindsight_only'])
        self.assertTrue(changed['future_information_used_for_decisions'])
        self.assertEqual(changed['decision_trace']['reasons'], ['hybrid_hindsight','hybrid_hindsight','learned'])
        self.assertEqual(source['decision_trace']['reasons'], ['learned']*3)


if __name__ == '__main__': unittest.main()
