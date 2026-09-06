"""Synthetic-only inventory, paired probability, mean and economic gates."""
import copy
import unittest

import numpy as np
import pandas as pd

import test_oracle_soft_direction_decisions as parent_tests
from unidream.experiments.oracle_short_direction_decisions import (
    FIXED, SOURCES, PARENT_ROOT, FEATURE_ROOT, FOLDS, RULES, SEGMENTS,
    GROUP, MODEL_IDS, CLASSIFIERS, CONTROLS, POLICIES, MEANS, OLD_MEANS,
    NEW_MEAN, NEW_IDS, REFERENCES, WEIGHTINGS, manifest_paths,
    validate_config, summarize, action_masks, check_action_support,
)


class ShortDirectionDecisionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED), 'source_bindings': {p: '0'*64 for p in SOURCES},
                'parent_config_sha256': '0'*64, 'feature_config_sha256': '0'*64,
                'preflight_sha256': '0'*64,
                'parent_manifest_bindings': {p: '0'*64 for p in manifest_paths(PARENT_ROOT)},
                'feature_manifest_bindings': {p: '0'*64 for p in manifest_paths(FEATURE_ROOT)}}

    def family(self):
        rows, scores, cs = parent_tests.SoftDirectionDecisionTests().family()
        for f in FOLDS:
            regime = next(r['regime'] for r in rows if r['fold'] == f)
            for cid in NEW_IDS:
                x = {'alpha_ex': .4, 'maxdd_delta': -.06, 'turnover': .5, 'trades': 1}
                rows.append({'fold': f, 'candidate_id': cid, 'regime': regime,
                             'base': x.copy(), 'stress_2x': x.copy()})
            for segment in SEGMENTS:
                scores.append({'fold': f, 'segment': segment, 'mean_id': NEW_MEAN,
                    'regime': regime, 'rows': 20, 'return_mse': .5, 'return_mae': .5,
                    'return_sign_accuracy': .5, 'return_rank_ic': None,
                    'zero_return_mse': 3., 'fit_mean_return_mse': 3.})
                for mid in MODEL_IDS:
                    cs.append({'fold': f, 'segment': segment, 'classifier_id': mid,
                        'regime': regime, 'rows': 20, 'log_loss': .1, 'brier': .1,
                        'binary_accuracy': .5, 'signed_return_mean': .01,
                        'weighted_log_loss': .1, 'weighted_brier': .1,
                        'weighted_binary_accuracy': .5, 'absolute_return_sum': 2.,
                        'absolute_return_mean': .1, 'zero_actual_rows': 0, 'zero_logit_rows': 0})
        return rows, scores, cs

    def test_fixed_family_has_only_two_fits_and_one_mapped_mean_per_fold(self):
        c = self.config(); validate_config(c)
        self.assertEqual((len(CONTROLS), len(POLICIES), len(MEANS), len(CLASSIFIERS)), (80, 82, 25, 12))
        self.assertEqual(MODEL_IDS, (GROUP+'_ordinary', GROUP+'_magnitude'))
        self.assertEqual(len(NEW_IDS), 2)
        self.assertTrue(all('magnitude_soft' in cid for cid in NEW_IDS))
        self.assertEqual(len(REFERENCES), 6)
        self.assertEqual(len(SOURCES), len(set(SOURCES)))
        self.assertEqual((FIXED['new_model_fits'], FIXED['new_unique_fit_priors']), (16, 0))
        for key, value in [('schema', 'unknown'), ('new_model_fits', 8), ('new_unique_fit_priors', 2),
                ('new_causal_names', 4), ('group_dimension', 29), ('group', 'technical'),
                ('weightings', ['magnitude']), ('ordinary_probability_mapped_to_returns', True),
                ('logistic_settings', 'exact_Stage18_C1_over_n'), ('additional_support_removal_permitted', True),
                ('surrogate_mean', 'a*tanh(z/2)'), ('development_folds', [15]),
                ('additional_test_permitted', True), ('selection_permitted', True),
                ('teacher_use_allowed', True), ('adaptive_total_causal_names', 218)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config({**c, key: value})
        for key in ('parent_manifest_bindings', 'feature_manifest_bindings', 'source_bindings'):
            altered = copy.deepcopy(c); altered[key].pop(next(iter(altered[key])))
            with self.assertRaises(ValueError): validate_config(altered)
        with self.assertRaises(ValueError): validate_config({**c, 'unregistered': True})

    def test_complete_family_true_matched_gates_do_not_establish_generalization(self):
        data = self.family()
        self.assertEqual(tuple(map(len, data)), (656, 400, 192))
        s = summarize(*data)
        self.assertEqual(s['regime_counts'], {'bull': 2, 'bear': 4, 'sideways': 2})
        self.assertTrue(s['both_classifier_families_improve_matched_losses_all_strata_both_segments'])
        for mid in MODEL_IDS:
            self.assertTrue(all(s['probability_gates'][mid].values()))
        self.assertEqual(set(s['paired']['all']), set(REFERENCES))
        for cid in NEW_IDS:
            g = s['short_direction'][cid]
            self.assertTrue(g['economic_means_all_strata_both_costs'])
            self.assertTrue(g['economic_improvement_vs_all_six_references_all_strata_both_costs'])
            self.assertTrue(all(g['mapped_mse_vs_all_six_references_improved_all_strata'].values()))
            self.assertTrue(all(g['magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata'].values()))
            self.assertFalse(g['high_probability_generalization_established'])
            self.assertFalse(g['regime_count_gate_pass'])
        self.assertFalse(s['high_probability_generalization_established'])
        self.assertFalse(s['selection_performed'])
        self.assertTrue(s['interval_regime_strata_are_retrospective_evaluation_groupings'])
        self.assertIsNone(s['prediction']['all']['evaluation'][NEW_MEAN]['return_rank_ic'])

    def test_entire_inherited_summary_is_exact_and_separate_from_augmented_summary(self):
        data = self.family()
        old_data = parent_tests.SoftDirectionDecisionTests().family()
        expected = parent_tests.summarize(*old_data)
        s = summarize(*data)
        self.assertEqual(s['inherited_Stage19_summary'], expected)
        s['economics']['all'][CONTROLS[0]]['base']['alpha_ex'] = -99.
        self.assertEqual(s['inherited_Stage19_summary'], expected)

    def test_ordinary_failure_falsifies_family_but_not_magnitude_policy_score_gate(self):
        d = self.family()
        for r in d[2]:
            if r['classifier_id'] == GROUP+'_ordinary' and r['segment'] == 'interval':
                r['brier'] = .5
        s = summarize(*d)
        self.assertFalse(s['probability_gates'][GROUP+'_ordinary']['interval'])
        self.assertTrue(s['probability_gates'][GROUP+'_ordinary']['evaluation'])
        self.assertFalse(s['both_classifier_families_improve_matched_losses_all_strata_both_segments'])
        for cid in NEW_IDS:
            g = s['short_direction'][cid]
            self.assertTrue(all(g['magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata'].values()))
            self.assertTrue(g['economic_means_all_strata_both_costs'])

    def test_each_objective_uses_its_matching_proper_losses(self):
        d = self.family()
        for r in d[2]:
            if r['classifier_id'] == GROUP+'_ordinary':
                r.update(weighted_log_loss=.8, weighted_brier=.8)
            if r['classifier_id'] == GROUP+'_magnitude':
                r.update(log_loss=.8, brier=.8)
        s = summarize(*d)
        self.assertTrue(s['both_classifier_families_improve_matched_losses_all_strata_both_segments'])
        for weighting, keys in [('ordinary', ('brier', 'log_loss')),
                                ('magnitude', ('weighted_brier', 'weighted_log_loss'))]:
            for ref in ('technical_'+weighting, 'prior_'+weighting):
                for key in keys:
                    d = self.family()
                    for r in d[2]:
                        if r['classifier_id'] == ref and r['segment'] == 'evaluation' and r['fold'] >= 11:
                            r[key] = .1  # Tie in just the sideways stratum.
                    s = summarize(*d)
                    self.assertFalse(s['probability_gates'][GROUP+'_'+weighting]['evaluation'])
                    self.assertTrue(s['probability_gates'][GROUP+'_'+weighting]['interval'])

    def test_weighted_null_is_preserved_and_never_converted_to_probability_pass(self):
        d = self.family()
        for r in d[2]:
            if r['fold'] == 5 and r['segment'] == 'interval':
                r.update(absolute_return_sum=0., absolute_return_mean=0., zero_actual_rows=20,
                         weighted_log_loss=None, weighted_brier=None, weighted_binary_accuracy=None)
        s = summarize(*d)
        self.assertIsNone(s['classification']['all']['interval'][GROUP+'_magnitude']['weighted_log_loss'])
        self.assertFalse(s['probability_gates'][GROUP+'_magnitude']['interval'])
        self.assertTrue(s['probability_gates'][GROUP+'_magnitude']['evaluation'])
        self.assertTrue(all(s['probability_gates'][GROUP+'_ordinary'].values()))
        self.assertFalse(s['both_classifier_families_improve_matched_losses_all_strata_both_segments'])
        for cid in NEW_IDS:
            self.assertFalse(s['short_direction'][cid]['magnitude_probability_losses_vs_Technical29_and_prior_improved_all_strata']['interval'])
        for r in d[2]:
            if r['fold'] == 5 and r['segment'] == 'interval' and r['classifier_id'] == GROUP+'_magnitude':
                r['weighted_log_loss'] = 0.
        with self.assertRaises(ValueError): summarize(*d)

    def test_all_six_references_and_stress_sideways_can_falsify_paired_economics(self):
        for ref in REFERENCES:
            for key, value in [('alpha_ex', .5), ('maxdd_delta', -.08)]:
                d = self.family()
                for r in d[0]:
                    if r['candidate_id'] == ref+'_'+RULES[0] and r['fold'] >= 11:
                        r['stress_2x'][key] = value
                s = summarize(*d)
                g = s['short_direction'][NEW_MEAN+'_'+RULES[0]]
                self.assertTrue(g['economic_means_all_strata_both_costs'])
                self.assertFalse(g['economic_improvement_vs_all_six_references_all_strata_both_costs'])
                self.assertTrue(all(g['mapped_mse_vs_all_six_references_improved_all_strata'].values()))
                self.assertTrue(s['short_direction'][NEW_MEAN+'_'+RULES[1]]['economic_improvement_vs_all_six_references_all_strata_both_costs'])

    def test_mse_requires_strict_improvement_over_each_of_six_references_by_segment(self):
        for ref in REFERENCES:
            d = self.family()
            for r in d[1]:
                if r['fold'] >= 11 and r['segment'] == 'interval':
                    if r['mean_id'] == ref: r['return_mse'] = .5
                    if ref.endswith('_soft_zero'):
                        r['zero_return_mse'] = .5
                        if r['mean_id'].endswith('_soft_zero'): r['return_mse'] = .5
                    if ref.endswith('_soft_fit_mean'):
                        r['fit_mean_return_mse'] = .5
                        if r['mean_id'].endswith('_soft_fit_mean'): r['return_mse'] = .5
            s = summarize(*d)
            for cid in NEW_IDS:
                g = s['short_direction'][cid]
                self.assertFalse(g['mapped_mse_vs_all_six_references_improved_all_strata']['interval'])
                self.assertTrue(g['mapped_mse_vs_all_six_references_improved_all_strata']['evaluation'])

    def test_equal_quarter_and_row_pooled_mse_are_distinct_and_pair_differences_have_sign(self):
        d = self.family()
        for r in d[1]:
            if r['fold'] == 5: r['rows'] = 180
            if r['mean_id'] == NEW_MEAN and r['fold'] == 5: r['return_mse'] = 4.
        for r in d[2]:
            if r['fold'] == 5:
                r['rows'] = 180; r['absolute_return_mean'] = r['absolute_return_sum']/180
        s = summarize(*d)
        p = s['prediction']['all']['evaluation'][NEW_MEAN]
        self.assertEqual(p['rows'], 320)
        self.assertEqual(p['return_mse'], (4.+7*.5)/8)
        self.assertEqual(p['pooled_row_mse'], (4.*180+7*.5*20)/320)
        self.assertNotEqual(p['return_mse'], p['pooled_row_mse'])
        pair = s['paired']['all']['technical_magnitude_soft']
        self.assertEqual(pair['prediction']['evaluation']['mse_difference'], (4.+7*.5)/8-1.)
        e = pair['economics'][RULES[0]]['base']
        self.assertAlmostEqual(e['alpha_ex'], .1)
        self.assertAlmostEqual(e['maxdd_delta'], -.02)
        self.assertEqual(e['turnover'], -.5)
        self.assertEqual(e['trades'], -1.)
        cp = s['classification_paired']['all']['evaluation'][GROUP+'_magnitude']['prior_magnitude']
        self.assertAlmostEqual(cp['weighted_log_loss'], -.1)

    def test_missing_duplicate_or_replaced_inventory_rejects_new_and_inherited_rows(self):
        for i in range(3):
            d = self.family(); d[i].pop()
            with self.assertRaises(ValueError): summarize(*d)
            d = self.family(); d[i].append(copy.deepcopy(d[i][0]))
            with self.assertRaises(ValueError): summarize(*d)
            d = self.family(); d[i][-1]['fold'] = 15
            with self.assertRaises(ValueError): summarize(*d)

    def test_new_return_scores_must_share_support_baselines_and_regime(self):
        for key, value in [('rows', 19), ('rows', True), ('return_mse', np.nan),
                ('return_mae', -1.), ('return_sign_accuracy', 1.1), ('return_rank_ic', 2.),
                ('zero_return_mse', 2.), ('fit_mean_return_mse', 2.), ('regime', {'trend': 'bull'})]:
            d = self.family(); d[1][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): summarize(*d)

    def test_new_classifiers_must_share_exact_label_denominators_support_and_regime(self):
        for key, value in [('rows', 19), ('rows', True), ('absolute_return_sum', 3.),
                ('absolute_return_sum', 2), ('absolute_return_mean', .11), ('zero_actual_rows', 1),
                ('zero_logit_rows', True), ('zero_logit_rows', 21), ('zero_logit_rows', -1),
                ('weighted_log_loss', None), ('brier', 1.1), ('log_loss', np.inf),
                ('weighted_brier', -1.), ('binary_accuracy', True), ('signed_return_mean', np.nan),
                ('regime', {'trend': 'bull'})]:
            d = self.family(); d[2][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): summarize(*d)

    def test_invalid_new_economics_and_regime_fail_closed(self):
        for key, value in [('alpha_ex', np.nan), ('maxdd_delta', np.inf),
                ('turnover', True), ('turnover', -1.), ('trades', -1.)]:
            d = self.family(); d[0][-1]['stress_2x'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): summarize(*d)
        d = self.family(); d[0][-1]['regime'] = {'trend': 'bull'}
        with self.assertRaises(ValueError): summarize(*d)

    def test_causal_action_support_is_independent_of_future_score_mask(self):
        index = pd.date_range('2000-01-01', periods=73, freq='15min', tz='UTC')
        opens = np.ones(len(index)); opens[48] = np.nan
        inference = np.zeros(len(index), bool); inference[[0, 24, 48]] = True
        masks = action_masks(index, opens, inference)
        score = inference.copy(); score[24] = False  # A still usable, unmatured-label decision.
        self.assertTrue(masks['learned_eligible'][24]); self.assertFalse(score[24])
        targets = np.full(len(index), np.nan); targets[24] = 1.04; targets[72] = 1.
        check_action_support(targets, masks)
        self.assertTrue(masks['missing_current_open'][48])
        self.assertTrue(masks['fallback_eligible'][72])
        for row, value in [(1, 1.), (48, 1.), (72, np.nan), (72, 1.04), (24, np.inf)]:
            bad = targets.copy(); bad[row] = value
            with self.assertRaises(ValueError): check_action_support(bad, masks)
        bad = inference.copy(); bad[1] = True
        with self.assertRaises(ValueError): action_masks(index, opens, bad)


if __name__ == '__main__':
    unittest.main()
