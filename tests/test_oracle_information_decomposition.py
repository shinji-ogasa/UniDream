import copy
from pathlib import Path
import unittest

from unidream.experiments.oracle_information_decomposition import (
    CONTROLS, FOLDS, FIXED, ORACLES, POLICIES, SOURCE_ROOT, SOURCES, summarize, validate_config,
)


class InformationDecompositionTests(unittest.TestCase):
    def config(self):
        return {**copy.deepcopy(FIXED),'source_bindings':{p:'0'*64 for p in SOURCES},
            'parity_config_sha256':'0'*64,'preflight_sha256':'0'*64,
            'source_manifest_bindings':{str(SOURCE_ROOT/p):'0'*64 for p in
                ['registration.json','preflight.json','results.json']+[f'fold_{f}.json' for f in FOLDS]}}

    def test_original_development_and_all16_diagnostics_are_fixed_before_outcomes(self):
        cfg=self.config();validate_config(cfg)
        self.assertEqual(len(CONTROLS),12);self.assertEqual(len(ORACLES),16)
        self.assertEqual(len(POLICIES)*len(FOLDS),224)
        for field,value in {'development_validation_folds':list(range(15,25)),
                'data_cutoff':'2026-07-16T13:45:00Z','selection_permitted':True,
                'rl_beam_width':64,'rl_missing_input_rules':['hold'],
                'new_model_fitting_permitted':True,'teacher_use_allowed':True,
                'replacement_support':'inference_mask'}.items():
            with self.subTest(field=field),self.assertRaises(ValueError):validate_config({**cfg,field:value})
        bad=copy.deepcopy(cfg);bad['source_manifest_bindings'].pop(next(iter(bad['source_manifest_bindings'])))
        with self.assertRaises(ValueError):validate_config(bad)

    def rows(self):
        rows=[]
        for f in FOLDS:
            for cid in POLICIES:
                value={'alpha_ex':.1 if cid in ORACLES else .01,'maxdd_delta':-.02,'turnover':.3,'trades':4}
                rows.append({'fold':f,'candidate_id':cid,'regime':{'trend':'bull'},'base':value,'stress_2x':value})
        return rows

    def test_summary_preserves_all_rows_own_reference_and_noncausal_scope(self):
        result=summarize(self.rows())
        self.assertEqual(len(result['policies']),28)
        self.assertEqual(len(result['oracle_minus_own_learned']),12)
        for row in result['oracle_minus_own_learned'].values():
            self.assertAlmostEqual(row['strata']['all']['base']['alpha_ex'],.09)
            self.assertEqual(row['strata']['bear']['quarters'],0)
            self.assertIsNone(row['strata']['bear']['base']['alpha_ex'])
        self.assertFalse(result['selection_performed']);self.assertFalse(result['teacher_use_allowed'])
        self.assertFalse(result['high_probability_generalization_established'])

    def test_missing_or_duplicated_counterfactual_is_not_silently_deduplicated(self):
        rows=self.rows()
        with self.assertRaises(ValueError):summarize(rows[:-1])
        with self.assertRaises(ValueError):summarize(rows+[rows[0]])


if __name__=='__main__':unittest.main()
