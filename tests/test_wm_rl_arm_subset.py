import unittest
from unidream.experiments.wm_rl_training import AC_ARMS, selected_ac_arms


class ArmSubsetTests(unittest.TestCase):
    def test_default_preserves_original_three_and_copy(self):
        selected = selected_ac_arms({})
        self.assertEqual(selected, AC_ARMS)
        self.assertEqual(list(selected), list(AC_ARMS))
        selected['ac_decay_dd25']['alpha_final'] = 100
        self.assertEqual(AC_ARMS['ac_decay_dd25']['alpha_final'], .05)

    def test_one_prebound_arm_and_explicit_order(self):
        self.assertEqual(selected_ac_arms({'release': {'ac_arm_names':['ac_decay_dd25']}}),
                         {'ac_decay_dd25': AC_ARMS['ac_decay_dd25']})
        names = ['ac_anchor_dd50','ac_decay_dd25']
        self.assertEqual(list(selected_ac_arms({'release':{'ac_arm_names':names}})),names)

    def test_malformed_empty_duplicate_and_unknown_fail(self):
        for names in ([], None, 'ac_decay_dd25', ['ac_decay_dd25']*2, ['unknown'], [True], [{}]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                selected_ac_arms({'release':{'ac_arm_names':names}})
