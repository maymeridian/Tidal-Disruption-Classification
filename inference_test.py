'''
tests/test_inference.py
'''
import unittest
import pandas as pd
import os
import sys
from src.inference import InferenceModel
from src.data_loader import load_lightcurves
from config import TEST_LOG_PATH, RESULTS_DIR

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = InferenceModel()
        cls.lc = load_lightcurves('test')
        cls.log = pd.read_csv(TEST_LOG_PATH).set_index('object_id')
        cls.reference_submission = pd.read_csv(os.path.join(RESULTS_DIR, 'submission_2026-01-31_0.7250.csv'))


    def test_single_prediction(self):
        """
        Verifies the model runs on a single object without crashing 
        and returns a valid probability (0.0 to 1.0).
        """
        oid = self.log.index[0]
        meta = self.log.loc[oid]
        lc_data = self.lc[self.lc['object_id'] == oid]

        pred, prob = self.model.predict(
            lc_data, 
            meta.get('Z', 0), 
            meta.get('Z_err', 0), 
            meta.get('EBV', 0)
        )
        
        print(f"\n[Test] ID: {oid} | Pred: {pred} | Prob: {prob:.4f}")
        self.assertIsInstance(pred, int)
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_batch_of_objects(self):

        common_ids = self.log.index.intersection(self.reference_submission.index)
        sample_ids = common_ids[:100] # test 100 objects

        for oid in sample_ids:
            lc_data = self.lc[self.lc['object_id'] == oid]

            meta = self.log.loc[oid]
            batch_pred = self.reference_submission.loc[oid, 'prediction']

            rt_pred, rt_prob = self.model.predict(
                lc_data, 
                meta.get('Z', 0), 
                meta.get('Z_err', 0), 
                meta.get('EBV', 0)
            )

            self.assertEqual(rt_pred, int(batch_pred), 
                f"Mismatch on {oid}: Batch={batch_pred}, RT={rt_pred} (Prob={rt_prob:.4f})")

if __name__ == '__main__':
    unittest.main()