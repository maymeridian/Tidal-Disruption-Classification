'''
tests/test_inference.py
'''
import unittest
import pandas as pd
import os
import sys
from src.inference import load_inference_model, process_single_object

from config import TEST_LOG_PATH, RESULTS_DIR
from src.data_loader import load_lightcurves

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, threshold = load_inference_model()
        cls.model = model
        cls.threshold = threshold
        cls.lc = load_lightcurves('test')
        cls.log = pd.read_csv(TEST_LOG_PATH).set_index('object_id')
        cls.reference_submission = pd.read_csv(os.path.join(RESULTS_DIR, 'submission_2026-01-31_0.7250.csv'))


    def test_single_prediction(self):

        # need to load redshift, ebv
        # as 'meta knowledge' since we aren't calculating this ourselves and in a 
        # real environment we wouldn't obviously be looking up from a file
        # ----------------------
        common_ids = self.log.index.intersection(self.reference_submission.index)
        sample_ids = common_ids[:1] # test 100 objects

        for oid in sample_ids:
            lc_data = self.lc[self.lc['object_id'] == oid]
            meta = self.log.loc[oid]
            redshift = meta.get('Z', 0)
            ebv = meta.get('EBV', 0)
            # ----------------------

            batch_pred = self.reference_submission.loc[oid, 'prediction'] # load our 'ground truth' for the purposes of this test
            features = process_single_object(lc_df, redshift, ebv)
            X = features.drop(columns=['object_id'])
            # get predictions
            pred, prob = self.model.predict(X)
            # apply threshold
            pred = (prob >= self.threshold).astype(int)

            print(f"\n[Test] ID: {oid} | Pred: {pred} | Prob: {prob:.4f}")
            self.assertIsInstance(pred, int)
            self.assertTrue(0.0 <= prob <= 1.0)
            self.assertEqual(rt_pred, int(batch_pred), 
                f"Mismatch on {oid}: Batch={batch_pred}, RT={pred} (Prob={prob:.4f})")
            print("test")

    def test_batch_of_objects(self):

        common_ids = self.log.index.intersection(self.reference_submission.index)
        sample_ids = common_ids[:3000] # test 3000 objects
       
        for oid in sample_ids:
            lc_data = self.lc[self.lc['object_id'] == oid]
            meta = self.log.loc[oid]
            batch_pred = self.reference_submission.loc[oid, 'prediction']
            redshift = meta.get('Z', 0)
            ebv = meta.get('EBV', 0)
            # ----------------------


            features = process_single_object(lc_df, redshift, ebv)
            X = features.drop(columns=['object_id'])
            # get predictions
            pred, prob = self.model.predict(X)
            # apply threshold
            y_preds = (y_probs >= self.threshold).astype(int)

            print(f"\n[Test] ID: {oid} | Pred: {pred} | Prob: {prob:.4f}")
            self.assertIsInstance(pred, int)
            self.assertTrue(0.0 <= prob <= 1.0)
            self.assertEqual(pred, int(batch_pred), 
                f"Mismatch on {oid}: Batch={batch_pred}, RT={pred} (Prob={prob:.4f})")

if __name__ == '__main__':
    unittest.main()
