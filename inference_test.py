'''
tests/test_inference.py
'''
import unittest
import pandas as pd
import os
from src.inference import load_inference_model, process_single_object, predict_single_object

from config import TEST_LOG_PATH, RESULTS_DIR
from src.data_loader import load_lightcurves
import warnings
from sklearn.exceptions import ConvergenceWarning

class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, threshold = load_inference_model()
        cls.model = model
        cls.threshold = threshold
        cls.lc = load_lightcurves('test')
        cls.log = pd.read_csv(TEST_LOG_PATH).set_index('object_id')
        cls.reference_submission = pd.read_csv(
                    os.path.join(RESULTS_DIR, 'submission_2026-01-31_0.7250.csv')
                ).set_index('object_id')

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Suppress missing value imputation warnings
        warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.impute')
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

    def test_single_prediction(self):
        common_ids = self.log.index.intersection(self.reference_submission.index)
        sample_ids = common_ids[:1]

        for oid in sample_ids:
            lc_data = self.lc[self.lc['object_id'] == oid].copy()
            meta = self.log.loc[oid]
            
            # prevent NaN imputation errors
            redshift = meta.get('Z', 0.0)
            if pd.isna(redshift): 
                redshift = 0.0
            
            z_err = meta.get('Z_err', 0.0)
            if pd.isna(z_err): 
                z_err = 0.0
            
            ebv = meta.get('EBV', 0.0)
            if pd.isna(ebv): 
                ebv = 0.0

            batch_pred = self.reference_submission.loc[oid, 'prediction']
            features = process_single_object(lc_data, redshift, z_err, ebv)
            
            # Handle fit_2d_gp failure, uncommon but happens with a few objects. 
            # will be updating rest of code base to fix this issue soon
            if features is None:
                pred, prob = 0, 0.0
            else:
                X = features.drop(columns=['object_id'])
                prob = predict_single_object(self.model, X)
                pred = int(prob >= self.threshold)

            print(f"\n[Test] ID: {oid} | Pred: {pred} | Prob: {prob:.4f}")
            self.assertIsInstance(pred, int)
            self.assertTrue(0.0 <= prob <= 1.0)
            self.assertEqual(pred, int(batch_pred), 
                f"Mismatch on {oid}: Batch={batch_pred}, RT={pred} (Prob={prob:.4f})")

    def test_batch_of_objects(self):
        common_ids = self.log.index.intersection(self.reference_submission.index)
        sample_ids = common_ids[:3000]
        
        total_objects = len(sample_ids)
        processed_count = 0
        correct_count = 0

        print(f"\n--- Starting Batch Inference Test on {total_objects} Objects ---")

        for oid in sample_ids:
            with self.subTest(oid=oid):
                try:
                    lc_data = self.lc[self.lc['object_id'] == oid].copy()
                    meta = self.log.loc[oid]
                    batch_pred = self.reference_submission.loc[oid, 'prediction']
                    
                    # Robust extraction to prevent NaN imputation errors
                    redshift = meta.get('Z', 0.0)
                    if pd.isna(redshift): 
                        redshift = 0.0
                    
                    z_err = meta.get('Z_err', 0.0)
                    if pd.isna(z_err): 
                        z_err = 0.0
                    
                    ebv = meta.get('EBV', 0.0)
                    if pd.isna(ebv): 
                        ebv = 0.0

                    features = process_single_object(lc_data, redshift, z_err, ebv)
                    
                    if features is None:
                        pred, prob = 0, 0.0
                    else:
                        X = features.drop(columns=['object_id'])
                        prob = self.model.predict_proba(X)[0, 1]
                        pred = int(prob >= self.threshold)

                    is_match = (pred == int(batch_pred))
                    if is_match:
                        correct_count += 1
                    
                    processed_count += 1
                    
                    if processed_count % 50 == 0 or processed_count == total_objects:
                        match_rate = (correct_count / processed_count) * 100
                        print(f"Progress: {processed_count}/{total_objects} processed | "
                              f"Exact Matches: {correct_count} | Match Rate: {match_rate:.1f}%")

                    # test assertion
                    self.assertEqual(pred, int(batch_pred), 
                        f"Mismatch on {oid}: Batch={batch_pred}, RT={pred} (Prob={prob:.4f})")
                
                except Exception as e:
                    print(f"\n[CRASH] Object ID {oid} failed: {type(e).__name__} - {str(e)}")
                    self.fail(f"Crashed on {oid}")

if __name__ == '__main__':
    unittest.main()