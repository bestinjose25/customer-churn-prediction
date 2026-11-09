import unittest
import tempfile
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from train import load_data, make_pipeline

class PipelineTests(unittest.TestCase):
    def test_raw_data_and_unseen_category(self):
        data = pd.DataFrame({'customerID':['a','b','c','d'], 'Churn':['No','Yes','No','Yes'],
                             'TotalCharges':[' ', '20', '10', '40'], 'Contract':['A','B','A','B']})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.csv'
            data.to_csv(path,index=False)
            x,y=load_data(path)
            self.assertTrue(pd.isna(x.iloc[0]['TotalCharges']))
            self.assertNotIn('customerID',x.columns)
            pipeline=make_pipeline(x,LogisticRegression()).fit(x,y)
            incoming=x.iloc[[0]].copy()
            incoming['Contract']='previously unseen'
            probability=pipeline.predict_proba(incoming)[0,1]
            self.assertTrue(0 <= probability <= 1)
            data.loc[1,'customerID']='a'
            data.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'unique'):
                load_data(path)

if __name__=='__main__':
    unittest.main()
