import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score


class XGB(xgb.XGBRegressor):

    def __init__(self, index, args):
        self.dims=2
        self.univariate = False
        self.name="xgb" #+ str(index)
        super().__init__(**args)
        self.load_from_cache = False

    def build_model(self,_,__):
        pass


    def apply_data_transformation(self,data):
        return data

    def train(self, x,y):
        super().fit(x,y)


    def score(self,x,y,target_month):

        p_test = super().predict(x)
        diffs=y.reshape(y.shape[0]) - p_test
        score=np.abs(diffs).sum() / len(diffs)
        #score = r2_score(y.flatten(), p_test.flatten())
        return score






