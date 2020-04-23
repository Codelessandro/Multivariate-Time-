import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from models.model import Model


class SES(Model):

    def __init__(self,name):
        self.univariate = True
        self.dims=2
        self.name=name
        super().__init__()

    def apply_data_transformation(self,data):
        return data

    def build_model(self,_,__):
        pass

    def train(self, x,y):
        pass




    def score(self,x,y,target_month):

        alphas=[0,0.2,0.4,0.6,0.8,1]
        best_score=np.Infinity

        for a in alphas:
            diffs=[]
            for index,_x in enumerate(x):

                p_test = SimpleExpSmoothing(_x).fit(smoothing_level=0.2, optimized=False).forecast(y.shape[1])
                diff = y[index] - p_test
                diffs.append(diff)

            score =  (np.abs(diffs).sum() / len(diffs))

            if score<best_score:
                best_score=score


        return score









