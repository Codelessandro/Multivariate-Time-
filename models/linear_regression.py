import numpy as np
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.metrics import r2_score

from models.model import Model


class LinearRegression(Model):

    def __init__(self, name, univariate):
        self.model = _LinearRegression()
        self.dims=2
        self.univariate= univariate
        self.name = name

        super().__init__()

    def build_model(self,_,__):
        pass

    def apply_data_transformation(self, data):
        return data

    def train(self,x,y):
        self.model.fit(x,y)

    def score(self,x,y,target_month):
        p_test = self.model.predict(x)
        diffs=y - p_test
        score=np.abs(diffs).sum() / len(diffs)
        return score




