import numpy as np
from models.model import Model


class MEAN(Model):
    def __init__(self,name):
        self.name=name
        self.dims = 2
        self.univariate=True
        super().__init__()


    def build_model(self,_,__):
        pass


    def apply_data_transformation(self,data):
        return data

    def train(self, x, y):
        pass

    def score(self,x,y,target_month):

        p_test = np.mean(x,axis=1)
        diffs= y.reshape(y.shape[0]) - p_test
        score=np.abs(diffs).sum() / len(diffs)
        return score
