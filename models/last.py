import numpy as np

from models.model import Model


class LAST(Model):
    def __init__(self,name):
        self.dims = 2
        self.univariate=True
        self.name=name
        super().__init__()


    def build_model(self,_,__):
        pass

    def apply_data_transformation(self,data):
        return data



    def train(self, x, y):
        pass
        pass

    def score(self,x,y,target_month):
        p_test = x[:,-1]
        diffs= y.reshape(y.shape[0]) - p_test
        score=np.abs(diffs).sum() / len(diffs)
        return score


