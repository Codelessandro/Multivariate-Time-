
import numpy as np
import pandas as pd
from fbprophet import Prophet

from config import config
from models.model import Model


class Gauss(Model):
    def __init__(self,name):
        self.name=name
        self.dims = 2
        self.univariate= True
        super().__init__()



    def build_model(self,_,__):
        pass


    def apply_data_transformation(self,data):
        self.org_df = data
        return data

    def train(self, x, y):
        self.train_std = np.std(x[:,0])
        self.train_mean = np.mean(x[:,0])



    def score(self,x,y,target_month):
        pred = np.random.normal(loc=self.train_mean, scale=self.train_std, size=x.shape[0])
        diffs = np.abs(pred-y.reshape(y.shape[0]))
        return np.mean(diffs)



