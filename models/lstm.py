import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import pandas as pd
from sklearn import preprocessing

from models.model import Model

class _LSTM(Model):

    def __init__(self,name,hyperparams_per_submodel=4):
        self.univariate= False
        self.dims=3
        self.name=name
        self.hyperparams_per_submodel=hyperparams_per_submodel
        self.models=[]
        self.model=None
        super().__init__()
        self.load_from_cache = True



    def build_model(self, window_length,target_month):


        def gen_random_hp():
            nr_dense_layers =  np.random.choice([1,2,3,4]),
            return {
                "units" : np.random.choice([2,4,8,16,32,64,128]),
                "units_dense_layers" :  list(map(lambda   el: el *np.random.choice([20,40,60,80,100,200]) , np.ones(nr_dense_layers) )),
                "epochs" : np.random.choice([25,50,100,100,500])
            }

        self.models=[]

        for hyperparam_index in np.arange(self.hyperparams_per_submodel):
            hp = gen_random_hp()

            # non-stationary
            model = Sequential()
            model.add(LSTM(hp["units"], input_shape=(window_length, 21)))
            for  units_dense_layer in hp["units_dense_layers"]:
                model.add(Dense(int(units_dense_layer)))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.hp = hp
            self.models.append(model)




    def apply_data_transformation(self,__data):
        import copy
        _data=copy.copy(__data)
        for index,column in enumerate(_data.columns):
            min_max_scaler = preprocessing.MinMaxScaler()
            if  column=='fpreis':
                self.orgfpreis_scaler = copy.copy(_data[column].values.reshape(-1, 1))
            _data[column] =  min_max_scaler.fit_transform(  _data[column].values.reshape(-1, 1) )
        return _data



    def train(self, x,y):
        best_mean_val_loss=np.Infinity
        for model in self.models:
            history = model.fit(x, y, epochs=model.hp["epochs"], batch_size=32, verbose=2, validation_split=0.2)
            mean_val_loss = np.mean(history.history["val_loss"][-5:])

            if mean_val_loss < best_mean_val_loss:
                self.model = model
                best_mean_val_loss = mean_val_loss


    def score(self,x,y,target_month):

        def transform_scaled_to_price(price_org, scaled):
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit_transform(price_org)
            return min_max_scaler.inverse_transform(scaled)

        p_test = self.model.predict(x)
        p_test = transform_scaled_to_price(self.orgfpreis_scaler, p_test)
        y = transform_scaled_to_price(self.orgfpreis_scaler, y.reshape(-1,1) ).reshape(-1)

        diffs=y - p_test.reshape(p_test.shape[0])
        score=np.abs(diffs).sum() / len(diffs)
        print(score)
        return score






