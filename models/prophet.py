
import numpy as np
import pandas as pd
from fbprophet import Prophet

from config import config
from models.model import Model


class _Prophet(Model):
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
        if config["split_method"] == "random":
            raise Exception('Prophet only works for Final Split - not for Random Split.')

        '''

        self.model = Prophet()
        if config["split_method"]=="random":
            raise Exception('Prophet only works for Final Split - not for Random Split.')


        last_element =  x[-1][0]
        last_row = self.org_df.loc[self.org_df['fpreis'] == last_element]

        if last_row.shape[0]>1:
            raise Exception('Error. Value duplicate. Please refactor this code again.')

        last_date = last_row.date
        self.last_train_date = last_date
        train_df = self.org_df[   (self.org_df['date'] <  last_date.values[0])]
        train_df = train_df.rename(columns={"date": "ds", "fpreis": "y"})
        self.model.fit(train_df)
        '''


    def score(self,x,y,target_month):

        scores=[]
        for index,_y in enumerate(y):

            try:
                last_element = x[index][-1]
            except:
                import pdb; pdb.set_trace()
            last_row = self.org_df.loc[self.org_df['fpreis'] == last_element]


            if last_row.shape[0]>1:
                raise Exception('Error. Value duplicate. Please refactor this code again.')

            last_date = last_row.date
            train_df = self.org_df[(self.org_df['date'] < last_date.values[0])]
            train_df = train_df.rename(columns={"date": "ds", "fpreis": "y"})

            model = Prophet()
            model.fit(train_df)

            future = model.make_future_dataframe(freq='M', periods=target_month, include_history=False)

            forecast = model.predict(future)
            pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            pred = pred.to_numpy()[:, 1][-1]


            score = np.abs(_y - pred)


            scores.append(score)

        print(np.mean(scores))
        return int(np.mean(scores))

