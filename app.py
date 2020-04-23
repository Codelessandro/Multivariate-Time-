from config import config
from data_handler import get_data_from_dataset, preprocess, series_to_supervised

from benchmark import save_score, plot_scores, best_scores
from models.ets import ETS
from models.arima import Arima
from models.ses import SES
from models.lstm import _LSTM
from models.convnet import CONVNET
from models.mean import MEAN
from models.last import LAST
from models.prophet import _Prophet
from models.gauss import Gauss
from models.linear_regression import LinearRegression

import pandas as pd

from utils import add_xgb
from data_handler import test_train

from matplotlib import pyplot as plt

import numpy as np


try:
    scores = pd.read_csv('cache/scores.csv', sep=" ")
except:
    scores = pd.DataFrame(columns=['model', 'target_month', 'window_length', 'score'])

scores = pd.DataFrame(columns=['model', 'target_month', 'window_length', 'score'])


dataset = 'datasets/r_non_stationary.pkl'


models=[_LSTM("LSTM"),CONVNET("CONVNET"), _Prophet("Prophet"),MEAN("MEAN"),LAST("LAST"),LinearRegression("Linear Regression", False),SES("SES")]


lr_u = LinearRegression("Univariate Linear Regression", True)
lr_m = LinearRegression("Multivariate Linear Regression", False)
mean = MEAN("MEAN")


models = [mean, lr_u, lr_m, _Prophet("Prophet")]
models=add_xgb(models)


models=[Gauss("Gauss-1"),Gauss("Gauss-2"),lr_m,mean]


data = get_data_from_dataset(dataset)
source_data = preprocess(data)




for window_length in config["window_length"]:
    for target_month in config["target_month"]:
        for model in models:

            if model.load_from_cache==False:

                transformed_data = model.apply_data_transformation(source_data)
                model.build_model(window_length,target_month)

                x,y = series_to_supervised(model,transformed_data, target_month,  n_in=window_length,n_out=target_month,dims=model.dims)

                test_x, test_y, train_x, train_y = test_train(x,y)


                model.train(train_x, train_y)
                score = model.score(test_x, test_y,target_month)

                scores = save_score(scores,dataset, target_month, window_length, score, model.name)


scores.to_csv('cache/scores.csv', sep=' ', index=False, header=True)



model_names = np.unique( list(map(lambda m: m.name, models)))
for model_name in model_names:
    plot_scores(scores, model_name)

plt.savefig('plots/plot.png')
plt.show()
