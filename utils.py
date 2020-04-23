from models.xgb import XGB
import random
import numpy as np

from config import config

def add_xgb(models):
    def random_xgb_config():
        return {
            "objective": "reg:linear",
            'colsample_bytree': np.random.rand(),
            'learning_rate': random.choice([0.1, 0.2, 0.3, 0.05, 0.01]),
            'max_depth': np.random.randint(2, 20),
            'alpha': np.random.randint(2, 20),
            'n_estimators': np.random.randint(2, 20)
        }

    xgb_params_set = [
        random_xgb_config() for _ in range(config["nr_xgb_models"])
    ]

    for index,xgb_params in enumerate(xgb_params_set):
        models.append(XGB(index,xgb_params))
    return models




