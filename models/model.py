import numpy as np
import pandas as pd


class Model():

    def __init__(self):
        self.load_from_cache = False


    def __to_csv(self):
        self.scores.to_csv('cache/' +  self.name + '.csv', sep=' ', index=False, header=False)

    def __from_csv(self):
        self.scores = pd.read_csv('cache/' +  self.name + '.csv')
