import pdb
import numpy as np
import pandas as pd

from config import config


def get_data_from_dataset(dataset):

    if dataset == 'Sales_Transactions_Dataset_Weekly.csv':
        data = open("datasets/" + dataset, "r+").read()
        '''        
        data = data.split("\n")
        columns = data[0]
        data = data[1:]
        columns_data = np.array(columns)
        import pdb
        pdb.set_trace()
        '''
        return data

    if len(dataset.split('r_stationary.pkl'))>1:
        return pd.read_pickle('datasets/' + dataset)

    if len(  dataset.split('r_non_stationary.pkl'))>1:
        return pd.read_pickle(dataset)


def preprocess(data):
    def add_month_onehot(data):
        data = data.copy()
        for i in range(11):
            data[f"month_{i + 1}"] = data['date(t)'].apply(lambda d: 1 if d.month == i + 1 else 0)
        return data

    #data = add_month_onehot(data)

    return data


def series_to_supervised(model,data,target_month,  n_in=16, n_out=1, dims=2, dropnan=True):
    target_month=target_month-1
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        three_dims: boolean whether to build data two dimensionsal for LSTMs or CNNS etc. that have dims: (time_step, input_dim)
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    if dims==2:
        n_vars = 1 if type(data) is list else data.shape[1]
        columns = data.columns
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in columns]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (j)) for j in columns]
            else:
                names += [('%s(t+%d)' % (j, i)) for j in columns]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)


        data = agg
        PREDICTORS = [c for c in data.columns if '(t)' not in c and 'date' not in c and not '(t+' in c]
        TARGETS = [f"fpreis(t+{target_month})" if target_month != 0 else "fpreis(t)"]

        if model.univariate:
            PREDICTORS = list(filter(lambda p: len(p.split("fpreis")) > 1, PREDICTORS))

        x = data[PREDICTORS].values
        y = data[TARGETS].values

        return x,y

    if dims==3:
        _columns = data.columns
        #_columns=np.delete(_columns,np.where(_columns=='date')[0][0]) #delete date

        x=[]
        y=[]
        y_index = n_in+n_out
        x_index = 0
        while y_index < data.shape[0]:
            _x = data.to_numpy()[x_index:x_index+n_in]
            _y = data.to_numpy()[y_index-1][np.where(data.columns=='fpreis')[0][0]]
            x.append(_x)
            y.append(_y)
            x_index+=1
            y_index+=1


        x=np.concatenate(x, axis=0).reshape((len(x), x[0].shape[0], x[0].shape[1]))
        y=np.array(y)
        x=x[:,:,1:] #remove date

        return x,y






def test_train(x,y):
    def randomize_split_data(x, y, split_value=-config["train_test_split"]):
        indices = np.arange(x.shape[0])  # [0,1,2,3,4,5,6,7,8....173]
        np.random.shuffle(indices)  # [1,10,11,0,2,3, ... 173.. 4 9]
        shuffle_indices = indices
        return x[shuffle_indices[split_value:]], y[shuffle_indices[split_value:]], x[shuffle_indices[:split_value]], y[shuffle_indices[:split_value]]

    def final_split(x,y,split_value=-config["train_test_split"]):
        return  x[-split_value:,:], y[-split_value:], x[:-split_value, :], y[:-split_value]


    if config["split_method"]=="final":
        return final_split(x,y,config["train_test_split"])

    if config["split_method"]=="random":
        return final_split(x,y,-config["train_test_split"])




