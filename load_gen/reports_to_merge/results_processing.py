import warnings
warnings.filterwarnings('ignore')
import tsai
from tsai.all import *
print('tsai       :', tsai.__version__)
print('fastai     :', fastai.__version__)
print('fastcore   :', fastcore.__version__)
print('torch      :', torch.__version__)
print('matplotlib :', matplotlib.__version__)

print(torch.cuda.get_device_name(0))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import ticker
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import sys


def create_experiments_dir(directory, model_name):
    if not os.path.exists(str(directory)+str(model_name)):
        os.makedirs(str(directory)+str(model_name))
        print(f"Diretório '{directory}' criado.")

        if not os.path.exists(str(directory)+str(model_name)+'/models'):
            os.makedirs(str(directory)+str(model_name)+'/models')
            print(f'Diretório models criado com sucesso!')
        else:
            print(f'Diretório models já existe.')
        return str(directory)+str(model_name)+'/'
    else:
        print(f"Diretório '{directory}' já existe.")
        return str(directory)+str(model_name)+'/'

#######Some customizations below here######
if len(sys.argv) > 1:
    model_name = str(sys.argv[1])
else:
    model_name = "ResNet"
operation = '/write'
directory = './results_paper'+str(operation)+'/'
directory = create_experiments_dir(directory, model_name)

#######Adjustments#######

if model_name == 'FCN':
    arch = FCN
elif model_name == 'FCNPlus':
    arch = FCNPlus
elif model_name == 'ResNet':
    arch = ResNet
elif model_name == 'ResNetPlus':
    arch = ResNetPlus
elif model_name == 'ResCNN':
    arch = ResCNN
elif model_name == 'TCN':
    arch = TCN
elif model_name == 'InceptionTime':
    arch = InceptionTime
elif model_name == 'InceptionTimePlus':
    arch = InceptionTimePlus
elif model_name == 'OmniScaleCNN':
    arch = OmniScaleCNN
elif model_name == 'XCM':
    arch = XCM
elif model_name == 'XCMPlus':
    arch = XCMPlus



#######End of customizations#############





large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
           'legend.fontsize': med,
           'figure.figsize': (10, 6),
           'axes.labelsize': med,
           'axes.titlesize': med,
           'xtick.labelsize': med,
           'ytick.labelsize': med,
           'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Version
print(mpl.__version__)
print(sns.__version__)

import hyperopt
print(hyperopt.__version__)
from hyperopt import Trials, STATUS_OK, STATUS_FAIL, tpe, fmin, hp
from hyperopt import space_eval
import time
from fastai.callback.tracker import EarlyStoppingCallback
import gc
import pickle
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_error

file_name = './write_sinusuidal/arquivo_final_bkp.csv'

history = 24  # input historical time steps
horizon = 1  # output predicted time steps
test_ratio = 0.2  # testing data ratio
max_evals = 100  # maximal trials for hyper parameter tuning


# Save the results
y_true_fn = '%s_true-%d-%d.pkl' % (model_name, history, horizon)
y_pred_fn = '%s_pred-%d-%d.pkl' % (model_name, history, horizon)

data = pd.read_csv(file_name)
data = data.drop(columns=['type total', 'Src IP', 'Dst IP', 'Label', 'Src Port', 'Dst Port'])
data['time'] = pd.to_datetime(data['time'], unit='s')
data.index = data['time']
data.set_index('time', inplace=True)


#print(data)

#divide data into train and test
train_ind = int(len(data)*0.8)
train = data[:train_ind]
test = data[train_ind:]
#print(train.head())
#print(test.head())
train_length = train.shape[0]
test_length = test.shape[0]

print('Training size: ', train_length)
print('Test size: ', test_length)
print('Test ratio: ', test_length / (test_length + train_length))
#print(data.columns)

#print("Test: "+str(test['mean']))

plt.figure(figsize=[12, 6])
plt.plot(data.index[:train_length], data['mean'][:train_length], label='Training', color='blue')
plt.plot(data.index[train_length:], data['mean'][train_length:], label='Test', color='red')
plt.axvspan(data.index[train_length:][0], data.index[train_length:][-1],  facecolor='g', alpha=0.1)

plt.xlabel('Time')
plt.ylabel('Cassandra Write (Latency)')
plt.legend(loc='best')
plt.grid(False)
# plt.show()
plt.savefig(directory+str(model_name)+'_training_test_split.pdf', bbox_inches = 'tight', pad_inches = 0.1)
#print(data)


data.rename(columns={'mean': 'target'}, inplace=True)
columns = ['ops', 'row/s', '.95', 'max', 'target']
df = data[columns]
#print(df)
n_vars = 5
columns=[f'{columns[i]}' for i in range(n_vars-1)]+['target']
X, y = SlidingWindow(5, stride=1, horizon=0, get_x=columns[:-1], get_y='target', seq_first=True)(df)
splits = TrainValidTestSplitter(valid_size=.2, shuffle=False)(y)
#print(X.shape)
#print(y.shape)
#print(splits)
#plot_splits(splits)
X.shape, y.shape, splits

def check_feature_importance(df):
    colunas_desejadas = ['ops', 'op/s', 'pk/s', 'row/s', 'mean', 'med',
                         '.95', '.99', '.999', 'max', 'time', 'stderr',
                         'errors', 'gc: #', 'max ms', 'sum ms', 'sdv ms']

    novo_df = df[colunas_desejadas].copy()
    print(df)

def save_training_time(i, training_time):
    with open(directory+str(i)+'_training_time'+str(model_name)+'.txt', 'w') as f:
        f.write(str(i)+'\n'+str(training_time))

def check_error(orig, pred, index):
    name_col = ''
    index_name = ''

    orig = np.array(orig)
    pred = np.ravel(np.array(pred))

    bias = np.mean(orig - pred)
    bias = "{:.2f}".format(bias)

    mse = mean_squared_error(orig, pred)
    mse = "{:.2f}".format(mse)

    rmse = sqrt(mean_squared_error(orig, pred))
    rmse = "{:.2f}".format(rmse)

    mae = mean_absolute_error(orig, pred)
    mae = "{:.2f}".format(mae)

    mape = np.mean(np.abs((orig - pred) / orig))
    mape = "{:.2f}".format(mape)

    error_group = [bias, mse, rmse, mae, mape]
    result = pd.DataFrame(error_group, index=['BIAS', 'MSE', 'RMSE', 'MAE', 'MAPE'], columns=[name_col])
    result.index.name = index_name
    print("Result: " + str(result))
    with open(directory+str(index)+str("_")+str(model_name)+'_FINAL_REPORTS.txt', 'w') as f:
        f.write(str(i)+'\n'+str(result))

def save_default_metrics(learn, index):
    #[mae, mse, rmse, mape]
    #print("Metrics: "+str(learn.recorder.final_record))
    #print("Metrics Names: "+str(learn.recorder.metric_names))
    metrics = learn.recorder.final_record[2:]
    metrics_names = learn.recorder.metric_names[3:-1]

    with open(directory+str(index)+str("_")+str(model_name)+f'_FINAL_METRICS_last_epoch.txt', 'w') as f:
        # Itera sobre as métricas e seus respectivos nomes e escreve no arquivo
        for name, value in zip(metrics_names, metrics):
            line = f'{name}: {value:.2f}\n'  # Formata a linha
            f.write(line)  # Escreve a linha no arquivo

def save_metrics_plot(learn, X, y, index):
    learn.plot_metrics(path=directory+str(index)+str("_")+str(model_name)+str("_")+f'FINAL_METRICS.pdf')
    #learn.plot_top_losses(X[splits[1]], y[splits[1]], largest=True)
    #learn.top_losses(X[splits[1]], y[splits[1]], largest=True)
    #learn.show_probas()
    #learn.feature_importance()

def save_trained_model(learn, i):
    learn.export(directory+str("models/")+str(i)+str("_")+str(model_name)+f'.pth')



search_space = {
    'batch_size': hp.choice('bs', [16, 32, 64, 128]),
    "lr": hp.choice('lr', [0.1, 0.01, 0.001]),
    "epochs": hp.choice('epochs', [20, 50, 100]),  # we would also use early stopping
    "patience": hp.choice('patience', [5, 10]),  # early stopping patience
    # "optimizer": hp.choice('optimizer', [Adam, SGD, RMSProp]),  # https://docs.fast.ai/optimizer
    "optimizer": hp.choice('optimizer', [Adam]),
    # model parameters
    "n_layers": hp.choice('n_layers', [1, 2, 3, 4, 5]),
    "hidden_size": hp.choice('hidden_size', [50, 100, 200]),
    "bidirectional": hp.choice('bidirectional', [True, False])
}


# %%
def create_model_hypopt(params):
    try:
        # clear memory
        gc.collect()
        print("Trying params:", params)
        batch_size = params["batch_size"]

        # Create data loader
        tfms = [None, [TSRegression()]]
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        # set num_workers for memory bottleneck
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)

        # Create model
        global arch
        k = {
            'n_layers': params['n_layers'],
            'hidden_size': params['hidden_size'],
            'bidirectional': params['bidirectional']
        }
        model = create_model(arch, d=False, dls=dls)
        print(model.__class__.__name__)

        # Add a Sigmoid layer
        model = nn.Sequential(model, nn.Sigmoid())

        # Training the model
        learn = Learner(dls, model, metrics=[mae, rmse, mse, mape], opt_func=params['optimizer'])
        start = time.time()
        learn.fit_one_cycle(params['epochs'], lr_max=params['lr'],
                            cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=params['patience']))
        #learn.plot_metrics()
        elapsed = time.time() - start
        print(elapsed)

        vals = learn.recorder.values[-1]
        print(vals)
        # vals[0], vals[1], vals[2]
        # train loss, valid loss, accuracy
        val_loss = vals[1]

        # delete tmp variables
        del dls
        del model
        del learn
        return {'loss': val_loss, 'status': STATUS_OK}  # if accuracy use '-' sign, model is optional
    except:
        return {'loss': None, 'status': STATUS_FAIL}


#trials = Trials()
#best = fmin(create_model_hypopt,
#    space=search_space,
#    algo=tpe.suggest,
#    max_evals=max_evals,  # test trials
#    trials=trials)
#print("Best parameters:")
#print(space_eval(search_space, best))
#params = space_eval(search_space, best)

#with open(directory+str(model_name)+f'_best_params.txt', 'w') as f:
#    f.write(str(space_eval(search_space, best)))


params = {'batch_size': 32, 'bidirectional': False, 'epochs': 100, 'hidden_size': 50, 'lr': 0.001, 'n_layers': 5, 'optimizer': Adam, 'patience': 10}


for i in range(10):
    batch_size = params["batch_size"]
    tfms = [None, [TSRegression()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size], num_workers=0)
    k = {
        'n_layers': params['n_layers'],
        'hidden_size': params['hidden_size'],
        'bidirectional': params['bidirectional']
    }
    model = create_model(arch, d=False, dls=dls)
    print(str("\n### ")+model.__class__.__name__+str(" ###"))
    learn = Learner(dls, model, metrics=[mae, mse, rmse, mape], opt_func=params['optimizer'])
    start = time.time()
    learn.fit_one_cycle(params['epochs'], lr_max=params['lr'], cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.0, patience=params['patience']))
    training_time = time.time() - start

    save_training_time(i, training_time)
    save_metrics_plot(learn, X, y, i)
    save_default_metrics(learn, i)
    save_trained_model(learn, i)
    check_feature_importance(X, y)
    exit()

    #Prediction

    raw_preds, target, preds = learn.get_X_preds(X[splits[1]], y[splits[1]])


    # Plot the target and predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(target, label='Real')
    plt.plot(preds, label='Predicted', linestyle='dashed')
    plt.xlabel('Time Steps')
    plt.ylabel('Cassandra Write (Latency) ms')
    plt.title('Cassandra Latency Estimation (ms)')
    plt.legend()
    plt.grid(False)
    #plt.show()
    check_error(target, preds, index=i)
