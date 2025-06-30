import numpy as np
import pandas as pd
import os
import math
import torch
import random

import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error as mse_error
from sklearn.preprocessing import StandardScaler

from transformers import set_seed
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit import RecursivePredictor, RecursivePredictorConfig

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
)

import matplotlib_config

TTM_MODEL_REVISION = '90-30-ft-r2.1'
CONTEXT            = 90
PREDICTION         = 30
OUT_DIR            = 'Exportado/TTM'
DEVICE             = 'cpu'
SEED               = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

freq = 'W'
tempo = 'Data'

geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
    .fillna(0)
    .resample(freq).mean()
)

geracao_2025 = geracao[geracao.index.year == 2025]
geracao = geracao[geracao.index.year < 2025]
geracao = geracao.drop(columns = [cols for cols in geracao.columns if 'Fotovoltaica' in cols or 'Outras' in cols])
targets = geracao.columns.tolist()
geracao = geracao.reset_index()

split_config = {
    'train': [0, 0.6],      # 0% to 60%
    'valid': [0.6, 0.8],    # 60% to 80%
    'test': [0.8, 1.0]      # 80% to 100%
}

df_train, df_valid, df_test = prepare_data_splits(
    geracao,
    context_length = CONTEXT, 
    split_config = split_config,
)

tsp = TimeSeriesPreprocessor(
    target_columns = targets,
    timestamp_column = tempo,
    context_length = CONTEXT,
    prediction_length = PREDICTION,
    scaling = True,
    scaling_type = 'standard',
    freq = freq,
)

zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    # 'ibm-granite/granite-timeseries-ttm-r2',
    'Exportado\TTM\\finetune',
    revision = TTM_MODEL_REVISION,
    num_input_channels = len(targets),
)

trained_tsp = tsp.train(df_train)

pipeline = TimeSeriesForecastingPipeline(
    zeroshot_model,
    device = DEVICE,
    feature_extractor = tsp,
    batch_size = 512,
    explode_forecasts = True,
    freq = freq,
)

df_pred = pipeline(df_test)

df_pred = df_pred.set_index('Data')
df_test = df_test.set_index('Data')
df_train = df_train.set_index('Data')
df_valid = df_valid.set_index('Data')

df_pred = df_pred.groupby(df_pred.index).mean()

idx_comum = df_test.index.intersection(df_pred.index)
df_test_ = df_test.loc[idx_comum]
df_pred_ = df_pred.loc[idx_comum]

df_pred = df_pred.resample('ME').mean()
df_test = df_test.resample('ME').mean()
df_train = df_train.resample('ME').mean()
df_valid = df_valid.resample('ME').mean()
df_test_ = df_test_.resample('ME').mean()
df_pred_ = df_pred_.resample('ME').mean()

geracao_2025 = geracao_2025.resample('ME').mean()
geracao_2025 = pd.concat([df_test.tail(1), geracao_2025])

df_train = pd.concat([df_train, df_valid]).drop_duplicates(keep = 'last')

print(df_pred)
print(df_test)

layout = [['a', 'b']]
for col in targets:
    r2 = r2_score(df_test_[col], df_pred_[col])
    mse = mse_error(df_test_[col], df_pred_[col])

    fig, ax = plt.subplot_mosaic(layout, figsize = (9, 3), width_ratios= [1, 2])

    ax['a'].scatter(df_test_[col], df_pred_[col], s = 1, color = "#FF5B5B", alpha = 1, label = 'Previsto')
    ax['a'].plot([df_test_[col].min(), df_test_[col].max()], [df_test_[col].min(), df_test_[col].max()], 
               color = "#202020", linewidth = .66, ls = '--', label = 'Ideal')
    
    ax['a'].set_xticks(np.linspace(df_test[col].min(), df_test[col].max(), num = 5))
    ax['a'].set_yticks(np.linspace(df_test[col].min(), df_test[col].max(), num = 5))
    ax['a'].legend(loc = 'upper center', bbox_to_anchor = (.5, 1.15), ncol = 2, frameon = False, fancybox = False)
    ax['a'].ticklabel_format(axis = 'both', style = 'sci', scilimits = (3, 3))
    ax['a'].set_xlabel('Real')
    ax['a'].set_ylabel('Previsto')
    
    ax['b'].plot(df_train.index, df_train[col], label = 'Treino', color = "#43AAFF", linewidth = .66)
    ax['b'].plot(df_test.index, df_test[col], label = 'Real', color = "#969696", linewidth = .66)
    ax['b'].plot(df_pred.index, df_pred[col], label = 'Previsto', color = "#FF5B5B", linewidth = .66)
    ax['b'].plot(geracao_2025.index, geracao_2025[col], color = "#969696", linewidth = .66)
    ax['b'].xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax['b'].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax['b'].legend(loc = 'upper center', bbox_to_anchor = (.5, 1.15), ncol = 5, frameon = False, fancybox = False)
    ax['b'].set_xlabel('Série histórica')
    ax['b'].set_ylabel(f'Geração {col.replace('_', ' ')} (MWMed)')
    ax['b'].ticklabel_format(axis = 'y', style = 'sci', scilimits = (3, 3))

    ax['b'].text(.02, .95, (f'R² = {r2:.3f}\nRMSE = {mse:.2E}'), 
                            transform = ax['b'].transAxes, verticalalignment = 'top', fontsize = 7,
                            bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none'))
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'Graficos/Neural/FineTune/{col}_{freq}{CONTEXT}-{PREDICTION}.png', bbox_inches = 'tight')