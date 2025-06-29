import os
import math
import torch
import random

import pandas as pd
import numpy as np
# import torch_directml

from transformers import set_seed
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit import RecursivePredictor, RecursivePredictorConfig
from matplotlib import rc, pyplot as plt, dates as mdates
from sklearn.metrics import r2_score

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
)


# Configurações de plotagem --------------------------------------------------------------------------------------------
rc('font', family = 'serif', size = 8)
rc('grid', linestyle = '--', alpha = .5)
rc('axes', axisbelow = True, grid = True)
rc('lines', linewidth = .8, markersize = 1.5)
rc('axes.spines', top = False, right = False, left = True, bottom = True)

# Definições do TTM ----------------------------------------------------------------------------------------------------
# torch.Tensor.clamp_min = lambda self, v: self.clamp(min=v.item() if isinstance(v, torch.Tensor) else v)
# _dml = torch_directml.device()
# print(Using DirectML device:, _dml)

TTM_MODEL_REVISION = '180-60-ft-l1-r2.1'
CONTEXT            = 180
PREDICTION         = 60
OUT_DIR            = 'Exportado/TTM'
DEVICE             = 'cpu'
# DEVICE             = _dml
SEED               = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

# Entrada de dados -----------------------------------------------------------------------------------------------------
geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'],
    )
    .set_index('Data')
    .fillna(0)
    # .resample('D').mean()
)

freq = 'D'
tempo = 'Data'
fontes = ['Hidráulica', 'Eólica', 'Fotovoltaica', 'Térmica', 'Outras']
targets = np.zeros((len(fontes)), dtype = object)
subsistemas = ['N', 'NE', 'S', 'SE']

for i, fonte in enumerate(fontes):
    targets[i] = []
    targets[i].append(f'{fonte}')
    for subsistema in subsistemas:
        targets[i].append(f'{fonte}_{subsistema}')

print(f'\nTargets:\n {targets}')

for target in targets:
    df = geracao[target]
    df = df[df.index.year <= 2024]
    df = df.reset_index()
    # print(f'\nX:\n {df}')

    # Zero shot
    split_config = {
        'train': [0, 0.6],      # 0% to 60%
        'valid': [0.6, 0.8],    # 60% to 80%  
        'test': [0.8, 1.0]      # 80% to 100%
    }

    df_train, df_valid, df_test = prepare_data_splits(
        df,
        context_length = CONTEXT, 
        split_config = split_config,
    )

    # print(f'\nTrain:\n {df_train}')

    tsp = TimeSeriesPreprocessor(
        target_columns = target,
        timestamp_column = tempo,
        context_length = CONTEXT,
        prediction_length = PREDICTION,
        scaling = True,
        scaling_type = 'standard',
        freq = freq,
    )

    zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
        'ibm-granite/granite-timeseries-ttm-r2',
        revision = TTM_MODEL_REVISION,
        num_input_channels = len(target),
    )

    trained_tsp = tsp.train(df_train)
    # print(f'\nTrained TSP:\n {trained_tsp}')

    pipeline = TimeSeriesForecastingPipeline(
        zeroshot_model,
        device = DEVICE,
        feature_extractor = tsp,
        batch_size = 512,
        explode_forecasts = True,
        freq = freq,
    )

    forecast = pipeline(df_test)

    # Resultados
    print(forecast)

    ##################################################### GRÁFICOS #####################################################
    
    forecast = forecast.set_index('Data')
    # pred_men = zeroshot_forecast.resample('ME').mean()
    # pred_men.index = pred_men.index.to_period('M').to_timestamp()

    df_train.set_index('Data', inplace = True)
    # df_train = df_train.resample('ME').mean()
    # df_train.index = df_train.index.to_period('M').to_timestamp()

    df_test.set_index('Data', inplace = True)
    # df_test = df_test.resample('ME').mean()
    # df_test.index = df_test.index.to_period('M').to_timestamp()

    # geracao_men = geracao[target].resample('ME').mean()
    geracao_men = geracao_men[geracao_men.index.year > 2024]
    geracao_men = pd.concat([df_test.tail(1), geracao_men])

    # y_pred = pred_men.reindex(df_test.index).dropna()
    # y_true = df_test.reindex(y_pred.index).dropna()

    # print(f'\nPrevisão:\n {y_pred}')
    # print(f'\nGeração observada:\n {y_true}')
    
    fig, axs = plt.subplots(len(target), 1, figsize = (6, len(target) * 3), sharex = True)
    for i, col in enumerate(forecast.columns):
        # erro_r2 = r2_score(y_true[col], y_pred[col])

        axs[i].plot(df_train.index, df_train[col], label = 'Treino', c = '#000000')
        axs[i].plot(df_test.index, df_test[col], label = 'Teste', c = '#000000', alpha = .33)
        axs[i].plot(geracao_men.index, geracao_men[col], label = 'Observado', c = "#000000", alpha = .33, ls = '--')
        axs[i].plot(forecast.index, forecast[col], label = 'Previsão', c = "#4A7AFF", lw = .8)

        # axs[i].text(
        #     0.02, .98, 
        #     f'R² = {erro_r2:.4f}',
        #     transform = axs[i].transAxes, verticalalignment = 'top', fontsize = 7,
        #     bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none')
        # )

        axs[i].set_ylabel(f'{col.replace('_', ' ')} [MWmed]')
        axs[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    axs[0].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 4, frameon = False)
    axs[-1].set_xlabel('Série temporal')
    plt.tight_layout()
    plt.savefig(f'Graficos/Neural/{target[0]}_zeroshot_{freq}{CONTEXT}.png')
    plt.show()