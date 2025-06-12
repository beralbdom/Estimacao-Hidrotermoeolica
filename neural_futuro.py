import os
import math
import torch
import random

import pandas as pd
import numpy as np
# import torch_directml


# fazer caso passando todas as gerações de uma vez para verificar se o modelo aprende as relações entre as fontes
# passar variaveis de controle pós 2025 e deixar as fontes até 2024, com o objetivo de prever as fontes de 2025

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    TrackingCallback,
    count_parameters,
    get_datasets,
)

from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit import RecursivePredictor, RecursivePredictorConfig
from matplotlib import rc, pyplot as plt, dates as mdates
from sklearn.metrics import r2_score

# Configurações de plotagem --------------------------------------------------------------------------------------------
rc('font', family = 'serif', size = 8)
rc('grid', linestyle = '--', alpha = .5)
rc('axes', axisbelow = True, grid = True)
rc('lines', linewidth = .8, markersize = 1.5)
rc('axes.spines', top = False, right = False, left = True, bottom = True)

# Definições do TTM ----------------------------------------------------------------------------------------------------
# torch.Tensor.clamp_min = lambda self, v: self.clamp(min=v.item() if isinstance(v, torch.Tensor) else v)
# _dml = torch_directml.device()

TTM_MODEL_REVISION = '180-60-ft-l1-r2.1'
CONTEXT            = 180
PREDICTION         = 60
OUT_DIR            = 'Exportado/TTM'
MODELO_DIR         = 'Exportado/TTM/output/finetuned_diario_180_60' # 'ibm-granite/granite-timeseries-ttm-r2'
DEVICE             = 'cpu'
# DEVICE             = _dml
SEED               = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

torch.set_num_threads(24)
torch.set_num_interop_threads(24)

# Entrada de dados -----------------------------------------------------------------------------------------------------
# carga = (
#     pd.read_csv(
#         'Exportado/carga_subsist_diario_MWmed.csv', 
#         parse_dates = ['Data'],
#     )
#     .set_index('Data')
# )

sst = (
    pd.read_csv(
        'Exportado/ECMWF/derived-era5-single-levels-daily-statistics_sea_surface_temperature_reanalysis.csv', 
        parse_dates = ['Data'],
    )
    .set_index('Data')
    # .resample('W').mean()
)

geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'],
    )
    .set_index('Data')
    .fillna(0)
    # .resample('W').mean()
)

sst_cols = sst.columns.tolist()
# carga_cols = carga.columns.tolist()
freq = 'D'
tempo = 'Data'
control_cols = sst_cols
fontes = ['Hidráulica', 'Eólica', 'Fotovoltaica', 'Térmica', 'Outras']
subsistemas = ['N', 'NE', 'S', 'SE']

geracao_passado = geracao.loc[geracao.index.year < 2024]
dataset = pd.concat([sst, geracao_passado], axis = 1)

print(f'\nDataset:\n {dataset}')

targets = np.zeros((len(fontes)), dtype = object)
for i, fonte in enumerate(fontes):
    targets[i] = []
    targets[i].append(f'{fonte}')
    for subsistema in subsistemas:
        targets[i].append(f'{fonte}_{subsistema}')

print(f'\nTargets:\n {targets}')

for target in targets:
    df = dataset[target + sst_cols]
    df = df.reset_index()
    print(f'\nX:\n {df}')

    tsp = TimeSeriesPreprocessor(
        timestamp_column = tempo,
        target_columns = target,
        control_columns = control_cols,
        encode_categorical = False,
        context_length = CONTEXT,
        prediction_length = PREDICTION,
        scaling = True,
        scaler_type = 'standard',
        freq = freq,
    )

    tsp.train(df)

    print(f"Target columns: {target}")
    print(f"Control columns: {control_cols}")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of NaN values: {df.isnull().sum().sum()}")
    print(f"TSP num_input_channels: {tsp.num_input_channels}")
    print(f"TSP prediction_channel_indices: {tsp.prediction_channel_indices}")
    print(f"TSP exogenous_channel_indices: {tsp.exogenous_channel_indices}")

    finetune_model = TinyTimeMixerForPrediction.from_pretrained(
        MODELO_DIR,
        revision = TTM_MODEL_REVISION,
        num_input_channels = tsp.num_input_channels,
        decoder_mode = 'mix_channel',
        prediction_channel_indices = tsp.prediction_channel_indices,
        exogenous_channel_indices = tsp.exogenous_channel_indices,
        fcm_context_length = 180,
        fcm_use_mixer = True,
        # fcm_mix_layers = 1,
        # enable_forecast_channel_mixing = False,
        # fcm_prepend_past = True,
    )

    pipeline = TimeSeriesForecastingPipeline(
        finetune_model,
        device = DEVICE,
        feature_extractor = tsp,
        batch_size = 64,
        explode_forecasts = True,
        freq = freq,
    )

    forecast = pipeline(df)
    forecast.head()

    ##################################################### GRÁFICOS #####################################################
    
    forecast = forecast.set_index('Data')
    pred_men = forecast.resample('ME').mean()
    pred_men.index = pred_men.index.to_period('M').to_timestamp()

    geracao_men = geracao[target].resample('ME').mean()
    geracao_men.index = geracao_men.index.to_period('M').to_timestamp()
    geracao_obs = geracao_men.loc[geracao_men.index.year > 2024]

    y_pred, y_true = pred_men.align(geracao_men, join = 'inner')

    print(f'\nPrevisão:\n {y_pred}')
    print(f'\nGeração observada:\n {y_true}')
    
    fig, axs = plt.subplots(len(target), 1, figsize = (6, len(target) * 3), sharex = False)
    for i, col in enumerate(pred_men.columns):
        erro_r2 = r2_score(y_true[col], y_pred[col])

        axs[i].plot(geracao_obs.index, geracao_obs[col], label = 'Observado', c = "#000000", alpha = .33, ls = '--')
        axs[i].plot(pred_men.index, pred_men[col], label = 'Previsão', c = "#4A7AFF", lw = .8)

        axs[i].text(
            0.02, .98, 
            f'R² = {erro_r2:.4f}',
            transform = axs[i].transAxes, verticalalignment = 'top', fontsize = 7,
            bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none')
        )

        axs[i].set_ylabel(f'{col.replace('_', ' ')} [MWmed]')
        axs[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axs[0].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 2, frameon = False)
    axs[-1].set_xlabel('Série temporal')
    plt.tight_layout()
    plt.savefig(f'Graficos/Neural/{target[0]}_finetune_{freq}{CONTEXT}_CompletoSST.png')
    # plt.show()