import os
import math
import torch
import random

import pandas as pd
import numpy as np
# import torch_directml

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

TTM_MODEL_REVISION = '512-96-ft-r2.1'
CONTEXT            = 512
PREDICTION         = 96
OUT_DIR            = 'Exportado/TTM'
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
)

geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'],
    )
    .set_index('Data')
    .fillna(0)
)

sst_cols = sst.columns.tolist()
# carga_cols = carga.columns.tolist()
tempo = 'Data'
control_cols = sst_cols
fontes = ['Hidráulica', 'Eólica', 'Fotovoltaica', 'Térmica', 'Outras']
subsistemas = ['N', 'NE', 'S', 'SE']

dataset = pd.concat([sst, geracao], axis = 1)
dataset = dataset[dataset.index.year <= 2024]

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

    split_config = {'train': .5, 'test': .5}

    df_train, df_valid, df_test = prepare_data_splits(
        df,
        context_length = CONTEXT, 
        split_config = split_config,
    )

    tsp = TimeSeriesPreprocessor(
        timestamp_column = tempo,
        target_columns = target,
        control_columns = control_cols,
        encode_categorical = False,
        context_length = CONTEXT,
        prediction_length = PREDICTION,
        scaling = True,
        scaler_type = 'standard',
    )

    finetune_model = TinyTimeMixerForPrediction.from_pretrained(
        'ibm-granite/granite-timeseries-ttm-r2',
        revision = TTM_MODEL_REVISION,
        num_input_channels = tsp.num_input_channels,
        decoder_mode = 'mix_channel',
        prediction_channel_indices = tsp.prediction_channel_indices,
        exogenous_channel_indices = tsp.exogenous_channel_indices,
        fcm_context_length = 60,
        fcm_use_mixer = True,
        fcm_mix_layers = 2,
        enable_forecast_channel_mixing = True,
        fcm_prepend_past = True,
    )

    print(
    "Number of params before freezing backbone",
    count_parameters(finetune_model),
    )

    # Freeze the backbone of the model
    for param in finetune_model.backbone.parameters():
        param.requires_grad = False

    # Count params
    print(
        "Number of params after freezing the backbone",
        count_parameters(finetune_model),
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp, df, split_config, use_frequency_token = finetune_model.config.resolution_prefix_tuning
    )
    print(f"Data lengths: train = {len(train_dataset)}, val = {len(valid_dataset)}, test = {len(test_dataset)}")

    # Important parameters
    learning_rate = 0.000298  # 0.000298364724028334
    num_epochs = 200  # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
    batch_size = 256

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir = os.path.join(OUT_DIR, "output"),
        overwrite_output_dir = True,
        learning_rate = learning_rate,
        num_train_epochs = num_epochs,
        do_eval = True,
        eval_strategy = "epoch",
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        # gradient_accumulation_steps = 2,
        dataloader_num_workers = 0,
        report_to = None,
        save_strategy = "epoch",
        logging_strategy = "epoch",
        save_total_limit = 1,
        logging_dir = os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end = True,                # Load the best model when training ends
        metric_for_best_model = "eval_loss",          # Metric to monitor for early stopping
        greater_is_better = False,
        use_cpu = True,
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience = 10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold = 0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs = num_epochs,
        steps_per_epoch = math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model = finetune_model,
        args = finetune_forecast_args,
        train_dataset = train_dataset,
        eval_dataset = valid_dataset,
        callbacks = [early_stopping_callback, tracking_callback],
        optimizers = (optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()

    pipeline = TimeSeriesForecastingPipeline(
        finetune_model,
        device = DEVICE,
        feature_extractor = tsp,
        batch_size = 512,
        explode_forecasts = True,
    )

    # Make a forecast on the target column given the input data.
    finetune_forecast = pipeline(df_test)
    finetune_forecast.head()

    ##################################################### GRÁFICOS #####################################################
    
    finetune_forecast = finetune_forecast.set_index('Data')
    pred_men = finetune_forecast.resample('ME').mean()
    pred_men.index = pred_men.index.to_period('M').to_timestamp()

    df_train.set_index('Data', inplace = True)
    df_train = df_train.resample('ME').mean()
    df_train.index = df_train.index.to_period('M').to_timestamp()

    df_test.set_index('Data', inplace = True)
    df_test = df_test.resample('ME').mean()
    df_test.index = df_test.index.to_period('M').to_timestamp()

    geracao_men = geracao[target].resample('ME').mean()
    geracao_men = geracao_men[geracao_men.index.year > 2024]
    geracao_men = pd.concat([df_test.tail(1), geracao_men])

    y_pred = pred_men.reindex(df_test.index).dropna()
    y_true = df_test.reindex(y_pred.index).dropna()

    # print(f'\nPrevisão:\n {y_pred}')
    # print(f'\nGeração observada:\n {y_true}')
    
    fig, axs = plt.subplots(len(target), 1, figsize = (6, len(target) * 3), sharex = True)
    for i, col in enumerate(pred_men.columns):
        erro_r2 = r2_score(y_true[col], y_pred[col])

        axs[i].plot(df_train.index, df_train[col], label = 'Treino', c = '#000000')
        axs[i].plot(df_test.index, df_test[col], label = 'Teste', c = '#000000', alpha = .33)
        axs[i].plot(geracao_men.index, geracao_men[col], label = 'Observado', c = "#000000", alpha = .33, ls = '--')
        axs[i].plot(pred_men.index, pred_men[col], label = 'Previsão', c = "#4A7AFF", lw = .8)

        axs[i].text(
            0.02, .98, 
            f'R² = {erro_r2:.4f}',
            transform = axs[i].transAxes, verticalalignment = 'top', fontsize = 7,
            bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none')
        )

        axs[i].set_ylabel(f'{col.replace('_', ' ')} [MWmed]')
        axs[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    axs[0].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 4, frameon = False)
    axs[-1].set_xlabel('Série temporal')
    plt.tight_layout()
    plt.savefig(f'Graficos/Neural/{target[0]}_finetune_{CONTEXT}.png')
    # plt.show()