import numpy as np
import pandas as pd
import math, torch, random
import matplotlib.dates as mdates
import matplotlib_config

from matplotlib import pyplot as plt
from alive_progress import alive_bar
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_error
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from tsfm_public.toolkit.time_series_preprocessor import prepare_data_splits
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public import (
    TimeSeriesForecastingPipeline,
    TrackingCallback,
    TimeSeriesPreprocessor,
    TinyTimeMixerForPrediction,
    get_datasets,
)

TTM_MODEL  = '512-96-ft-r2.1'
CONTEXT    = 512
PREDICTION = 96
OUT_DIR    = 'Exportado/TTM'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED       = 1337

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
set_seed(SEED)

freq = 'D'
tempo = 'Data'

geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
    .fillna(0)
    # .resample(freq).mean()
)

enso = (
    pd.read_csv(
        'Exportado/ECMWF/derived-era5-single-levels-daily-statistics_sea_surface_temperature_reanalysis.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
    # .resample(freq).mean()
)

carga = (
    pd.read_csv(
        'Exportado/carga_subsist_diario_MWmed.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
    # .resample(freq).mean()
)

geracao = geracao.drop(columns = [cols for cols in geracao.columns if 'Fotovoltaica' in cols or 'Outras' in cols])
target_cols = geracao.columns.tolist()
exog_cols = enso.columns.append(carga.columns).tolist()

dataset = pd.concat([geracao, enso, carga], axis = 1).dropna()
dataset = dataset[dataset.index.year < 2025]
dataset = dataset.reset_index()

print(f'Dataset:\n{dataset}')

split_config = {"train": 0.7, "test": 0.2}

df_train, df_valid, df_test = prepare_data_splits(
    dataset,
    context_length = CONTEXT, 
    split_config = split_config,
)

tsp = TimeSeriesPreprocessor(
    timestamp_column = tempo,
    target_columns = target_cols,
    control_columns = exog_cols,
    encode_categorical = False,
    context_length = CONTEXT,
    prediction_length = PREDICTION,
    scaling = True,
    scaler_type = 'standard',
    freq = freq,
)

finetune_model = TinyTimeMixerForPrediction.from_pretrained(
    'ibm-granite/granite-timeseries-ttm-r2',
    revision = TTM_MODEL,
    num_input_channels = tsp.num_input_channels,
    decoder_mode = 'mix_channel',
    prediction_channel_indices = tsp.prediction_channel_indices,
    exogenous_channel_indices = tsp.exogenous_channel_indices,
    # fcm_context_length = 60,
    fcm_use_mixer = True,
    fcm_mix_layers = 3,
    enable_forecast_channel_mixing = True,
    fcm_prepend_past = True,
)

train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp, dataset, split_config, use_frequency_token = finetune_model.config.resolution_prefix_tuning
)

num_epochs = 500
batch_size = 64
learning_rate = 5e-5

# learning_rate, finetune_model = optimal_lr_finder(
#     finetune_model,
#     train_dataset,
#     batch_size = batch_size,
#     enable_prefix_tuning = True,
# )

tracking_callback = TrackingCallback()
optimizer = AdamW(finetune_model.parameters(), lr = learning_rate)

print(f'Quantidade de valores faltantes: {dataset.isna().sum().sum()}')
print(f'Variáveis alvo: {target_cols}')
print(f'Variáveis exógenas: {exog_cols}')
print(f'Learning rate: {learning_rate}')
print(f'Batch size: {batch_size}')
print(f'Número de épocas: {num_epochs}')

finetune_forecast_args = TrainingArguments(
    output_dir = f'{OUT_DIR}/output',
    overwrite_output_dir = True,
    learning_rate = learning_rate,
    # warmup_steps = 10,
    num_train_epochs = num_epochs,
    do_eval = True,
    eval_strategy = 'epoch',
    eval_steps = 1,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    dataloader_num_workers = 0,
    report_to = None,
    save_strategy = 'epoch',
    save_steps = 1,
    logging_steps = 1,
    save_total_limit = 5,
    logging_strategy = 'epoch',
    logging_dir = f'{OUT_DIR}/logs',
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss",
    greater_is_better = False,
    use_cpu = DEVICE == 'cpu',
)

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience = 30,
    early_stopping_threshold = 1e-6,
)

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

finetune_forecast_trainer.train()
finetune_model.save_pretrained(f'{OUT_DIR}/finetune_{freq}{CONTEXT}-{PREDICTION}')