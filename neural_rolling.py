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

TTM_MODEL_REVISION = '180-60-ft-l1-r2.1'
CONTEXT            = 180
PREDICTION         = 60
ROLLING            = 120
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

    print(f"Target columns: {target}")
    print(f"Control columns: {control_cols}")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of NaN values: {df.isnull().sum().sum()}")
    print(f"TSP num_input_channels: {tsp.num_input_channels}")
    print(f"TSP prediction_channel_indices: {tsp.prediction_channel_indices}")
    print(f"TSP exogenous_channel_indices: {tsp.exogenous_channel_indices}")

    finetune_model = TinyTimeMixerForPrediction.from_pretrained(
        'Exportado/TTM/output/finetuned',
        revision = TTM_MODEL_REVISION,
        num_input_channels = tsp.num_input_channels,
        decoder_mode = 'mix_channel',
        prediction_channel_indices = tsp.prediction_channel_indices,
        exogenous_channel_indices = tsp.exogenous_channel_indices,
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp, df, split_config, use_frequency_token = finetune_model.config.resolution_prefix_tuning
    )

    # Standard 60-day predictions
    standard_pipeline = TimeSeriesForecastingPipeline(
        finetune_model,
        device = DEVICE,
        feature_extractor = tsp,
        batch_size = 32,
        explode_forecasts = True,
    )

    print(f"Making standard {PREDICTION}-day predictions...")
    standard_forecast = standard_pipeline(df_test)

    def true_rolling_forecast_with_real_sst(model, historical_data, full_sst_data, tsp, target_cols, sst_cols, context_length=180, prediction_length=60, num_steps=2):
        """
        True rolling forecast using real future SST values
        """
        print(f"Starting rolling forecast with real SST for {num_steps} steps of {prediction_length} days each")
        
        # Start with historical data
        current_data = historical_data.copy().reset_index()
        all_predictions = []
        
        for step in range(num_steps):
            print(f"  Step {step + 1}/{num_steps}")
            
            # Use last context_length points as input
            if len(current_data) < context_length:
                print(f"    Warning: Not enough data for context (need {context_length}, have {len(current_data)})")
                break
                
            # Get context window
            context_data = current_data.tail(context_length).copy()
            
            # Determine future period for this step
            last_date = pd.to_datetime(context_data['Data'].iloc[-1])
            future_start = last_date + pd.Timedelta(days=1)
            future_end = last_date + pd.Timedelta(days=prediction_length)
            
            print(f"    Predicting from {future_start.date()} to {future_end.date()}")
            
            # Get real future SST data for this period
            future_dates = pd.date_range(start=future_start, end=future_end, freq='D')
            future_sst = full_sst_data[full_sst_data.index.isin(future_dates)].copy()
            
            if len(future_sst) < prediction_length:
                print(f"    Warning: Only found {len(future_sst)} SST values out of {prediction_length} needed")
            
            # Create prediction input: context + future period with real SST + NaN for power
            prediction_input = context_data.copy()
            
            # Add future rows with real SST but NaN power generation
            for date in future_dates:
                future_row = {'Data': date}
                
                # Add real SST values if available
                if date in future_sst.index:
                    for sst_col in sst_cols:
                        future_row[sst_col] = future_sst.loc[date, sst_col]
                else:
                    # Use last available SST values if date not found
                    last_sst = future_sst.iloc[-1] if len(future_sst) > 0 else context_data[sst_cols].iloc[-1]
                    for sst_col in sst_cols:
                        future_row[sst_col] = last_sst[sst_col]
                
                # Add NaN for power generation (to be predicted)
                for target_col in target_cols:
                    future_row[target_col] = np.nan
                
                prediction_input = pd.concat([prediction_input, pd.DataFrame([future_row])], ignore_index=True)
            
            # Make prediction
            pipeline = TimeSeriesForecastingPipeline(
                model,
                device=DEVICE,
                feature_extractor=tsp,
                explode_forecasts=True
            )
            
            try:
                # Predict (pipeline will fill in the NaN power values using context + real SST)
                full_prediction = pipeline(prediction_input)
                
                # Extract only the future predictions (last prediction_length rows)
                prediction = full_prediction.tail(prediction_length).copy()
                prediction['step'] = step + 1
                
                # Store prediction
                all_predictions.append(prediction.copy())
                
                # For next iteration: append predicted power + real SST to current_data
                prediction_to_append = prediction.drop(['step'], axis=1).copy()
                current_data = pd.concat([current_data, prediction_to_append], ignore_index=True)
                
                print(f"    Successfully predicted {len(prediction)} days, total data now: {len(current_data)} days")
                
            except Exception as e:
                print(f"    Error in prediction: {e}")
                break
        
        return pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    # Replace your existing rolling forecast call with this:
    print(f"Making rolling forecast with real future SST...")
    print(f"SST data available from {sst.index.min()} to {sst.index.max()}")
    print(f"Test data ends at {df_test['Data'].max()}")

    rolling_forecast_result = true_rolling_forecast_with_real_sst(
        finetune_model,
        df_test,  # Historical context
        sst,      # Full SST data (including future)
        tsp,
        target,   # Power generation columns to predict
        sst_cols, # SST columns (exogenous)
        context_length=CONTEXT,
        prediction_length=PREDICTION,
        num_steps=6  # 2 * 60 = 120 days total
    )

    print(f"Rolling forecast result shape: {rolling_forecast_result.shape}")

    # Continue with your existing code for concatenation and plotting
    if not rolling_forecast_result.empty:
        # Combine standard (60d) + rolling (120d) forecasts
        print("Combining standard and rolling forecasts...")
        
        # Process rolling forecast
        rolling_forecast_result = rolling_forecast_result.set_index('Data')
        rolling_pred_men = rolling_forecast_result.resample('ME').mean()
        rolling_pred_men.index = rolling_pred_men.index.to_period('M').to_timestamp()
        
        # Process standard forecast  
        standard_forecast = standard_forecast.set_index('Data')
        standard_pred_men = standard_forecast.resample('ME').mean()
        standard_pred_men.index = standard_pred_men.index.to_period('M').to_timestamp()
        
        # Combine both for extended forecast
        all_forecasts = pd.concat([standard_forecast, rolling_forecast_result], ignore_index=False)
        all_forecasts = all_forecasts[~all_forecasts.index.duplicated(keep='last')]  # Remove duplicates
        finetune_forecast = all_forecasts.reset_index()
        
    else:
        print("Rolling forecast failed, using only standard forecast")
        finetune_forecast = standard_forecast

    ##################################################### GRÁFICOS #####################################################
    
    finetune_forecast = finetune_forecast.set_index('Data')
    pred_men = finetune_forecast.resample('ME').mean().drop(columns=['step'])
    print(pred_men)
    pred_men.index = pred_men.index.to_period('M').to_timestamp()

    df_train.set_index('Data', inplace = True)
    df_train = df_train.resample('ME').mean()
    df_train.index = df_train.index.to_period('M').to_timestamp()

    df_valid.set_index('Data', inplace = True)
    df_valid = df_valid.resample('ME').mean()
    df_valid.index = df_valid.index.to_period('M').to_timestamp()

    df_test.set_index('Data', inplace = True)
    df_test = df_test.resample('ME').mean()
    df_test.index = df_test.index.to_period('M').to_timestamp()

    geracao_men = geracao[target].resample('ME').mean()
    geracao_men = geracao_men[geracao_men.index.year > 2024]
    geracao_men = pd.concat([df_test.tail(1), geracao_men])

    # y_pred = pred_men.reindex(df_test.index).dropna()
    # y_true = df_test.reindex(y_pred.index).dropna()

    # print(f'\nPrevisão:\n {y_pred}')
    # print(f'\nGeração observada:\n {y_true}')
    
    fig, axs = plt.subplots(len(target), 1, figsize = (6, len(target) * 3), sharex = True)
    for i, col in enumerate(pred_men.columns):
        # erro_r2 = r2_score(y_true[col], y_pred[col])

        axs[i].plot(df_train.index, df_train[col], label = 'Treino', c = '#000000')
        axs[i].plot(df_valid.index, df_valid[col], label = 'Validação', c = '#000000', alpha = .5)
        axs[i].plot(df_test.index, df_test[col], label = 'Teste', c = '#000000', alpha = .33)
        axs[i].plot(geracao_men.index, geracao_men[col], label = 'Observado', c = "#000000", alpha = .33, ls = '--')
        axs[i].plot(pred_men.index, pred_men[col], label = 'Previsão', c = "#4A7AFF", lw = .8)

        # axs[i].text(
        #     0.02, .98, 
        #     f'R² = {erro_r2:.4f}',
        #     transform = axs[i].transAxes, verticalalignment = 'top', fontsize = 7,
        #     bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none')
        # )

        axs[i].set_ylabel(f'{col.replace('_', ' ')} [MWmed]')
        axs[i].xaxis.set_major_locator(mdates.AutoDateLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axs[0].legend(loc = 'upper center', bbox_to_anchor = (0.5, 1.15), ncol = 4, frameon = False)
    axs[-1].set_xlabel('Série temporal')
    plt.tight_layout()
    plt.savefig(f'Graficos/Neural/{target[0]}_rolling_{CONTEXT}.png')
    # plt.show()