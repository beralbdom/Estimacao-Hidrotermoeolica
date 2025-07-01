import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from alive_progress import alive_bar
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse_error
from sklearn.preprocessing import StandardScaler

import matplotlib_config

geracao = (
    pd.read_csv(
        'Exportado/geracao_fontes_diario_MWmed.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
)

carga = (
    pd.read_csv(
        'Exportado/carga_subsist_diario_MWmed.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
)

vazoes = (
    pd.read_csv(
        'Exportado/dados_hidrologicos_diarios.csv', 
        parse_dates = ['Data'])
    .set_index('Data')
)

enso = (
    pd.read_csv(
        'Exportado/ECMWF/derived-era5-single-levels-daily-statistics_sea_surface_temperature_reanalysis.csv', 
        parse_dates = ['Data'])
        .set_index('Data')
)

geracao = geracao.fillna(0) # sem geração eólica e fotovoltaica no início dos anos 2000
geracao = geracao.drop(columns = [cols for cols in geracao.columns if 'Fotovoltaica' in cols or 'Outras' in cols]) 

enso['ano'] = enso.index.year
enso['mes'] = enso.index.month
enso['dia_mes'] = enso.index.day
enso['dia_semana'] = enso.index.dayofweek
enso['dia_ano'] = enso.index.dayofyear
enso['quadrimestre'] = enso.index.quarter

# lags = [15, 30, 60]
# for lag in lags: 
#     for fonte in geracao.columns: enso[f'geracao_{fonte}_lag_{lag}'] = geracao[fonte].shift(lag)

dataset = pd.concat([geracao, carga, enso], axis = 1).dropna()
dataset = dataset[dataset.index.year > 2009]
dataset = dataset[dataset.index.year < 2025]

# dataset = dataset.resample('ME').mean()

target_cols = geracao.columns
exog_cols = enso.columns.append(carga.columns)
print(f'Colunas objetivo: {target_cols}')
print(f'Colunas exógenas: {exog_cols}')
# print(dataset.head())

# Normalização dos dados
scaler = StandardScaler()
dataset[exog_cols] = scaler.fit_transform(dataset[exog_cols])

X_train, X_test, y_train, y_test = train_test_split(
    dataset[exog_cols], dataset[target_cols], 
    test_size = .3, shuffle = False
)

modelo = RandomForestRegressor(n_estimators = 1000,
                               random_state = 69,
                               max_features = .5,
                            #    max_depth = 30,
                            #    min_samples_split = 5,
                            #    min_samples_leaf = 2,
                            #    bootstrap = True,
#                                criterion = 'squared_error',
                               n_jobs = -1,
                               verbose = 1)

# Melhores parâmetros:  {'bootstrap': True, 
# 'max_depth': 30, 'max_features': 1.0, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 1500}

# param_grid = {
#     'n_estimators': [500, 1000, 1500],
#     'max_features': ['sqrt', 'log2', 1.0],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# tscv = TimeSeriesSplit(n_splits = 5)

# grid_search = GridSearchCV(
#     estimator = modelo,
#     param_grid=param_grid,
#     cv=tscv,
#     n_jobs=-1,  # Use all available CPU cores
#     verbose=2,  # Show progress
#     scoring='neg_root_mean_squared_error'
# )

# reg = grid_search.fit(X_train, y_train)
# print("\nMelhores parâmetros: ", reg.best_params_)
# print("Melhor score (Neg RMSE): ", reg.best_score_)

reg = modelo.fit(X_train, y_train)
# print(f'Caminho de decisão: {reg.estimators_[0].tree_}')

y_pred = reg.predict(X_test)
y_pred = pd.DataFrame(y_pred, columns = target_cols, index = y_test.index) 
y_pred[y_pred < 0] = 0
# print(y_test)

y_pred = y_pred.resample('ME').mean()
y_test = y_test.resample('ME').mean()
y_train = y_train.resample('ME').mean()

layout = [['a', 'b']]
for col in target_cols:
    r2 = r2_score(y_test[col], y_pred[col])
    mse = mse_error(y_test[col], y_pred[col])

    fig, ax = plt.subplot_mosaic(layout, figsize = (7, 2.5), width_ratios= [1, 2])

    ax['a'].scatter(y_test[col], y_pred[col], s = 1, color = "#FF5B5B", alpha = .5, label = 'Previsto')
    ax['a'].plot([y_test[col].min(), y_test[col].max()], [y_test[col].min(), y_test[col].max()], 
               color = "#202020", linewidth = .66, ls = '--', label = 'Ideal')
    
    ax['a'].set_xticks(np.linspace(y_test[col].min(), y_test[col].max(), num = 5))
    ax['a'].set_yticks(np.linspace(y_test[col].min(), y_test[col].max(), num = 5))
    ax['a'].legend(loc = 'upper center', bbox_to_anchor = (.5, 1.15), ncol = 2, frameon = False, fancybox = False)
    ax['a'].ticklabel_format(axis = 'both', style = 'sci', scilimits = (3, 3))
    ax['a'].set_xlabel('Real')
    ax['a'].set_ylabel('Previsto')
    
    ax['b'].plot(y_train.index, y_train[col], label = 'Treino', color = "#43AAFF", linewidth = .66)
    ax['b'].plot(y_test.index, y_test[col], label = 'Real', color = "#969696", linewidth = .66, ls = '--')
    ax['b'].plot(y_pred.index, y_pred[col], label = 'Previsto', color = "#FF5B5B", linewidth = .66)
    
    ax['b'].legend(loc = 'upper center', bbox_to_anchor = (.5, 1.15), ncol = 3, frameon = False, fancybox = False)
    ax['b'].set_xlabel('Série histórica')
    ax['b'].set_ylabel(f'Geração {col.replace('_', ' ')} (MWMed)')
    ax['b'].ticklabel_format(axis = 'y', style = 'sci', scilimits = (3, 3))

    ax['b'].text(.02, .95, (f'R² = {r2:.3f}\nMSE = {mse:.2E}'), 
                            transform = ax['b'].transAxes, verticalalignment = 'top', fontsize = 7,
                            bbox = dict(boxstyle = 'square', facecolor = 'white', edgecolor = 'none'))
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'LateX/figuras/nlinear/rf_{col}.svg', bbox_inches = 'tight')