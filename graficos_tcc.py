import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import rc, pyplot as plt, dates as mdates, patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from statsmodels.tsa.seasonal import MSTL

# Configurações de plotagem
rc('font', family = 'serif', size = 8)                                                                                  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
rc('grid', linestyle = '--', alpha = 0.5)
rc('axes', axisbelow = True, grid = True)
rc('lines', linewidth = .5, markersize = 1.5)
rc('axes.spines', top = False, right = False, left = True, bottom = True)


# Decomposição de séries temporais ----------------------------------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_mensal_MWmed.csv', parse_dates=['Data']).set_index('Data')
# geracao = geracao.fillna(0)  # Preenchendo valores ausentes com zero
# for col in ['Hidráulica', 'Eólica', 'Térmica']:
#     geracao[col] = StandardScaler().fit_transform(geracao[[col]])  # Normalizando a série de geração hidráulica
#     decomposicao = MSTL(geracao[col]).fit()

#     fig, ax = plt.subplots(4, 1, figsize=(6, 4), sharex=True)
#     ax[0].plot(decomposicao.observed, color="#4596F3", label='Observado')
#     # ax[0].set_ylabel('Observado')
#     # ax[0].yaxis.set_label_position("right")
#     ax[1].plot(decomposicao.trend, color="#ff5656", label='Tendência')
#     # ax[1].set_ylabel('Tendência')
#     # ax[1].yaxis.set_label_position("right")
#     ax[2].plot(decomposicao.seasonal, color="#FFA722", label='Sazonalidade')
#     # ax[2].set_ylabel('Sazonalidade')
#     # ax[2].yaxis.set_label_position("right")
#     ax[3].plot(decomposicao.resid, color="#47A730", label='Resíduo')
#     # ax[3].set_ylabel('Resíduo')
#     ax[3].set_xlabel('Série histórica')
#     # ax[3].yaxis.set_label_position("right")
#     ax[3].xaxis.set_major_locator(mdates.YearLocator(base=4))
#     ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#     handles, labels = [], []
#     for axis in ax:
#         h, l = axis.get_legend_handles_labels()
#         handles.extend(h)
#         labels.extend(l)
#     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)

#     plt.tight_layout()
#     plt.savefig(f'LateX/figuras/decomposicao_{col}.svg', bbox_inches='tight')
#     plt.close()


# enso = (
#     pd.read_csv(
#         'Exportado/ECMWF/derived-era5-single-levels-daily-statistics_sea_surface_temperature_reanalysis.csv', 
#         parse_dates = ['Data'])
#         .set_index('Data')
#         .resample('ME').mean()
#     )

# enso['media'] = enso.mean(axis=1)
# enso['media'] = StandardScaler().fit_transform(enso[['media']])
# decomposicao = MSTL(enso['media']).fit()
# fig, ax = plt.subplots(4, 1, figsize=(6, 4), sharex=True)
# ax[0].plot(decomposicao.observed, color="#4596F3", label='Observado')
# ax[0].set_ylabel('Observado')
# ax[1].plot(decomposicao.trend, color="#ff5656", label='Tendência')
# ax[1].set_ylabel('Tendência')
# ax[2].plot(decomposicao.seasonal, color="#FFA722", label='Sazonalidade')
# ax[2].set_ylabel('Sazonalidade')
# ax[3].plot(decomposicao.resid, color="#47A730", label='Resíduo')
# ax[3].set_ylabel('Resíduo')
# ax[3].set_xlabel('Série histórica')
# ax[3].xaxis.set_major_locator(mdates.YearLocator(base=4))
# ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# handles, labels = [], []
# for axis in ax:
#     h, l = axis.get_legend_handles_labels()
#     handles.extend(h)
#     labels.extend(l)
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
# plt.tight_layout()
# plt.savefig(f'LateX/figuras/decomposicao_ENSO.svg', bbox_inches='tight')
# plt.close()

# carga = pd.read_csv('Exportado/carga_subsist_mensal_MWmed.csv', parse_dates=['Data']).set_index('Data')
# carga['Total'] = carga.sum(axis=1)
# carga = carga.fillna(0)  # Preenchendo valores ausentes com zero
# carga['Total'] = StandardScaler().fit_transform(carga[['Total']])
# decomposicao = MSTL(carga['Total']).fit()
# fig, ax = plt.subplots(4, 1, figsize=(6, 4), sharex=True)
# ax[0].plot(decomposicao.observed, color="#4596F3", label='Observado')
# # ax[0].set_ylabel('Observado')
# # ax[0].yaxis.set_label_position("right")
# ax[1].plot(decomposicao.trend, color="#ff5656", label='Tendência')
# # ax[1].set_ylabel('Tendência')
# # ax[1].yaxis.set_label_position("right")
# ax[2].plot(decomposicao.seasonal, color="#FFA722", label='Sazonalidade')
# # ax[2].set_ylabel('Sazonalidade')
# # ax[2].yaxis.set_label_position("right")
# ax[3].plot(decomposicao.resid, color="#47A730", label='Resíduo')
# # ax[3].set_ylabel('Resíduo')
# ax[3].set_xlabel('Série histórica')
# # ax[3].yaxis.set_label_position("right")
# import matplotlib.dates as mdates
# ax[3].xaxis.set_major_locator(mdates.YearLocator(base=4))
# ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# handles, labels = [], []
# for axis in ax:
#     h, l = axis.get_legend_handles_labels()
#     handles.extend(h)
#     labels.extend(l)
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
# plt.tight_layout()
# plt.savefig(f'LateX/figuras/decomposicao_carga.svg', bbox_inches='tight')
# plt.close()


# Grafico geográfico indices ----------------------------------------------------------------------------------------- #
# regioes = {                                                                                                             # [North, West, South, East]
#     'NIN 1.2': [0, -90, -10, -80],
#     'NIN 3':   [5, -150, -5, -90],
#     'NIN 3.4': [5, -170, -5, -120],
#     'NIN 4': [5, 160, -5, 210],  # Região unificada que cruza a linha de data internacional
# }

# # Criar um mapa global
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': None})

# # Carregar dados geográficos do mundo
# world = gpd.read_file('Misc/ne_110m_admin_0_countries.shp')
# world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

# # Definir cores para cada região ENSO

# # cmap = ListedColormap(["#8dc454", "#ff8b81", "#7ABAFF", "#f0cc58"])
# cores_regioes = {
#     'NIN 1.2': "#ff8b81",  # Vermelho claro
#     'NIN 3': "#f0cc58",    # Verde água
#     'NIN 4': "#8dc454",    # Verde claro
#     'NIN 3.4': "#7ABAFF",  # Azul claro
# }

# # Plotar as regiões ENSO como retângulos
# handles = []
# labels = []

# for nome_regiao, coords in regioes.items():
#     north, west, south, east = coords
    
#     # Tratar a região NIN 4 que cruza a linha de data internacional
#     if nome_regiao == 'NIN 4':
#         # Primeira parte: de 160°E a 180°E
#         rect1 = patches.Rectangle((160, south), 20, north - south,
#                                 linewidth=0, edgecolor='black', 
#                                 facecolor=cores_regioes[nome_regiao], alpha=0.7)
#         ax.add_patch(rect1)
        
#         # Segunda parte: de -180°W a -150°W
#         rect2 = patches.Rectangle((-180, south), 30, north - south,
#                                 linewidth=0, edgecolor='black', 
#                                 facecolor=cores_regioes[nome_regiao], alpha=0.7)
#         ax.add_patch(rect2)
        
#     elif nome_regiao == 'NIN 3.4':
#         # Região NIN 3.4
#         rect = patches.Rectangle((west, south), east - west, north - south,
#                                linewidth=1, edgecolor='black', linestyle='--',
#                                facecolor='none', alpha=0.7)
#         ax.add_patch(rect)
        
#     else:
#         rect = patches.Rectangle((west, south), east - west, north - south,
#                                linewidth=0, edgecolor='black', 
#                                facecolor=cores_regioes[nome_regiao], alpha=0.7)
#         ax.add_patch(rect)
        
#     # Handle para legenda
#     if nome_regiao == 'NIN 3.4':
#         handles.append(patches.Patch(facecolor='none', 
#                                     edgecolor='black', linewidth=1, linestyle='--', label=nome_regiao))
#     else:
#         handles.append(patches.Patch(facecolor=cores_regioes[nome_regiao], 
#                                     edgecolor='black', label=nome_regiao))

# # Configurar o mapa
# ax.set_xlim(-180, 180)
# ax.set_ylim(-60, 60)
# ax.set_xlabel('Longitude [°]')
# ax.set_ylabel('Latitude [°]')

# # Adicionar grade
# ax.grid(True, linestyle='--', alpha=0.5)

# # Adicionar legenda
# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
#           ncol=4, frameon=False, fancybox=False)

# plt.tight_layout()
# plt.savefig('LateX/figuras/regioes_enso_global.svg', transparent=True, bbox_inches='tight')
# plt.show()



# # Gráfico exemplo modelo linear ------------------------------------------------------------------------------------ #
# from sklearn.linear_model import LinearRegression
# X = np.random.rand(100)
# y = np.sin(X * 2 * np.pi) + np.random.normal(0, 0.1, 100)
# model = LinearRegression()
# model.fit(X.reshape(-1, 1), y)
# plt.figure(figsize=(3, 3))
# plt.scatter(X, y, color="#4596F3", label='Dados')
# plt.plot(X, model.predict(X.reshape(-1, 1)), color="#FF7070", label='Regressão Linear')
# plt.xlabel('Variável Independente')
# plt.ylabel('Variável Dependente')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
#           ncol=2, frameon=False, fancybox=False)
# plt.tight_layout()
# plt.savefig('LateX\\figuras\modelo_linear_exemplo.svg', transparent = True)




# Violinplots e boxplots --------------------------------------------------------------------------------------------- #
# indices = pd.read_csv('Exportado/teleconexoes.csv').set_index('Data')                                                   # Dados de teleconexões (índices climatológicos)
# geracao = pd.read_csv('Exportado/geracao_fontes_mensal.csv').set_index('Data')                                          # Dados de geração em MWMed
# carga = pd.read_csv('Exportado/carga_subsistemas_diario.csv').set_index('Data')
# geracao.index = pd.to_datetime(geracao.index).to_period('M')
# carga.index = pd.to_datetime(carga.index).to_period('M')

# geracao = geracao[geracao.index.year >= 2018]                                                                         # Filtrando os dados de geração para o período desejado
# geracao_norm = StandardScaler().fit_transform(geracao)
# geracao = pd.DataFrame(geracao_norm, columns = geracao.columns, index = geracao.index)                                   # Normalizando os dados de geração

# fig,ax = plt.subplots(figsize = (6, 3))
# ax.violinplot([geracao['Hidráulica'], geracao['Térmica'], geracao['Eólica'], geracao['Fotovoltaica']], showmeans = True, showextrema = True)
# plt.tight_layout()
# plt.show()



# Gráficos geográficos ----------------------------------------------------------------------------------------------- #
# brasil = gpd.read_parquet('Misc/brasil_subsistemas.parquet')
# print(brasil.head())
# cmap = ListedColormap(["#8dc454", "#ff8b81", "#7ABAFF", "#f0cc58"])
# # cmap = ListedColormap(['#3E944F', '#ff6038', '#3665ff', '#ffc738'])
# fig, ax = plt.subplots(figsize = (4, 4))
# brasil.plot(ax = ax, column = 'subsistema', cmap = cmap, edgecolor = '#151515', lw = 0.2, 
#             label = 'Subsistemas', missing_kwds = {"color": "#AAAAAA", "edgecolor": "#000000", "label": "Isolado"},
#             legend = True)

# subsist_cores = {
#     'Norte': '#8dc454',
#     'Nordeste': '#ff8b81',
#     'Sul': '#7ABAFF',
#     'Sudeste/Centro-Oeste': '#f0cc58'
# }

# handles = []
# labels = []
# for subsys, color in subsist_cores.items():
#     handles.append(mpatches.Patch(facecolor = color, edgecolor='none'))
#     labels.append(subsys)

# ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fancybox=False)

# for x, y, label in zip(brasil.geometry.representative_point().x, brasil.geometry.representative_point().y, brasil['uf']):
#     if   label == 'DF': ax.annotate(label, xy=(x, y), xytext=(-7, 3), textcoords='offset points', fontsize=6)
#     elif label == 'RJ': ax.annotate(label, xy=(x, y), xytext=(6, -6), textcoords='offset points', fontsize=6)
#     elif label == 'ES': ax.annotate(label, xy=(x, y), xytext=(6, -6), textcoords='offset points', fontsize=6)
#     elif label == 'SE': ax.annotate(label, xy=(x, y), xytext=(4, -7), textcoords='offset points', fontsize=6)
#     elif label == 'AL': ax.annotate(label, xy=(x, y), xytext=(8, -3), textcoords='offset points', fontsize=6)
#     elif label == 'PE': ax.annotate(label, xy=(x, y), xytext=(22, -2), textcoords='offset points', fontsize=6)
#     elif label == 'PB': ax.annotate(label, xy=(x, y), xytext=(14, 0), textcoords='offset points', fontsize=6)
#     elif label == 'RN': ax.annotate(label, xy=(x, y), xytext=(3, 7), textcoords='offset points', fontsize=6)

#     else: ax.annotate(label, xy=(x, y), xytext=(-3, 0), textcoords='offset points', fontsize=6)

# # ax.set_title('Subsistemas do SIN')
# ax.set_xlabel('Latitude [graus]')
# ax.set_ylabel('Longitude [graus]')
# plt.tight_layout()
# # plt.show()
# plt.savefig('LateX/figuras/subsistemas_brasil.svg', transparent = True, bbox_inches='tight')



# Gráfico do dia de maior demanda e as gerações ---------------------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_horario_MWmed.csv').set_index('Data')
# carga = pd.read_csv('Exportado/carga_subsist_horario_MWmed.csv').set_index('Data')
# carga['Total'] = carga.sum(axis = 1)
# ger_hidr = geracao[['Hidráulica']]
# ger_eol = geracao[['Eólica']]
# ger_sol = geracao[['Fotovoltaica']]
# ger_ter = geracao[['Térmica']]

# ger_hidr.index = pd.to_datetime(ger_hidr.index, format = '%Y-%m-%d %H:%M:%S')
# ger_eol.index = pd.to_datetime(ger_eol.index, format = '%Y-%m-%d %H:%M:%S')
# ger_sol.index = pd.to_datetime(ger_sol.index, format = '%Y-%m-%d %H:%M:%S')
# ger_ter.index = pd.to_datetime(ger_ter.index, format = '%Y-%m-%d %H:%M:%S')
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d %H:%M:%S')

# max_carga_date = carga['Total'].idxmax().date()
# print(f'Dia com maior carga: {max_carga_date}')

# # Filtrando os dados para o dia com maior carga
# carga_max_day = carga.loc[carga.index.date == max_carga_date]
# ger_eol_max_day = ger_eol.loc[ger_eol.index.date == max_carga_date]
# ger_sol_max_day = ger_sol.loc[ger_sol.index.date == max_carga_date]
# ger_hidr_max_day = ger_hidr.loc[ger_hidr.index.date == max_carga_date]
# ger_ter_max_day = ger_ter.loc[ger_ter.index.date == max_carga_date]

# # Plotando gráfico de linha com a carga e a geração de energia para o dia com maior carga
# fig, ax = plt.subplots(figsize = (6, 3), sharex = False)
# ax.plot(carga_max_day.index, carga_max_day['Total'], linestyle = '--', color = '#FF4A4D', label = 'Carga')

# ax.plot(ger_hidr_max_day.index, ger_hidr_max_day['Hidráulica'], linestyle='-', color='#3867ff', label='Hidráulica')
# ax.plot(ger_sol_max_day.index, ger_sol_max_day['Fotovoltaica'], linestyle='-', color='#E3B132', label='Fotovoltaica')
# ax.plot(ger_eol_max_day.index, ger_eol_max_day['Eólica'], linestyle='-', color='#3E944F', label='Eólica')
# ax.plot(ger_ter_max_day.index, ger_ter_max_day['Térmica'], linestyle='-', color='#ff6038', label='Térmica')
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5, frameon=False, fancybox=False)
# ax.set_xlabel('Horário')
# ax.set_ylabel('Energia [MWmed]')
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh'))
# plt.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
# ax.grid(True)
# plt.tight_layout()
# plt.savefig(f'LateX/figuras/carga_max_dia_{max_carga_date}.svg', transparent=True)
# plt.show()



# Gráfico de linha com a carga do SIN -------------------------------------------------------------------------------- #
# carga = pd.read_csv('Exportado/carga_subsist_mensal_MWh.csv').set_index('Data')
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d')
# carga = carga[carga.index.year < 2025]  # Filtrando os dados de carga para o período desejado
# carga['Total'] = carga.sum(axis = 1)
# # carga = carga / 1e3
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d')
# fig, ax = plt.subplots(figsize=(6, 3), sharey=False)
# ax.plot(carga.index, carga['Total'], color="#FF6467", label='Carga', lw=0.66)
# ax.set_ylabel('Carga [MWh]')
# ax.set_xlabel('Série histórica')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
# # years = carga_df.index
# # ax.set_xticks(years)
# # ax.set_xticklabels(years, rotation=45, ha='right')
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.135), ncol=5, frameon=False, fancybox=False)
# plt.tight_layout()
# # plt.show()
# plt.savefig('LateX/figuras/carga_anual.svg', transparent=True, bbox_inches='tight')
# # plt.close()



# Gráfico de linha com a geração de energia hidráulica bruta --------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_mensal_mwh.csv').set_index('Data')
# geracao.index = pd.to_datetime(geracao.index, format = '%Y-%m-%d')
# geracao = geracao[geracao.index.year < 2025]  # Filtrando os dados de geração para o período desejado

# fig, ax = plt.subplots(figsize = (6, 3))
# ax.plot(geracao.index, geracao['Hidráulica'], color = "#5d82ff", label = 'Geração', lw = .66)
# # ax.set_xlim(ger_hidr.index.min(), ger_hidr.index.max())
# # ax.set_title('Geração hidráulica bruta')
# ax.set_ylabel('Geração [MWh]')
# ax.set_xlabel('Série histórica')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
# ax.xaxis.set_major_locator(mdates.YearLocator(base = 2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.tight_layout()
# # plt.show()
# plt.savefig('LateX/figuras/geracao_hidraulica_bruta.svg', transparent = True, bbox_inches='tight')



# Gráfico do índice ONI ---------------------------------------------------------------------------------------------- #
indices_df = pd.read_csv('Exportado/teleconexoes.csv').set_index('Data')
indices_df.index = pd.to_datetime(indices_df.index, format='%Y-%m')
fig, ax = plt.subplots(figsize=(6, 3))
oni = indices_df['ONI']
oni = oni[oni.index.year >= 2000]
vmin = oni.min(); vmax = oni.max()
norm = plt.Normalize(vmin, vmax)
# cmap = plt.get_cmap('coolwarm')
cmap = LinearSegmentedColormap.from_list('custom_diverging', ["#5EACFF", "#C0DEFF", "#E0E0E0", "#ffa097", "#ff5d4f"])
ax.plot(oni, color='black', label='ONI', alpha = 1, linewidth = 0.66)
ax.set_ylim(-3, 3)

for i in range(len(oni) - 1): 
    ax.fill_between(oni.index[i:i+20], oni[i:i+20], color=cmap(norm(oni[i])), alpha=1)

sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, aspect = 50)
ln_f = vmin; el_f = vmax
neutro = (vmax + vmin) / 2
lf_m = vmin / 4; ef_m = vmax / 2
cbar.set_ticks([ln_f, lf_m, neutro, ef_m, el_f])
cbar.set_ticklabels(['La Niña forte', 'La Niña médio', 'Neutralidade', 'El Niño médio', 'El Niño forte'])
# ax.set_xticks([ano for ano in oni.index.year])
ax.set_ylabel('Anomalia [°C]')
ax.set_xlabel('Série histórica')
plt.tight_layout()
# plt.show()
plt.savefig('LateX/figuras/oni.svg', transparent = True, bbox_inches='tight')


# --- GRÁFICO DE GERAÇÃO ANUAL PERCENTUAL ---------------------------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_mensal_MWmed.csv', usecols = ['Data', 'Hidráulica', 'Térmica', 'Eólica', 'Fotovoltaica', 'Outras']).set_index('Data')  
# geracao.index = pd.to_datetime(geracao.index, format='%Y-%m-%d')
# geracao = geracao[geracao.index.year <= 2024]  # Filtrando os dados de geração para o período desejado
# geracao_anual = geracao.resample('YE').sum()
# sorted_columns = geracao_anual.sum().sort_values(ascending = False).index
# geracao_anual_sort = geracao_anual[sorted_columns]

# geracao_perc_sorted = geracao_anual_sort.div(geracao_anual_sort.sum(axis = 1), axis = 0) * 100
# geracao_perc_sorted.index = geracao_perc_sorted.index.year

# from matplotlib.colors import ListedColormap
# cmap = ListedColormap(["#7ABAFF", "#ff8b81", "#8dc454", "#949494", "#f0cc58"])
# ax = geracao_perc_sorted.plot(kind='bar', stacked=True, figsize=(6, 3), colormap=cmap)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.135), ncol=5, frameon=False, fancybox=False)
# ax.set_xticklabels(geracao_perc_sorted.index, rotation=45, ha='right')
# ax.set_yticks(np.arange(0, 101, 20))
# ax.set_ylabel('Geração [%]')
# ax.set_xlabel('Série histórica')
# plt.tight_layout()
# # plt.show()
# plt.savefig('LateX/figuras/geracao_anual_percentual.svg', transparent = True)


# # geracao_df = pd.read_csv(f'Exportado/geracao_usinas_bruto.csv').set_index('Data')
# carga_df = pd.read_csv(f'Exportado/carga_subsistemas_bruto.csv').set_index('Data')
# # geracao_df.index = pd.to_datetime(geracao_df.index, format = '%Y-%m')
# carga_df.index = pd.to_datetime(carga_df.index, format = '%Y-%m')
# carga_df['Total'] = carga_df.sum(axis = 1)
# carga_df = carga_df.drop(columns = ['N', 'NE', 'S', 'SE'])


# geracao_df = geracao_df.div(geracao_df.std())
# # carga_df = carga_df.div(carga_df.std())
# indices_df = indices_df.div(indices_df.std())
# # regressao(geracao_df, indices_df, export = True)
# geracao_df['Hidráulica N-NE'] = geracao_df[['Hidráulica-N', 'Hidráulica-NE']].sum(axis = 1, skipna = True)
# geracao_df['Hidráulica S-SE'] = geracao_df[['Hidráulica-S', 'Hidráulica-SE']].sum(axis = 1, skipna = True)
# geracao_df = geracao_df.drop(columns = ['Hidráulica-N', 'Hidráulica-NE', 'Hidráulica-S', 'Hidráulica-SE'])
# geracao_df['Térmica'] = geracao_df[['Carvão',
#                                     'Gás',
#                                     'Óleo Diesel',
#                                     'Óleo Combustível',
#                                     'Multi-Combustível Diesel/Óleo',
#                                     'Nuclear',
#                                     'Outras Multi-Combustível']].sum(axis=1, skipna=True)
# geracao_df = geracao_df.drop(columns = ['Carvão',
#                                         'Gás',
#                                         'Óleo Diesel',
#                                         'Óleo Combustível',
#                                         'Multi-Combustível Diesel/Óleo',
#                                         'Nuclear',
#                                         'Outras Multi-Combustível'])
# geracao_df['Outras'] = geracao_df[['Biomassa', 'Resíduos Industriais']].sum(axis = 1, skipna = True)
# geracao_df = geracao_df.drop(columns = ['Biomassa', 'Resíduos Industriais'])

# geracao_df = geracao_df.div(geracao_df.std())
# plot(geracao_df, indices_df, 'linha', 'Todos')
# plot(geracao_df, indices_df, 'scatter')
# plot(geracao_df, indices_df, 'boxplot')