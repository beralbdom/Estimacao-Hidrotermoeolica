import baixar_dados
import tratamento_dados
import pandas as pd
import geopandas as gpd
import numpy as np
from matplotlib import rc, pyplot as plt, dates as mdates, patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches

# Configurações de plotagem
rc('font', family = 'serif', size = 8)                                                                                  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html
rc('grid', linestyle = '--', alpha = 0.5)
rc('axes', axisbelow = True, grid = True)
rc('lines', linewidth = .5, markersize = 1.5)
rc('axes.spines', top = False, right = False, left = True, bottom = True)

# # Grafico geográfico indices
# regioes = {                                                                                                             # [North, West, South, East]
#     'NIN 1.2': [0, -90, -10, -80],
#     'NIN 3':   [5, -150, -5, -90],
#     'NIN 3.4': [5, -170, -5, -120],
#     'NIN 4': [5, 160, -5, 210],  # Região unificada que cruza a linha de data internacional
# }

# # Criar um mapa global
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={'projection': None})

# # Carregar dados geográficos do mundo
# world = gpd.read_file('Arquivos/ne_110m_admin_0_countries.shp')
# world.plot(ax=ax, color='lightgray', edgecolor='black', linewidth=0.5)

# # Definir cores para cada região ENSO
# cores_regioes = {
#     'NIN 1.2': "#FF5656",  # Vermelho claro
#     'NIN 3': "#FFCC24",    # Verde água
#     'NIN 4': "#2D9929",    # Verde claro
#     'NIN 3.4': "#46A4C9",  # Azul claro
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
# ax.set_ylim(-60, 20)
# ax.set_xlabel('Longitude [°]')
# ax.set_ylabel('Latitude [°]')

# # Adicionar grade
# ax.grid(True, linestyle='--', alpha=0.5)

# # Adicionar legenda
# ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
#           ncol=4, frameon=False, fancybox=False)

# plt.tight_layout()
# plt.savefig('Graficos/regioes_enso_global.svg', transparent=True, dpi=300, bbox_inches='tight')
# plt.show()



# # Gráfico exemplo modelo linear
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




# violinplots e boxplots
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
# brasil = gpd.read_parquet('Dados/brasil_subsistemas.parquet')
# cmap = ListedColormap(['#3E944F', '#ff6038', '#3665ff', '#ffc738'])
# fig, ax = plt.subplots(figsize = (6, 6))
# brasil.plot(ax = ax, column = 'subsistema', cmap = cmap, edgecolor = '#151515', lw = 0.2, label = 'Subsistemas', missing_kwds = {
#                                                                                                                     "color": "lightgrey",
#                                                                                                                     "edgecolor": "#000000",
#                                                                                                                     "label": "Isolado"},
#                                                                                                                     legend = True)


# subsystem_colors = {
#     'Norte': '#3E944F',
#     'Nordeste': '#ff6038',
#     'Sul': '#3665ff',
#     'Sudeste/Centro-Oeste': '#ffc738'
# }

# handles = []
# labels = []
# for subsys, color in subsystem_colors.items():
#     handles.append(mpatches.Patch(facecolor = color, edgecolor='none'))
#     labels.append(subsys)

# ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, frameon=False, fancybox=False)

# for x, y, label in zip(brasil.geometry.representative_point().x, brasil.geometry.representative_point().y, brasil['uf']):
#     ax.annotate(label, xy=(x, y), xytext=(-3, 0), textcoords='offset points', fontsize=7)

# # ax.set_title('Subsistemas do SIN')
# ax.set_xlabel('Latitude [graus]')
# ax.set_ylabel('Longitude [graus]')
# plt.savefig('Graficos/subsistemas_brasil.svg', transparent = True)



# geracao_df = pd.read_csv(f'Exportado/geracao_usinas_horario.csv')
# geracao = geracao_df.pivot_table(index = 'Data', columns = 'Tipo', values = 'Geracao', aggfunc='sum')

# geracao_hidraulica = geracao['Hidráulica']
# geracao_hidraulica.to_csv('Exportado/geracao_hidraulica_bruta_horaria.csv')
# geracao_eolica = geracao['Eólica']
# geracao_eolica.to_csv('Exportado/geracao_eolica_bruta_horaria.csv')
# geracao_solar = geracao['Fotovoltaica']
# geracao_solar.to_csv('Exportado/geracao_solar_bruta_horaria.csv')
# geracao_termica = geracao.drop(columns = ['Hidráulica', 'Eólica', 'Fotovoltaica', 'Biomassa'])
# geracao_termica.to_csv('Exportado/geracao_termica_bruta_horaria.csv')

# carga = pd.read_csv('Exportado/carga_subsistemas_horario.csv')
# carga = carga.pivot_table(index = 'Data', columns = 'Subsistema', values = 'Carga')
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d %H:%M:%S')
# carga['Total'] = carga.sum(axis = 1)
# carga.to_csv('Exportado/carga_subsistemas_bruto_horario.csv')


# Gráfico do dia de maior demanda e as gerações ---------------------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_horario_MWmed.csv').set_index('Data')                                               # Dados de geração em MWMed
# carga = pd.read_csv('Exportado/carga_subsist_horario_MWmed.csv').set_index('Data')                                                  # Dados de carga em MWMed
# carga['Total'] = carga.sum(axis = 1)                                                                                      # Somando a carga total                                                                                            # Encontrando a carga total máxima
# ger_hidr = geracao[['Hidráulica']]
# ger_eol = geracao[['Eólica']]
# ger_sol = geracao[['Fotovoltaica']]
# ger_ter = geracao[['Térmica']]



# ger_hidr.index = pd.to_datetime(ger_hidr.index, format = '%Y-%m-%d %H:%M:%S')
# ger_eol.index = pd.to_datetime(ger_eol.index, format = '%Y-%m-%d %H:%M:%S')
# ger_sol.index = pd.to_datetime(ger_sol.index, format = '%Y-%m-%d %H:%M:%S')
# ger_ter.index = pd.to_datetime(ger_ter.index, format = '%Y-%m-%d %H:%M:%S')
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d %H:%M:%S')

# max_carga_date = carga['Total'].idxmax().date()                                                                         # Encontrando o dia com maior carga
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
# plt.savefig(f'Graficos/carga_max_dia_{max_carga_date}.svg', transparent=True)
# plt.show()


# Gráfico de linha com a geração de energia hidráulica bruta --------------------------------------------------------- #
# geracao = pd.read_csv('Exportado/geracao_fontes_mensal.csv').set_index('Data')
# geracao.index = pd.to_datetime(geracao.index, format = '%Y-%m-%d')

# fig, ax = plt.subplots(figsize = (6, 3))
# ax.plot(geracao.index, geracao['Hidráulica'], color = '#3867ff', label = 'Geração')
# # ax.set_xlim(ger_hidr.index.min(), ger_hidr.index.max())
# # ax.set_title('Geração hidráulica bruta')
# ax.set_ylabel('Geração [MWh]')
# ax.set_xlabel('Série histórica')
# ax.xaxis.set_major_locator(mdates.YearLocator(base = 2))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# plt.tight_layout()
# plt.savefig('Graficos/geracao_hidraulica_bruta.svg', transparent = True)


# Gráfico do índice ONI ---------------------------------------------------------------------------------------------- #
# indices_df = pd.read_csv('Exportado/teleconexoes.csv').set_index('Data')
# fig, ax = plt.subplots(figsize=(6, 3))
# oni = indices_df['ONI']
# oni = oni[oni.index >= '2000-01-01']
# vmin = oni.min(); vmax = oni.max()
# norm = plt.Normalize(vmin, vmax)
# cmap = plt.get_cmap('coolwarm')
# # cmap = ListedColormap(["#BBD9FA", "#FFFFFF", "#f7b6b0",])
# ax.plot(oni.index, oni, color='black', label='ONI', alpha = 1, linewidth = 0.66)
# ax.set_ylim(-2, 3)

# for i in range(len(oni) - 1): 
#     ax.fill_between(oni.index[i:i+20], oni[i:i+20], color=cmap(norm(oni[i])), alpha=1)

# sm = plt.cm.ScalarMappable(cmap = cmap, norm = norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax)
# ln_f = vmin; el_f = vmax
# neutro = (vmax + vmin) / 2
# lf_m = vmin / 4; ef_m = vmax / 2
# cbar.set_ticks([ln_f, lf_m, neutro, ef_m, el_f])
# cbar.set_ticklabels(['L.N intenso', 'L.N moderado', 'Neutro', 'E.N moderado', 'E.N intenso'])
# ax.set_ylabel('Anomalia')
# ax.set_xlabel('Série histórica')
# plt.tight_layout()
# plt.show()
# plt.savefig('Graficos/oni.svg', transparent = True)


# ---
geracao = pd.read_csv('Exportado/geracao_fontes_mensal_MWmed.csv', usecols = ['Data', 'Hidráulica', 'Térmica', 'Eólica', 'Fotovoltaica', 'Outras']).set_index('Data')  
geracao.index = pd.to_datetime(geracao.index, format='%Y-%m-%d')
geracao = geracao[geracao.index.year <= 2024]  # Filtrando os dados de geração para o período desejado
geracao_yearly = geracao.resample('YE').sum()
sorted_columns = geracao_yearly.sum().sort_values(ascending = False).index
geracao_yearly_sorted = geracao_yearly[sorted_columns]

geracao_percentage_sorted = geracao_yearly_sorted.div(geracao_yearly_sorted.sum(axis = 1), axis = 0) * 100
geracao_percentage_sorted.index = geracao_percentage_sorted.index.year

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#7ABAFF", "#ff8b81", "#8dc454", "#949494", "#f0cc58"])
ax = geracao_percentage_sorted.plot(kind='bar', stacked=True, figsize=(6, 3), colormap=cmap)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.135), ncol=5, frameon=False, fancybox=False)
ax.set_xticklabels(geracao_percentage_sorted.index, rotation=45, ha='right')
ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylabel('Geração [%]')
ax.set_xlabel('Série histórica')
plt.tight_layout()
# plt.show()
plt.savefig('Graficos/geracao_anual_percentual.svg', transparent = True)

# carga = pd.read_csv('Exportado/carga_subsist_mensal_MWh.csv').set_index('Data')
# carga['Total'] = carga.sum(axis = 1)
# carga = carga / 1e3
# carga.index = pd.to_datetime(carga.index, format = '%Y-%m-%d')
# fig, ax = plt.subplots(figsize=(6, 3), sharey=False)
# ax.plot(carga.index, carga['Total'], color='#FF4A4D', label='Carga')
# ax.set_ylabel('Carga [GWh]')
# ax.set_xlabel('Série histórica')
# plt.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
# # years = carga_df.index
# # ax.set_xticks(years)
# # ax.set_xticklabels(years, rotation=45, ha='right')
# # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.135), ncol=5, frameon=False, fancybox=False)
# plt.tight_layout()
# plt.show()
# plt.savefig('Graficos/carga_anual.svg', transparent=True)






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