import os
import netCDF4
import zipfile
import tempfile
import shutil
import numpy as np
import pandas as pd
import cdsapi
import xarray as xr
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from alive_progress import alive_bar


# https://cds.climate.copernicus.eu/datasets/projections-cmip6?tab=download
# Regiões para download dos dados (IMPORTANTE: sistema de coord. WGS 1984)
regioes = {                                                                                                             # [North, West, South, East]
    'NIN 1.2': [0, -90, -10, -80],
    'NIN 3':   [5, -150, -5, -90],
    'NIN 3.4': [5, -170, -5, -120],
    'NIN 4_1': [5, 160, -5, 180],                                                                                     # Primeira parte da região 4
    'NIN 4_2': [5, -180, -5, -150],                                                                                   # Segunda parte da região 4


    # 'NE': [-2, -45, -18, -34],
    # 'N': [5, -74, -14, -45],
    # 'C.O.': [-14, -60, -24, -45],
    # 'SE': [-18, -50, -25, -39],
    # 'S': [-24, -56, -33, -48]
}

def DadosCMIP(regioes, anos, dataset, request):
    df_regional_final = pd.DataFrame()                                                                                  # DataFrame para armazenar os dados
    for regiao in regioes.fromkeys(regioes):

        if dataset == 'projections-cmip6':
            dataset = dataset
            request = request
            request['area'] = regioes[regiao]
            request['year'] = [f'{ano}' for ano in anos]

            base_dir = f"Exportado/ECMWF/Dados/{regiao}/{dataset}"
            arq_nome = f"{regiao}_{request['variable']}_{request['experiment']}_{request['model']}.zip"

            if not os.path.exists(f'{base_dir}/{arq_nome}'):
                client = cdsapi.Client()
                os.makedirs(base_dir, exist_ok = True)
                client.retrieve(dataset, request).download(target = f"{base_dir}/{arq_nome}")
                print(f"Arquivo baixado: {arq_nome}")

        elif dataset == 'derived-era5-single-levels-daily-statistics':
            base_dir = f"Exportado/ECMWF/Dados/{regiao}/{dataset}/{request['product_type']}"
            def download_data(ano):
                try:
                    dataset_copy = dataset
                    request_copy = request.copy()
                    request_copy['area'] = regioes[regiao]
                    request_copy['year'] = f'{ano}'

                    arq_nome = f"{regiao}_{request_copy['variable']}_{request_copy['product_type']}_{request_copy['year']}.nc"
                    target_file = os.path.join(base_dir, arq_nome)

                    if not os.path.exists(target_file):
                        os.makedirs(base_dir, exist_ok = True)

                        with tempfile.NamedTemporaryFile(suffix = '.nc', delete = False) as tmp:
                            temp_path = tmp.name

                        client = cdsapi.Client()
                        client.retrieve(dataset_copy, request_copy).download(temp_path)
                        shutil.move(temp_path, target_file)
                        print(f'\n[✓] Arquivo baixado: {arq_nome}')
                        bar()

                    else: 
                        print(f'\n[i] Arquivo já existe: {arq_nome}')
                        bar()

                except Exception as e:
                    print(f'\n[!] Falha ao baixar dados para o ano {ano}: {e}')

            with alive_bar(
                len(anos),                                                                                              # Tamanho
                title = f'Baixando dados da região {regiao}...',                                                        # Título
                bar = None,                                                                                             # Barra de progresso
                spinner = 'classic',                                                                                    # Spinner
                enrich_print = False,                                                                                   # Desativando 'on <iter>:' antes dos prints
                stats = '(ETA: {eta})',
                monitor = '{count}/{total}',
                elapsed = '    ⏱ {elapsed}'                                                                             # Tempo decorrido
            ) as bar:

                with ThreadPoolExecutor(max_workers = 20) as executor:
                    executor.map(download_data, anos)

            with alive_bar(
                len(anos),
                title = f'Processando arquivos da região {regiao}...',
                bar = None,
                spinner = 'classic',
                enrich_print = False,
                stats = '(ETA: {eta})',
                monitor = '{count}/{total}',
                elapsed = '    ⏱ {elapsed}'
            ) as bar:

                dfs_anuais_regiao = []
                arqs_nc = [f for f in os.listdir(base_dir) if f.endswith('.nc')]

                for arq in arqs_nc:
                    try:
                        nc = netCDF4.Dataset(os.path.join(base_dir, arq))
                        # var_keys = nc.variables.keys()
                        # print(f'[i] Variáveis: {var_keys}')

                        time = nc.variables['valid_time']
                        data_inicial = netCDF4.num2date(
                            time[0],
                            time.units,
                            calendar = time.calendar
                        )

                        data = pd.date_range(
                            start = data_inicial.strftime('%Y-%m-%d'),
                            periods = len(time),
                            freq = 'D').to_period('D'
                        )

                        if request['variable'] == 'sea_surface_temperature': var = 'sst'                                # Variável de temperatura da superfície do mar (sst)
                        if request['variable'] == '2m_temperature': var = 't2m'                                         # Variável de temperatura a 2m (t2m)
                        if request['variable'] == 'total_precipitation': var = 'tp'                                          # Variável de precipitação total (tp)
                            
                        var_data = nc.variables[var][:].squeeze()
                        
                        lats = nc.variables['latitude'][:].squeeze()
                        lons = nc.variables['longitude'][:].squeeze()

                        var_2d = var_data.reshape(var_data.shape[0], -1)                                            # Convertendo de (data, lat, lon) pra (data, lat*lon)
                        lat_grid, lon_grid = np.meshgrid(lats, lons)                                                # Grade de latitudes e longitudes (lat x lon)
                        lat_flat, lon_flat = lat_grid.ravel(), lon_grid.ravel()                                     # Transforma a grade em vetores

                        cols = [f'lon_{ln:.2f}_lat_{lt:.2f}'
                            for lt, ln in zip(lat_flat, lon_flat)]

                        df_ano = pd.DataFrame(data = var_2d, index = data, columns = cols)                          # DataFrame para o ano atual
                        dfs_anuais_regiao.append(df_ano)

                        nc.close()

                    except Exception as e:
                        print(f'\n[!] Error ao processar o arquivo {arq}: {e}')

                    bar()

            regiao_completa = pd.concat(dfs_anuais_regiao, axis = 0).sort_index()                                       # Juntando os DataFrames de cada ano em um único DataFrame
            # print(regiao_completa)
            regiao_mediana = regiao_completa.median(axis = 1)                                                           # Calculando a mediana para cada dia do ano ano ao longo das coordenadas
            df_regional_final[f'{regiao}_{var}'] = regiao_mediana                                       # Adicionando a série resultante como uma coluna no DataFrame final

    df_regional_final.index = df_regional_final.index.rename('Data')
    df_regional_final.to_csv(f"Exportado/ECMWF/{dataset}_{request['variable']}_{request['product_type']}.csv")          # Exportando o DataFrame final para um arquivo CSV


def download_global_CMIP(variables, anos, dataset, request):
    """
    Baixa para cada variável e ano um arquivo global (.nc).
    """
    base_dir = f"Exportado/ECMWF/Global/{dataset}/{request['product_type']}"
    os.makedirs(base_dir, exist_ok=True)
    client = cdsapi.Client()

    for var in variables:
        for ano in anos:
            rq = request.copy()
            rq['variable'] = var
            rq['year'] = f"{ano}"
            # removemos 'area' para baixar GLOBAL
            if 'area' in rq: del rq['area']

            arq = f"Global_{var}_{request['product_type']}_{ano}.nc"
            target = os.path.join(base_dir, arq)

            if not os.path.exists(target):
                client.retrieve(dataset, rq).download(target)
                print(f"[✓] Baixado {arq}")
            else:
                print(f"[i] Já existe {arq}")

def extract_regions_from_global(regioes, variables, anos, dataset, product_type):
    """
    Usa netCDF4 para ler cada .nc global, extrair sub-regiões e salvar um CSV
    com a média diária da variável em cada região.
    """
    global_dir = f"Exportado/ECMWF/Global/{dataset}/{product_type}"
    out_dir    = f"Exportado/ECMWF/Regions/{dataset}/{product_type}"
    os.makedirs(out_dir, exist_ok=True)

    for var in variables:
        # mapeia nome CDS → variável no NetCDF
        var_nc = {
            'sea_surface_temperature': 'sst',
            '2m_temperature':        't2m',
            'total_precipitation':   'tp'
        }[var]

        for ano in anos:
            file_global = os.path.join(
                global_dir,
                f"Global_{var}_{product_type}_{ano}.nc"
            )
            if not os.path.isfile(file_global):
                print(f"[!] Arquivo faltando, pulando: {file_global}")
                continue

            nc = netCDF4.Dataset(file_global)
            # extrai o tempo e monta índice pandas
            time_var   = nc.variables.get('time') or nc.variables.get('valid_time')
            units      = time_var.units
            cal        = getattr(time_var, 'calendar', 'standard')
            dates      = netCDF4.num2date(time_var[:], units, calendar=cal)
            date_strs  = [dt.strftime('%Y-%m-%d') for dt in dates]
            idx        = pd.to_datetime(date_strs).to_period('D')

            # extrai os dados brutos e coords
            data3d = nc.variables[var_nc][:].squeeze()    # (nt, nlat, nlon)
            lats   = nc.variables['latitude'][:].squeeze()
            raw_lons = nc.variables['longitude'][:].squeeze()
            # converte 0–360 → –180–180
            lons = (raw_lons + 180) % 360 - 180

            for regiao, area in regioes.items():
                north, west, south, east = area
                lat_idx = np.where((lats >= south) & (lats <= north))[0]
                lon_idx = np.where((lons >= west)  & (lons <= east))[0]

                # sub-array apenas da região: (nt, nlat_reg, nlon_reg)
                sub      = data3d[:, lat_idx[:,None], lon_idx]

                # calcula média diária sobre lat x lon
                region_mean = np.nanmean(sub, axis=(1,2))

                # monta DataFrame com uma coluna: {regiao}_{var_nc}
                col_name = f"{regiao}_{var_nc}"
                df       = pd.DataFrame(region_mean, index=idx, columns=[col_name])

                csvf = os.path.join(
                    out_dir,
                    f"{regiao}_{var}_{product_type}_{ano}.csv"
                )
                df.to_csv(csvf)
                print(f"[✓] Salvo média diária: {regiao} | {var} | {ano}")

            nc.close()


variaveis = [
        # 'sea_surface_temperature',
        # '2m_temperature',
        'total_precipitation',
        # ...
]
anos = np.arange(2000, 2026, 1)
request_base = {
    'product_type': 'reanalysis',
    'daily_statistic': 'daily_mean',
    'time_zone': 'utc+00:00',
    'frequency': '1_hourly',
    # variável e ano serão setados em download_global_CMIP
    'month' : [f'{m}' for m in np.arange(1,13)],
    'day'   : [f'{d}' for d in np.arange(1,32)],
}
dataset = 'derived-era5-single-levels-daily-statistics'

# 1) baixa GLOBAL
download_global_CMIP(variaveis, anos, dataset, request_base)

# 2) extrai todas as regiões definidas no dicionário regioes
extract_regions_from_global(regioes, variaveis, anos, dataset, request_base['product_type'])

# DadosCMIP(regioes, anos = np.arange(2000, 2025, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': 'sea_surface_temperature',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2025, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': 'total_precipitation',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': '2m_temperature',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': '10m_u_component_of_wind',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': '10m_v_component_of_wind',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': 'surface_solar_radiation_downwards',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': 'total_precipitation',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )

# DadosCMIP(regioes, anos = np.arange(2000, 2026, 1), dataset = 'derived-era5-single-levels-daily-statistics',
#           request = {
#             'product_type': 'reanalysis',
#             'daily_statistic': 'daily_mean',
#             'time_zone': 'utc+00:00',
#             'frequency': '1_hourly',
#             'variable': 'evaporation',
#             'month' : [f'{mes}' for mes in np.arange(1, 13)],
#             'day' : [f'{dia}' for dia in np.arange(1, 32)],
#         }
#     )