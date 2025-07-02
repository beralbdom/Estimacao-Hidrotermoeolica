import baixar_dados as bd
import baixar_previsoes as bp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------> Download e exportação de dados <----------------------------------------- #

# https://dados.ons.org.br/dataset/
base_url_ons_ger = 'https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/'                         # Base URL para os dados de geração
base_url_ons_car = 'https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/curva-carga-ho/'                             # Base URL para os dados de carga
base_url_ons_vaz = 'https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/dados_hidrologicos_di/'                      # Base URL para os dados de vazões

anos = np.arange(2000, 2026, 1)                                                                                         # Anos dos dados a serem baixados

# bd.GetIndices()                                                                                                         # Baixa os dados de teleconexões                (/Exportado/Indices/)
# bd.GetGeracao(anos = anos, base_url = base_url_ons_ger)                                                                 # Baixa os dados de geração                     (/Exportado/ONS/)
# bd.GetCarga(anos = anos, base_url = base_url_ons_car)                                                                   # Baixa os dados de carga                       (/Exportado/ONS/)
# bd.GetVazao(anos = anos, base_url = base_url_ons_vaz)                                                                   # Baixa os dados de vazões                      (/Exportado/ONS/)

# bd.ExportarDados('indices', anos = anos, merge = True, export = True)                                                   # Cria os dataframes de teleconexões            (/Exportado/teleconexoes.csv)
bd.ExportarDados('geracao', anos = anos, merge = True, export = True)                                                   # Cria os dataframes de geração                 (/Exportado/geracao_usinas_<tipo>.csv)
# bd.ExportarDados('carga', anos = anos, merge = True, export = True)                                                     # Cria os dataframes de carga                   (/Exportado/carga_subsistemas_<tipo>.csv)
# bd.ExportarDados('vazoes', anos = anos, merge = True, export = True)                                                    # Cria os dataframes de vazões                  (/Exportado/vazoes_mensais_bacias.csv)