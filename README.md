### Estimação da Geração Hidrotermoeólica Utilizando Redes Neurais e Variáveis do El Niño
###### <p style="text-align: center;"> Este repositório contém o código e os dados (sem dados energéticos e climáticos brutos) do meu projeto de conclusão de curso "Estimação da Geração Hidrotermoeólica Utilizando Redes Neurais e Variáveis do Fenômeno El Niño", apresentado à Universidade Federal Fluminense (UFF) em julho de 2025. </p>

###### <center> [Publicação](https://app.uff.br/riuff/handle/1/39428) &nbsp; | &nbsp; [Artigo TSMixer](https://arxiv.org/abs/2303.06053) &nbsp; | &nbsp; [Modelo no Hugging Face](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) &nbsp; | &nbsp; [Dados energéticos](https://dados.ons.org.br/dataset/?tags=Hist%C3%B3rico+da+Opera%C3%A7%C3%A3o&tags=Hor%C3%A1rio) &nbsp; | &nbsp; [Dados climáticos](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview) </center>
-----
### Resumo/Abstract
##### <p style="text-align: justify;"> A matriz elétrica brasileira, caracterizada pela forte dependência da fonte hidráulica, é vulnerável à fenômenos climáticos como o El Niño-Oscilação Sul (ENSO), cujos impactos não são considerados diretamente pela maior parte dos modelos computacionais utilizados pelo setor. Este trabalho investiga o impacto de variáveis exógenas associadas ao ENSO na estimação da geração das fontes hidráulica, eólica e térmica no Sistema Interligado Nacional (SIN). Para tal, foram aplicados e comparados modelos de regressão linear, não-linear (Random Forest) e um modelo neural baseado na arquitetura TSMixer. O modelo neural foi avaliado em sua aplicação direta (sem dados do ENSO), e também com o ajuste fino (com dados do ENSO). Os resultados indicam que os modelos de regressão tradicionais não são capazes de capturar as dinâmicas não-lineares das fontes eólica e térmica. O modelo neural, por sua vez, demonstrou alta capacidade de generalização, e a inclusão dos dados do ENSO resultou em uma diminuição do erro médio quadrático (MSE) para a fonte hidráulica de 21,38% para a janela de previsão de 96 dias e 5,8% para a janela de previsão de 30 semanas. As fontes eólica e térmica também apresentaram melhoria. Sendo assim, a incorporação de variáveis climáticas externas em modelos neurais avançados aprimora significativamente a acurácia da estimação de geração, apresentando-se como uma metodologia promissora para compor as ferramentas de planejamento energético do setor elétrico. </p>

###### <p style="text-align: justify;"> The brazilian electrical matrix, which is historically dependent on the hydroelectric generation, is vulnerable to external climatological phenomena such as El Niño-Southern  Oscilation (ENSO). These external eﬀects are not directly considered on the main computational models used by the electric sector. This work investigates the impact of these variables on the hydro, eolic and thermal electricity generation estimation. Diﬀerent models were implemented to measure the impact of these variables on the accuracy of the results: multivariate linear regression and random forest regressor as baselines, and a neural model based on the TSMixer architecture. The neural model was evaluated on its oneshot (no ENSO data) and finetune (with ENSO data) approaches. The results show that the introduction of ENSO data as exogenous variables on the finetuned neural model resulted in a 21,38% smaller mean squared error (MSE) compared to the oneshot implementation for the hydro source, considering the prediction window of 96 days. For the 30 week prediction window, the observed improvement was 5,8%. The eolic and thermal sources also showed improvement on the MSE. Therefore, the usage of climatological variables of external phenomena, such as ENSO, is a valid approach to improve the accuracy of the estimations produced by advanced regression models, contributing to the tools already available to the electric sector. </p>

### Estrutura do código
* **`baixar_dados.py` : Funções para download dos dados de geração e carga, tratamento e conversão.**
    * Download dos dados brutos (2000 a 2024) de geração e carga horários do ONS;
    * Conversão para séries de carga diários por subsistema;
    * Conversão para séries de geração diários por subsistema e fonte.
* **`baixar_previsoes.py` : Funções para download dos dados climáticos do ERA5, tratamento e conversão.**
    * Download dos dados brutos de SST de cada região do ENSO;
    * Cálculo da média de SST para cada região do ENSO;
    * Conversão para séries para cada região, de 2000 a 2024.
* **`main.py` : Obtenção dos dados energéticos e climáticos, a partir das funções desenvolvidas.**
* **`linear.py` : Aplicação do modelo de regressão linear (Mínimos Quadrados Ordinários).**
* **`nlinear.py` : Aplicação do modelo de regressão não linear (Random Forest).**
* **`neural_oneshot.py` : Aplicação oneshot do modelo neural (sem dados do ENSO).**
* **`neural_finetune.py` : Treinamento do modelo com dados do ENSO.**
* **`neural_finetune_exog.py` : Execução do modelo treinado com dados do ENSO.**
##### ⚠️ Execução do modelo neural com aceleração da GPU requer placa da NVIDIA no Windows ou AMD no Linux.

### Bibliotecas utilizadas (Python 3.12)

| Biblioteca  | Versão |
| ------------- |:-------------:|
| numpy | 1.26.4
| pandas | 2.3.1
| scikit-learn | 1.7.0
| geopandas | 1.1.1
| statsmodels | 0.14.5
| matplotlib | 3.10.3
| transformers | 4.53.1
| torch | 2.7.1
| netcdf4 | 1.7.2
| cdsapi | 0.7.6
| granite-tsfm | 0.2.22
