# 📊 Projeto de Ciência de Dados – Análise e Pré-processamento de Datasets
Este projeto consiste em uma análise exploratória e pré-processamento de datasets reais e públicos, com foco em técnicas de limpeza, padronização e normalização de dados.
O objetivo é aplicar conceitos fundamentais de Ciência de Dados, preparando os dados para análises estatísticas, visualizações e modelagem preditiva.

## 🔧 Técnicas de Pré-processamento Aplicadas
### Padronização de colunas e categorias
* Renomeação de colunas com nomes claros.
* Correção de capitalização e remoção de espaços extras.
* Padronização de valores inconsistentes (ex.: United-States → United States).

### Tratamento de valores ausentes
* Conversão de valores "?" em NaN.
* Identificação de colunas com valores ausentes.
* Aplicação de estratégias:
* Remoção de linhas com nulos em colunas críticas.
* Imputação usando média, mediana ou moda.
* Imputação usando KNN para dados numéricos correlacionados.

### Análise exploratória e estatística
* Visualização das primeiras linhas (head()) e estrutura (info()).
* Estatísticas descritivas (describe()).
* Identificação de outliers via boxplots e IQR.
* Visualização de distribuições com histogramas e scatter plots.
* Mapas de calor de correlação (heatmap) para colunas numéricas.

### Normalização de dados numéricos
* Min-Max Scaling → valores entre 0 e 1.
* Z-score Standardization → média 0, desvio padrão 1.
* Robust Scaling → mediana 0, escala pelo IQR, robusto a outliers.
* Agrupamentos e análises categóricas

## 📊 Visualizações
* Boxplots para identificação de outliers.
* Histograma para análise de distribuição de colunas numéricas.
* Scatter plots para relações entre variáveis.
* Heatmaps de correlação entre variáveis numéricas.
* Mapas de missing values com missingno.

## ⚙️ Bibliotecas Utilizadas
* pandas – manipulação de dados
* numpy – cálculos numéricos
* matplotlib e seaborn – visualizações 
* scikit-learn – pré-processamento e normalização
* missingno – visualização de valores ausentes
