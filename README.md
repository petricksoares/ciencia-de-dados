# 📊 Projeto de Ciência de Dados – Análise e Pré-processamento de Datasets
Este projeto consiste em uma análise exploratória e pré-processamento de datasets reais e públicos, com foco em técnicas de limpeza, padronização e normalização de dados.
O objetivo é aplicar conceitos fundamentais de Ciência de Dados, preparando os dados para análises estatísticas, visualizações e modelagem preditiva.

## 🔧 Técnicas de Pré-processamento Aplicadas
### Padronização de colunas e categorias
No projeto, realizei um pré-processamento dos dados para garantir que estivessem consistentes e prontos para análise. Inicialmente, padronizei colunas e categorias, renomeando colunas com nomes claros e intuitivos, corrigindo capitalizações inconsistentes e removendo espaços extras. Também padronizei valores divergentes, como transformar “United-States” em “United States”, evitando inconsistências que poderiam prejudicar a análise.

### Tratamento de valores ausentes
O tratamento de valores ausentes foi uma etapa essencial. Substituí os valores “?” por NaN e identifiquei quais colunas apresentavam dados faltantes. Para lidar com esses casos, apliquei diferentes estratégias: removi linhas em colunas críticas quando necessário, utilizei imputação com média, mediana ou moda, e, em casos de dados numéricos correlacionados, usei imputação via KNN. Essas abordagens garantiram que os dados ficassem completos sem introduzir vieses significativos.

### Análise exploratória e estatística
Realizei também uma análise exploratória para entender a estrutura e distribuição dos dados. Visualizei as primeiras linhas do dataset e sua estrutura geral, gerei estatísticas descritivas e identifiquei outliers por meio de boxplots e do cálculo do IQR. Além disso, explorei distribuições e relações entre variáveis usando histogramas e scatter plots, enquanto mapas de calor (heatmaps) ajudaram a entender correlações entre colunas numéricas. Para verificar padrões de dados ausentes, utilizei a biblioteca missingno.

### Normalização de dados numéricos
Para preparar os dados numéricos para algoritmos de machine learning, apliquei técnicas de normalização e padronização. Usei Min-Max Scaling (que ajusta os valores para o intervalo entre 0 e 1), Z-score Standardization (que transforma os dados para média 0 e desvio padrão 1) e Robust Scaling (que utiliza a mediana e o IQR, sendo mais resistente a outliers).

## 📊 Visualizações
Também realizei análises categóricas e agrupamentos, identificando padrões e tendências nos dados. Durante o processo, algumas visualizações se mostraram especialmente úteis:
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
