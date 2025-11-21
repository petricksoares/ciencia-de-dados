import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive') 
path = '/content/drive/MyDrive/atividade3-4'
df = pd.read_csv(path + '/adult/adult.data')

# visualização de dados 
df.head()
df.info()

# colocando nome as colunas 
colunas = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]

df = pd.read_csv(path + '/adult/adult.data', names=colunas)
df.head()

# Tratamento de valores categoricos
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].str.strip()      
    df[col] = df[col].str.title()

# Padronização de valores categoricos
df['native_country'] = df['native_country'].replace({
    'United-States': 'United States'
})

# Tratar valores ausentes 
df.replace('?', pd.NA, inplace=True)
df.info()      # verifica valores ausentes
df.head()      # verifica nomes e padronização

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Selecionar apenas colunas numéricas
num_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

# Criar cópias do dataset
df_minmax = df.copy()
df_zscore = df.copy()
df_robust = df.copy()

# -----------------------------
# 1) Min-Max Scaling
# Transforma os valores para o intervalo [0,1]
minmax_scaler = MinMaxScaler()
df_minmax[num_cols] = minmax_scaler.fit_transform(df_minmax[num_cols])

# -----------------------------
# 2) Z-score Standardization
# Centraliza nos 0 e escala para desvio padrão 1
zscore_scaler = StandardScaler()
df_zscore[num_cols] = zscore_scaler.fit_transform(df_zscore[num_cols])

# -----------------------------
# 3) Robust Scaling
# Usa mediana e IQR, é robusto a outliers
robust_scaler = RobustScaler()
df_robust[num_cols] = robust_scaler.fit_transform(df_robust[num_cols])


# Ver primeiras linhas após Min-Max
df_minmax[num_cols].head()
# Ver primeiras linhas após Robust
df_robust[num_cols].head()
# Ver primeiras linhas após Z-score
df_zscore[num_cols].head()
