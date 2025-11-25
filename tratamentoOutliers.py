# conectando com drive 
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/tratamentoOut'
import pandas as pd
df = pd.read_csv(path+'/Finance_data.csv')

# Entendendo melhor os dados
df.head()
df.info()
df.describe()

# Filtrar colunas numéricas
numeric_cols = df.select_dtypes(include=['int64'])

# MÉTODO Z-SCORE
from scipy import stats
import numpy as np

z_scores = np.abs(stats.zscore(numeric_cols))
outliers_z = (z_scores > 3)

print("Outliers por Z-score (quantidade por coluna):")
print(outliers_z.sum())

df_outliers_z = df[outliers_z.any(axis=1)]
print("\nRegistros com pelo menos 1 outlier (Z-score):")
df_outliers_z

# MÉTODO IQR
Q1 = numeric_cols.quantile(0.25)
Q3 = numeric_cols.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers_iqr = (numeric_cols < lower) | (numeric_cols > upper)

print("\nOutliers por IQR (quantidade por coluna):")
print(outliers_iqr.sum())

df_outliers_iqr = df[outliers_iqr.any(axis=1)]
print("\nRegistros com pelo menos 1 outlier (IQR):")
df_outliers_iqr

# Gerando BOXPLOTS – Outliers por coluna
import matplotlib.pyplot as plt
import seaborn as sns

# Selecionando apenas colunas numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Criando boxplots
plt.figure(figsize=(14, 8))
df[numeric_cols].boxplot()
plt.title("Boxplot das Colunas Numéricas")
plt.xticks(rotation=45)
plt.show()

# BOXPLOTS individuais (um por coluna)
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot de {col}")
    plt.show()

# SCATTERPLOTS destacando outliers pelo Z-score
from scipy import stats
import numpy as np

# Calculando z-score
z_scores = np.abs(stats.zscore(df[numeric_cols]))

# Scatterplot para cada coluna numérica marcando outliers
for i, col in enumerate(numeric_cols):
    plt.figure(figsize=(6, 4))

    plt.scatter(range(len(df[col])), df[col], label="Dados")
    plt.scatter(np.where(z_scores[:, i] > 3), 
                df[col][z_scores[:, i] > 3],
                color='red', 
                label='Outliers')

    plt.title(f"Scatterplot de {col} com Outliers (Z-score)")
    plt.xlabel("Índice")
    plt.ylabel(col)
    plt.legend()
    plt.show()

# Remoção de Outliers usando IQR
df_iqr = df.copy()

for col in numeric_cols:
    Q1 = df_iqr[col].quantile(0.25)
    Q3 = df_iqr[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_iqr = df_iqr[(df_iqr[col] >= lower) & (df_iqr[col] <= upper)]

print("Formato após IQR:", df_iqr.shape)

# Capping (substituição por valores limites)
df_capped = df.copy()

for col in numeric_cols:
    Q1 = df_capped[col].quantile(0.25)
    Q3 = df_capped[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Substitui outliers pelos limites
    df_capped[col] = np.where(df_capped[col] < lower, lower, df_capped[col])
    df_capped[col] = np.where(df_capped[col] > upper, upper, df_capped[col])

print("Capping concluído!")

# Transformação logarítmica
df_log = df.copy()

for col in numeric_cols:
    # Evita erro de log(0)
    df_log[col] = np.log1p(df_log[col])  

df_log.head()

# Transformação raiz quadrada
df_sqrt = df.copy()

for col in numeric_cols:
    df_sqrt[col] = np.sqrt(df_sqrt[col])

df_sqrt.head()
