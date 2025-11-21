# importanto bibliotecas 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns       

from google.colab import drive
drive.mount('/content/drive') 
path = '/content/drive/MyDrive/atividade3-1'

df = pd.read_csv(path + '/Housing.csv')

# Análisando o dataset
df.head()
# Análise mais completa 
df.info()

# Contar valores ausentes por coluna
df.isnull().sum()

# Calcular a porcentagem de valores ausentes por coluna
(df.isnull().mean() * 100).round(2)

#  Visualize os padrões de valores ausentes (Sugestão: explore  a biblioteca missingno)
import missingno as msno
import matplotlib.pyplot as plt

msno.matrix(df)
plt.show()

msno.bar(df)
plt.show()
msno.heatmap(df)
plt.show()

# Implemente três estratégias diferentes para lidar com valores ausentes: 
 # Remoção de linhas com valores ausentes em colunas específicas 
 # Imputação usando média/mediana para colunas numéricas 
 # Imputação usando KNN para colunas numéricas (Consulte o site da Scikit 
# Learn) 
# Remoção de linhas com valores ausentes em colunas específicas
df_removido = df.dropna(subset=['price', 'area', 'bedrooms'])

# Exibindo o resultado
print(df_removido.shape)
df_removido.head()

# Usando média
df_media = df.copy()
df_media['area'] = df_media['area'].fillna(df_media['area'].mean())
df_media['price'] = df_media['price'].fillna(df_media['price'].mean())

# imputação usando KNN
from sklearn.impute import KNNImputer
import pandas as pd

# Seleciona apenas colunas numéricas
df_knn = df.select_dtypes(include=['int64', 'float64']).copy()

# Cria o imputador KNN
imputer = KNNImputer(n_neighbors=5)

# Aplica o KNN
df_knn_imputado = imputer.fit_transform(df_knn)

# Converte de volta ao DataFrame
df_knn = pd.DataFrame(df_knn_imputado, columns=df_knn.columns)

df_knn.head()
