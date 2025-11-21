# Conexão com drive
import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/aula-23 09'

df = pd.read_csv(path + '/boardgame-geek-dataset_organized.csv')

# Mostrando as 5 primeiras linhas do dataframe
df.head()
# Verificar tipos e valores ausentes
df.info()
# Estatísticas descritivas de todas as colunas numéricas
df.describe()


# Estatísticas de colunas específicas
df[['avg_rating', 'min_playtime', 'max_playtime']].describe()

# 1) Qual é a nota média de um jogo de tabuleiro neste dataset?
#    → A nota média (avg_rating) é aproximadamente 7.42.


# 2) Qual é o tempo de jogo médio e o desvio padrão?
#    → Tempo mínimo médio: 57.52 minutos (desvio padrão: ~48.65).
#    → Tempo máximo médio: 88.70 minutos (desvio padrão: ~89.12).


# 3) Existe alguma coluna com valores nulos que possa impactar análises futuras?
#    → Sim. Diversas colunas possuem valores ausentes, como:
#      'amazon_price', 'categories', 'mechanics', 'families', 'artists',
#      além de colunas com muitos nulos como 'solo_designers', 'developers',
#      'graphic_designers', 'sculptors', 'editors', 'writers', etc.
#    → Essas colunas podem impactar análises que dependam dessas informações.

# 1. Verifique a quantidade de valores nulos na coluna 'average_rating'.
df['avg_rating'].isnull().sum()
# Verificar quantos valores nulos existem na coluna
df['avg_rating'].isnull().sum()


# Calcular a mediana
mediana = df['avg_rating'].median()


# Preencher valores nulos com a mediana
df['avg_rating'] = df['avg_rating'].fillna(mediana)


# Escolhi preencher valores ausentes da coluna 'avg_rating' com a MEDIANA.
# A mediana é mais robusta contra outliers.
# a média poderia ser puxada por esses extremos e distorcer a análise.
# A mediana representa melhor o "valor típico" da coluna e mantém a distribuição mais estável.

# boxplot da coluna max_playtime
import matplotlib.pyplot as plt


plt.boxplot(df['max_playtime'])
plt.title('Boxplot – Tempo Máximo de Jogo (max_playtime)')
plt.ylabel('Minutos')
plt.show()


# 4. Calcule o IQR (Intervalo Interquartil) para a coluna 'playing_time' e identifique a quantidade de
# jogos que podem ser considerados outliers de tempo de jogo
# Calcular os quartis
Q1 = df['max_playtime'].quantile(0.25)
Q3 = df['max_playtime'].quantile(0.75)


# Calcular o IQR
IQR = Q3 - Q1


# Definir limites para outliers
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR


# Filtrar os outliers
outliers = df[(df['max_playtime'] < limite_inferior) | (df['max_playtime'] > limite_superior)]


# Quantidade de outliers
quantidade_outliers = outliers.shape[0]


# Exibir resultados
print("IQR:", IQR)
print("Limite inferior:", limite_inferior)
print("Limite superior:", limite_superior)
print("Quantidade de jogos considerados outliers:", quantidade_outliers)


# Crie um histograma da coluna 'average_rating' para visualizar a sua distribuição.
import matplotlib.pyplot as plt


plt.hist(df['avg_rating'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histograma – Distribuição de Avg Rating')
plt.xlabel('Average Rating')
plt.ylabel('Quantidade de Jogos')
plt.show()


# O histograma é assimétrico à direita porque a maior parte dos dados está
# concentrada na esquerda, e a distribuição apresenta uma cauda mais longa e estendida em
# direção aos valores mais altos, à direita.


# Aplique a transformação logarítmica na coluna 'max_playtime'
import numpy as np


# Criar uma nova coluna com a transformação logarítmica
df['log_max_playtime'] = np.log1p(df['max_playtime'])


# Mostrar as primeiras linhas para conferir
df[['max_playtime', 'log_max_playtime']].head()


# Histograma original
plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.hist(df['max_playtime'], bins=30, color='skyblue', edgecolor='black')
plt.title('Histograma Original – max_playtime')
plt.xlabel('Max Playtime (minutos)')
plt.ylabel('Quantidade de Jogos')


# Histograma log-transformado
plt.subplot(1,2,2)
plt.hist(df['log_max_playtime'], bins=30, color='lightgreen', edgecolor='black')
plt.title('Histograma Log-transformado – log_max_playtime')
plt.xlabel('log(1 + Max Playtime)')
plt.ylabel('Quantidade de Jogos')


plt.tight_layout()
plt.show()


# Crie um gráfico de dispersão (scatter plot) com 'min_players' no eixo x e 'max_players' no eixo y
plt.figure(figsize=(8,6))
plt.scatter(df['min_players'], df['max_players'], alpha=0.6, color='purple')
plt.title('Scatter Plot – Min Players x Max Players')
plt.xlabel('Min Players')
plt.ylabel('Max Players')
plt.grid(True)
plt.show()


# Visualmente, não é possível identificar uma tendência ou correlação linear clara e forte entre o número mínimo e o máximo de jogadores.




# Calcule a matriz de correlação de todas as colunas numéricas do seu DataFrame.


df_numerico = df.select_dtypes(include=['number'])
matriz_correlacao = df_numerico.corr()


import seaborn as sns
import matplotlib.pyplot as plt


# Configurar o tamanho do gráfico
plt.figure(figsize=(15,12))


# Criar o heatmap
sns.heatmap(matriz_correlacao, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)


plt.title('Heatmap – Matriz de Correlação das Colunas Numéricas')
plt.show()


# Identifique a coluna que representa a data de publicação do jogo ('yearpublished').
# 'release_year'


# Crie uma nova coluna chamada 'Decada' que categorize os jogos por década de publicação (ex:
# 1990, 2000, 2010).


# Criar a coluna 'Decada'
df['Decada'] = (df['release_year'] // 10) * 10


# Mostrar as primeiras linhas para conferir
df[['release_year', 'Decada']].head()


#  Faça a contagem de jogos lançados por década (agrupando por 'Decada') para entender a
# tendência de produção de jogos de tabuleiro ao longo do tempo.
contagem_decadas = df.groupby('Decada')['release_year'].count()
print(contagem_decadas)
# A década com o maior número de lançamentos foi a de 2010,
# com um total de 1031 jogos lançados.
