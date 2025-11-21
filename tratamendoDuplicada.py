# Carregando bibliotecas 
import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/atividade3-2'
df = pd.read_excel(path + '/KPMG_VI_New_raw_data_update_final.xlsx')

# 5 primeiras linhas 
df.head()
# Informações iniciais
df.info()

# Correção de carregamento
xls = pd.ExcelFile(path + '/KPMG_VI_New_raw_data_update_final.xlsx')
xls.sheet_names

path = '/content/drive/MyDrive/atividade3-2/KPMG_VI_New_raw_data_update_final.xlsx'

xls = pd.ExcelFile(path)
print(xls.sheet_names)
# Exemplo: carregar aba 'CustomerAddress'
df = pd.read_excel(path, sheet_name='CustomerAddress', header=0)  # header=0 se a primeira linha contém os nomes das colunas

# Verificar duplicatas exatas
exact_duplicates = df.duplicated().sum()
print(f"Número de duplicatas exatas: {exact_duplicates}")

# Remover duplicatas exatas
df_no_exact_duplicates = df.drop_duplicates()

# TODO: Implementar detecção de duplicatas aproximadas
# Dica: Criar uma função para comparar strings usando fuzzywuzzy
def fuzzy_match(str1, str2, threshold=80):
    return fuzz.ratio(str1, str2) >= threshold
