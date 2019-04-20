# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 23:29:01 2019

@author: Bruno Ferraz Silveira
"""

import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler

# leitura do arquivo csv e joga isso na variável base
base = pd.read_csv('credit-data.csv')

# para mostrar os dados dessa variável
# base.describe()

# faz a busca na coluna age por valores menores que 0 (zero)
# base.loc[base['age'] < 0]

# apagar a coluna
# base.drop('age', 1, inplace=True)

# apagar somente os registros com problema
# base.drop(base[base.age < 0].index, inplace=True)

# preencher os valores manualmente

# preencher os valores com a média
# base.mean() pega a média, base['age'].mean() pega a média na coluna age
base['age'].mean()

# base['age'][base.age > 0].mean() pega a média na coluna age ods valores positivos
# pega a média da coluna age dos valores positivos e atribui aos valores que são negativos na coluna age
base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

# pega valores nulos da coluna age
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

# separando em arrays as colunas previsoras e a coluna de classificação
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# padroniza dados para que não hajam diferenças discrepantes e os algorítmos baseados em distâncias (como distância euclidiana) não deem mais importância a algum valor por ter uma diferença maior de valores
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# colocando a média no lugar de valores NaN
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
