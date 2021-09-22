#Importar  librerias a utilizar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#Importar la información de los archivos CSV

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Analizar los datos de los archivos para ver como "acomodarlos" para que funcione bien la clasificacion.

#Ver los primeros 3 de cada archivo para ver que secciones tienen
# print(df_train.head(3))
# print(df_test.head(3))

#Tipos de datos que tienen los archivos
#print(df_train.info())
#print(df_test.info())

#Verificar si existen datos faltantes y se suman, para saber cuantos son de cada tipo
#print(pd.isnull(df_train).sum())
#print(pd.isnull(df_test).sum())


#Verificamos que datos hay en los objetos para sustituirlos por numeros (en caso de que se puedan agregar) para poder continuar
#print(df_train['Sex'].head())
#print(df_train['Embarked'].head())

#Sustituimos los datos anteriores por numeros
df_train['Sex'].replace(['female','male'], [0,1], inplace=True)
df_test['Sex'].replace(['female','male'], [0,1], inplace=True)

df_train['Embarked'].replace(['Q','S','C'], [0,1,2], inplace=True)
df_test['Embarked'].replace(['Q','S','C'], [0,1,2], inplace=True)

#Existen datos faltantes en la edad, asi que pondremos la media de edad.
promedio = (( (df_train['Age'].mean()) + (df_test['Age'].mean()) ) / 2).round()
promedio = int(promedio)
#print(promedio)


#Se cambia los nan por el valor del promedio
df_train['Age'] = df_train['Age'].replace(np.nan, promedio)
df_test['Age'] = df_test['Age'].replace(np.nan, promedio)


#Muestra una grafica para saber de que edades sobrevivieron
a = sns.FacetGrid(df_train, col='Survived')
a.map(plt.hist, 'Age', bins=20)


#Relacionar la edad y su clase para ver si sobrevivieron, para determinar si la edad influye
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend()
#plt.show()

#Podemos combinar sibSp y Parch ya que el primero es sobre hermanos y esposos y el segundo sobre padres y niños, se podria combinar en familia
all_data =[df_train, df_test]
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

#print(df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

#Se puede crear otra columna llamada IsAlone, para posteriormente solo dejar esta para saber si esta solo o iba con alguien

for dataset in all_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

#print(df_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# Eliminamos las columnas de sibSp, Parch y familysize para dejar solo la de IsAlone
df_train = df_train.drop(['Parch','SibSp','FamilySize'], axis=1)
df_test = df_test.drop(['Parch','SibSp','FamilySize'], axis=1)
all_data = [df_train, df_test]

#print(df_train.head())

#La columna Fare se reduce a 4 opciones para no tener muchas opciones diferentes
df_test['Fare'].fillna(df_test['Fare'].dropna().median(), inplace=True)
#print(df_test.head())

df_train['FareBand'] = pd.qcut(df_train['Fare'],4)
#print(df_train[['FareBand','Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand',ascending=True))

for dataset in all_data:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = df_train.drop(['FareBand'],axis=1)




#Eliminamos todas las columnas que consideramos que no tienen relacion
df_train = df_train.drop(['Ticket','Cabin'], axis=1)
df_test = df_test.drop(['Ticket','Cabin'], axis=1)


#print(df_train.head(10))
#print(df_test.head(10))

#Extraemos titulos personales para correlacionarlos con supervivencia
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#print(pd.crosstab(df_train['Title'], df_train['Sex']))

all_data = [df_train, df_test]

#Se reemplazan titulos con un nombre mas comun o clasificarlos como raros

for dataset in all_data:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col', 'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')

    dataset['Title'] =dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] =dataset['Title'].replace('Ms','Miss')
    dataset['Title'] =dataset['Title'].replace('Mme','Mrs')

#print(df_train[['Title','Survived']].groupby(['Title'], as_index=False).mean())

#Se puede convertir los titulos a numeros
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in all_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

#print(df_train.head())

# #Se elimina PassangerId y Name
df_train = df_train.drop(['Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Name'],axis=1)
all_data = [df_train, df_test]
#print(df_train.shape, df_test.shape)



# #Se crean grupos de edad en base a bandas de las edades
# 0-8 , 9-15, 16-18, 19-25, 26-40, 41-60, 61-100
bins = [0, 8, 15, 18, 25, 40, 60, 100]
names= ['1','2','3','4','5','6','7']
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)
df_test['Age'] = pd.cut(df_test['Age'],bins, labels = names)

# #Eliminar filas con datos vacios
df_train.dropna(axis=0, how='any', inplace=True)
df_test.dropna(axis=0, how='any', inplace=True)

# #Verificar los datos
# print(pd.isnull(df_train).sum())
# print(pd.isnull(df_test).sum())

# print(df_train.shape)
# print(df_test.shape)

# print(df_test.head())
# print(df_train.head())

# #Separar la columna con la informacion de los sobrevivientes
X = np.array(df_train.drop(['Survived'], axis=1))
y = np.array(df_train['Survived'])


# #Separar los datos de train en entrenamiento y prueba para probar los algoritmos
X_train = np.array(df_train.drop("Survived",axis=1))
y_train = df_train["Survived"]
X_test = np.array(df_test.drop("PassengerId",axis=1).copy())
# print(X_train.shape, y_train.shape, X_test.shape)

# #REGRESION LOGISTICA
logreg = LogisticRegression()
#Se entrena
logreg.fit(X_train, y_train)
#Precision
y_pred_logreg = logreg.predict(X_test)
print('Regresión Logistica: %.2f' % logreg.score(X_train, y_train))

# #SVC (SUPPORT VECTOR MACHINES)
svc = SVC()
#Entrenar
svc.fit(X_train, y_train)
#Precision
y_pred_svc = svc.predict(X_test)
print('SVC: %.2f' % svc.score(X_train, y_train))

# #K NEIGHBORS
knn = KNeighborsClassifier()
#Entrenar
knn.fit(X_train, y_train)
#Precision
y_pred_knn = knn.predict(X_test)
print('K Neighbors: %.2f' % knn.score(X_train, y_train))

# #DECISION TREE
dt = DecisionTreeClassifier()
#Entrenar
dt.fit(X_train, y_train)
#Precision
y_pred_dt = dt.predict(X_test)
print('Decision Tree: %.2f' % dt.score(X_train, y_train))

# #RANDOM FOREST
rf = RandomForestClassifier()
#Entrenar
rf.fit(X_train, y_train)
#Precision
y_pred_rf = rf.predict(X_test)
print('Random Forest: %.2f' % rf.score(X_train, y_train))

# #Prediccion utilizando los modelos
test = pd.read_csv('test.csv')
result = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": y_pred_dt

})
result.to_csv('result.csv', index=False)













