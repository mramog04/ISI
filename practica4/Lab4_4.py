import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('iris.csv')
points_setosa = {}
points_versicolor = {}
points_virginica = {}
points = {}

for i in range(df.shape[0]):
    x = df.iloc[i, 0]
    y = df.iloc[i, 1]
    z = df.iloc[i, 2]
    k = df.iloc[i, 3]
    if df.iloc[i, 4] == 'Setosa':
        points_setosa[f'Point_{i+1}'] = {'x': x, 'y': y, 'z': z, 'k': k,'color':'purple'}
    elif df.iloc[i,4] == 'Versicolor':  
        points_versicolor[f'Point_{i+1}'] = {'x': x, 'y': y, 'z': z, 'k': k,'color':'blue'}
    elif df.iloc[i,4] == 'Virginica':  
        points_virginica[f'Point_{i+1}'] = {'x': x, 'y': y, 'z': z, 'k': k,'color':'green'}
    else:
        points[f'Point_{i+1}'] = {'x': x, 'y': y, 'z': z, 'k': k,'color':'red'}
    
    
df_dic_setosa = pd.DataFrame.from_dict(points_setosa, orient='index')
df_dic_versicolor = pd.DataFrame.from_dict(points_versicolor, orient='index')
df_dic_virginica = pd.DataFrame.from_dict(points_virginica, orient='index')


fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': '3d'})

ax.scatter(df_dic_setosa['x'],df_dic_setosa['y'],df_dic_setosa['z'],s=df_dic_setosa['k']*30,c=df_dic_setosa['color'],alpha=0.6,label="Setosa")
ax.scatter(df_dic_versicolor['x'],df_dic_versicolor['y'],df_dic_versicolor['z'],s=df_dic_versicolor['k']*30,c=df_dic_versicolor['color'],alpha=0.6,label="Versicolor")
ax.scatter(df_dic_virginica['x'],df_dic_virginica['y'],df_dic_virginica['z'],s=df_dic_virginica['k']*30,c=df_dic_virginica['color'],alpha=0.6,label="Virginica")

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_zlabel('Petal Length')


ax.legend(['Setosa', 'Versicolor', 'Virginica'])



plt.show()  

