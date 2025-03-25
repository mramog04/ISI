import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('company_sales_data.csv')

df_dic = {}
for i in range(df.shape[0]):
    df_dic[f'Month_{i+1}'] = {'month_number': df.iloc[i, 0], 'facecream': df.iloc[i, 1], 'facewash': df.iloc[i, 2]}
    
df_dic = pd.DataFrame.from_dict(df_dic, orient='index')

fig, ax = plt.subplots()
ax.bar(df_dic['month_number'], df_dic['facecream'],color='blue',width=-0.35, align='edge')
ax.bar(df_dic['month_number'], df_dic['facewash'],color='orange',width=0.35, align='edge')

ax.set_xlabel('Month Number')
ax.set_ylabel('Sales units in number')

ax.set_xticks(df_dic['month_number'])

ax.legend(['Facecream', 'Facewash'])

plt.show()