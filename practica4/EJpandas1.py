import pandas as pd
series_exmaple = pd.Series([10,20,30,40])
print(series_exmaple)
data = {'Name': ['Victor','Rocio','Fran','Fran'],'Age': [25,30,35,35]}
df = pd.DataFrame(data)
print(df)
df_2 = pd.read_csv('data.csv')
print(df_2)
""" df.head() returns the first 5 rows of the dataframe"""
""" df.tail() returns the last 5 rows of the dataframe"""
""" df.head(10) returns the first 10 rows of the dataframe"""
""" df.tail(10) returns the last 10 rows of the dataframe"""
""" df.shape returns the number of rows and columns of the dataframe"""
""" df.columns returns the column names of the dataframe"""
""" df.dtypes returns the data types of each column of the dataframe"""
""" df.info() returns a summary of the dataframe"""
""" df.describe() returns a statistical summary of the dataframe"""
""" df.dropna() removes rows with missing values"""
""" df.fillna(value) replaces missing values with a value"""
""" df.drop_duplicates() removes duplicate rows"""
""" df.groupby(column) groups rows by a column"""
""" df.sort_values(column) sorts the rows by a column"""
""" df.pivot_table(values='D', index=['A', 'B'], columns=['C']) creates a pivot table"""
""" df.loc[row_indexer,column_indexer] selects a subset of rows and columns"""
""" df.iloc[row_indexer,column_indexer] selects a subset of rows and columns by integer location"""
