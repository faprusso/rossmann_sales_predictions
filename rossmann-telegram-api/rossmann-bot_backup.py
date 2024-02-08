import pandas as pd
import json
import requests



# loading test dataset
df10 = pd.read_csv('C:/Users/Felipe/OneDrive/Documentos/repos/ds_em_producao_mentoria/data/test.csv')
df_stores_raw = pd.read_csv('C:/Users/Felipe/OneDrive/Documentos/repos/ds_em_producao_mentoria/data/store.csv')

# merging test dataset with store dataset
df_test = pd.merge(df10, df_stores_raw, how='left', on='Store')

# choose store for prediction
df_test = df_test.loc[ df_test['Store'].isin( [24]), :]

# remove closed days
df_test = df_test.loc[ df_test['Open'] != 0, :]
df_test = df_test.loc[ ~df_test['Open'].isnull(), : ]
df_test = df_test.drop('Id', axis=1)

# convert DataFrame to json
data = json.dumps( df_test.to_dict(orient='records'))

# API Call
url = 'https://rossmann-sales-o2or.onrender.com/rossmann/predict'
# url = 'http://127.0.0.1:5000/rossmann/predict'
header = { 'Content-type': 'application/json'}
data = data

r = requests.post(url, data=data, headers=header)
print(f'Status Code {r.status_code}')

d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

d2 = d1.loc[ :, ['store', 'prediction']].groupby('store').sum().reset_index()

for i in range(len(d2)):
    print(f'Store Number {d2.loc[i, "store"]} will  sell {d2.loc[i, "prediction"]:,.2f} in the next 6 weeks')