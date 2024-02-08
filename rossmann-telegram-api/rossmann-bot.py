import pandas as pd
import json
import requests
from flask import Flask, request, Response
import os

# constants
TOKEN = '6958861406:AAGz0JIutwDuA4L7eQhmYNUBup3WcM0CmZA'

# Info about the bot
# https://api.telegram.org/bot6958861406:AAGz0JIutwDuA4L7eQhmYNUBup3WcM0CmZA/getMe

# endpoint: get updates
# https://api.telegram.org/bot6958861406:AAGz0JIutwDuA4L7eQhmYNUBup3WcM0CmZA/getUpdates

# endpoint: send message
# https://api.telegram.org/bot6958861406:AAGz0JIutwDuA4L7eQhmYNUBup3WcM0CmZA/sendMessage?chat_id=828489427&text=Hi Felipao, i am doing fine!

def send_message( chat_id, text ):
    url = f'https://api.telegram.org/bot{TOKEN}'
    url = url + f'/sendMessage?chat_id={chat_id}'

    r = requests.post( url, json={'text': text})
    print(f'Status Code {r.status_code}')

    return None


def load_dataset( store_id ):
    # loading test dataset
    df10 = pd.read_csv('C:/Users/Felipe/OneDrive/Documentos/repos/ds_em_producao_mentoria/data/test.csv')
    df_stores_raw = pd.read_csv('C:/Users/Felipe/OneDrive/Documentos/repos/ds_em_producao_mentoria/data/store.csv')

    # merging test dataset with store dataset
    df_test = pd.merge(df10, df_stores_raw, how='left', on='Store')

    # choose store for prediction
    df_test = df_test.loc[ df_test['Store'] == store_id, :]

    if not df_test.empty:
        # remove closed days
        df_test = df_test.loc[ df_test['Open'] != 0, :]
        df_test = df_test.loc[ ~df_test['Open'].isnull(), : ]
        df_test = df_test.drop('Id', axis=1)

        # convert DataFrame to json
        data = json.dumps( df_test.to_dict(orient='records'))
    else:
        data = 'error'

    return data

def predict( data ):
    # API Call
    url = 'https://rossmann-sales-o2or.onrender.com/rossmann/predict'
    header = { 'Content-type': 'application/json'}
    data = data

    r = requests.post(url, data=data, headers=header)
    print(f'Status Code {r.status_code}')

    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1



def parse_message( message ):
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']

    store_id = store_id.replace('/', '')

    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'

    return chat_id, store_id

# API initialize
app = Flask(__name__)

@app.route( '/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        message = request.get_json()

        chat_id, store_id = parse_message( message )

        if store_id != 'error':
            # loading data
            data = load_dataset( store_id )

            if data != 'error':
                # prediction
                d1 = predict( data )

                # calculation
                d2 = d1.loc[ :, ['store', 'prediction']].groupby('store').sum().reset_index()


                msg = f'Store Number {d2.loc["store"].values[0]} will  sell {d2.loc["prediction"].values[0]:,.2f} in the next 6 weeks'

                send_message(chat_id, msg)
                return Response('OK', status=200)

            else: 
                send_message(chat_id, 'Store Not Available')
                return Response('OK', status=200)
        
        else:
            send_message(chat_id, 'Store ID is wrong')
            return Response('OK', status=200)
    else:
        return '<h1> Rossmann Telegram BOT </h1>'


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( host='0.0.0.0', port=port)

