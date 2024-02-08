import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import pickle
import os

# loading model
model = pickle.load(open('C:/Users/Felipe/OneDrive/Documentos/repos/ds_em_producao_mentoria/api/model_rossmann.pkl', 'rb'))

app = Flask(__name__)

# defining endpoint com os métodos que recebe
@app.route('/rossmann/predict', methods=['POST'])

# after POST, endpoint will run rossmann_predict
def rossmann_predict():
    test_json = request.get_json()

    if test_json: # is there data?
        if isinstance(test_json, dict): # Unique Example
            test_raw = pd.DataFrame(test_json, index=[0])
        else: # multiple examples
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys())

        # Instantiate Rossmann class
        pipeline = Rossmann()

        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        # data preparation
        df3 = pipeline.data_preparation(df2)
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port,debug=False)
