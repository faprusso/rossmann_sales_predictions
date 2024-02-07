import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann( object):
    def __init__( self ):
        self.home_path = ''
        # pickle
        self.competition_distance_scaler   = pickle.load(open( self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open( self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open( self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open( self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open( self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    
    def data_cleaning( self, df1):
        print('data_cleaning')
        # rename columns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek','Promo2SinceYear', 'PromoInterval']

        #transformando nome de colunas em snakecase
        snakecase = lambda x: inflection.underscore( x )
        cols_new = list(map(snakecase, cols_old))

        #rename
        df1.columns = cols_new

        # Data Types
        df1['date'] = pd.to_datetime(df1['date'])

        # Fillout NA
        #competition_distance -> preenchendo o valor de competition distance com valor elevado (200000) se o valor for nan
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan(x) else x)   

        #competition_open_since_month -> aplicando a mês da data de venda corrente para a coluna 'competition_open_since_month', caso o valor seja nan
        # se ao fazer a subtração de meses para saber quando tempo o concorrente está aberto, vai dar zero.
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis =1)

        #competition_open_since_year -> mesma lógica acima
        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis =1)
                        
        #promo2_since_week -> mesma lógica acima, mas para a promo2 (pega a semana da data 'date' e transforma em semana.)
        df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis =1)

        #promo2_since_year  -> mesma lógica acima
        df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis =1)         

        # limpando os NAs finais relativa as promos e continuação com a promo2 e seus intervalos
        # proposta é criar um array /lista da coluna 'promo_interval' para checar se o mês de venda está dentro desse intervalo
        #dicionario para converter meses numéricos em meses objetc (string)
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec',}
        # preenchendo NA com 0 -> se não tem a informação é que não participou
        df1['promo_interval'].fillna(0, inplace=True)
        # criando a coluna month_map -> pegando o mês da coluna "date" (data de vendas)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        #criando coluna 'is_promo' para checar se a coluna month_map está no intervalo 'promo_interval'
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis = 1)

        # Change Data Types
        # competition
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        # promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1
    
    def feature_engineering( self, df2):
        
        # Derivando as variáveis necessárias
        # year
        df2['year'] = df2['date'].dt.year 
        # month
        df2['month'] = df2['date'].dt.month
        # day
        df2['day'] = df2['date'].dt.day
        # week of year
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        # year week
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W')
        # competition since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime(year = x['competition_open_since_year'], month = x['competition_open_since_month'], day = 1), axis = 1 )
        df2['competition_time_month'] = ( (df2['date'] - df2['competition_since'])/30).apply( lambda x: x.days).astype(int)
        # promo since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ( (df2['date'] - df2['promo_since'])/7).apply( lambda x: x.days).astype(int)
        # assortment
        df2['assortment'] = df2['assortment'].apply( lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        # state holiday
        df2['state_holiday'] = df2['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 'easter_holiday' if x == 'b' else 'christmas' if x == 'c' else 'regular_day')

        # Filtragem de Variáveis
        #queremos apenas as linhas de vendas > 0 e quando a loja está aberta (open != 0) para fazer a análise
        df2 = df2[ (df2['open'] != 0) ]

        # filtrando (dropando) colunas que não serão utilizadas na análise
        # customers é importante porém é uma restrição, uma vez que teríamos que criar um modelo de predição de clientes
        cols_drop = ['open', 'promo_interval', 'month_map']
        # axis = 0 -> linhas, axis = 1 -> colunas
        df2 = df2.drop(cols_drop, axis=1)

        return df2
    
    def data_preparation(self, df5):

        # Rescaling
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values)
        
        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values)
        
        # promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform( df5[['promo_time_week']].values)

        # year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values)

        #Encoding
        # state_holiday -> vamos usar o One Hot Encoding
        df5 = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'])

        # store_type -> vamos usar o Label Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'])
        
        # assortment -> vamos usar o Ordinal Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)

        # Nature Transformation
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * (2 * np.pi/7)))
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * (2 * np.pi/7)))
        # month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * (2 * np.pi/12)))
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * (2 * np.pi/12)))
        # day
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * (2 * np.pi/30)))
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * (2 * np.pi/30)))
        # week of year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * (2 * np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * (2 * np.pi/52)))

        cols_selected = ['store','promo','store_type','assortment','competition_distance','competition_open_since_month',
                         'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competition_time_month','promo_time_week','day_of_week_sin',
                         'day_of_week_cos','month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']                    

        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict( test_data)

        # join pred into the original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')

