from dash import Dash, html, dcc, dash, Input, Output
import plotly.express as px
import pandas as pd
import dash
from dash import html, dcc, ctx, callback
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
#--------------------------------------------------------------------------------------------------------------|
#                                       2nd Project                                                            |
#                                      André Ferreira                                                          |
#                                         Nª93574                                                              |
#                                      Energy Services                                                         |
#--------------------------------------------------------------------------------------------------------------|

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#                                               Retirar os dados
df_raw_data_19 = pd.read_csv('testData_2019_Central.csv')

df = pd.read_csv('df_Final')

#                        Tratamento dos dados de forma a que seja mais facil trabalhar com eles

#Passar para date.time e transformar em index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#adição de mais features para o user escolher
df['Power-1'] = df['Power (kw)'].shift(1) #

df['Hour'] = df.index.hour

# Ponto de separação
test_cutoff_date = '2019-01-01'

# Divisão do que será data de treino e data de test
df_data = df.loc[df.index < test_cutoff_date]  # 2017 e 2018
df_2019 = df.loc[df.index >= test_cutoff_date]  # 2019 (Na qual é a que vamos prever)

#Retirar linhas que não precisamos que não tem valores e restruturação da df
df_data = df_data.dropna()
df_data = df_data.iloc[:, [0, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
print(df_data.info())


# Separação da data
variables_19 = df_raw_data_19.columns[1:]

Date_19 = df_raw_data_19.iloc[:, 0]

real_data = df_raw_data_19.iloc[:, :2]

data_X_19 = df_raw_data_19.iloc[:, 1:]

#   X2(data para ser utilizada nos modelos 'fixos'
X2 = df_raw_data_19.iloc[:, 0:]
X2['Date'] = pd.to_datetime(X2['Date'])
X2.set_index('Date', inplace=True)
df_2019 = X2.copy()

# Mais features a adicionas mans neste caso relacionadas com o tempo
import holidays

pt_holidays = holidays.Portugal()


def Holiday(date):
    if date in pt_holidays:
        return 1
    else:
        return 0


X2['Holiday'] = X2.index.map(Holiday)

#mesmo cenário que em cima mas com diferente utilização
X2['Power-1'] = X2['Central (kWh)'].shift(1)
X2 = X2.dropna()

X2['Hour'] = X2.index.hour
X2 = X2.iloc[:, [10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
df_2019 = X2.copy()

#Y2(Dados Reais de consumo do edificio central)
y2 = X2.iloc[:, 0]
real_data = y2.reset_index()

#Melhorar os nomes das colunas
df_2019 = df_2019.rename(columns={'temp_C': 'Temp.(ºC)',
                                  'HR': 'RH(%)',
                                  'windSpeed_m/s': 'WindSpeed(m/s)',
                                  'windGust_m/s': 'WindGust(m/s)',
                                  'pres_mbar': 'press(mbar)',
                                  'solarRad_W/m2': 'SolarRad(W/m2)',
                                  'rain_mm/h': 'Rain(mm/h)',
                                  'rain_day': 'Rain(day)'})


X2.drop(columns=['Holiday'], inplace=True)

X2 = X2.values


#                                           linear Model
with open('LR_model.pkl', 'rb') as file:
    LR_model2 = pickle.load(file)

y2_pred_LR = LR_model2.predict(X2)
print(y2.size)
print(y2_pred_LR.size)

# Evaluate errors
MAE_LR = metrics.mean_absolute_error(y2, y2_pred_LR)
MBE_LR = np.mean(y2 - y2_pred_LR)
MSE_LR = metrics.mean_squared_error(y2, y2_pred_LR)
RMSE_LR = np.sqrt(metrics.mean_squared_error(y2, y2_pred_LR))
cvRMSE_LR = RMSE_LR / np.mean(y2)
NMBE_LR = MBE_LR / np.mean(y2)

#                                           Random forrest Model
with open('RF_model.pkl', 'rb') as file:
    RF_model2 = pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

# Evaluate errors
MAE_RF = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_RF = np.mean(y2 - y2_pred_RF)
MSE_RF = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_RF = RMSE_RF / np.mean(y2)
NMBE_RF = MBE_RF / np.mean(y2)

#                                                  GB Model
with open('GB_model.pkl', 'rb') as file:
    GB_model2 = pickle.load(file)

y2_pred_GB = GB_model2.predict(X2)

# Evaluate errors
MAE_GB = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_GB = np.mean(y2 - y2_pred_RF)
MSE_GB = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_GB = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_GB = RMSE_RF / np.mean(y2)
NMBE_GB = MBE_RF / np.mean(y2)

#                                           Decision Tree Model
with open('DT_regr_model.pkl', 'rb') as file:
    DT_regr_model = pickle.load(file)

y2_pred_DT = DT_regr_model.predict(X2)

# Evaluate errors
MAE_DT = metrics.mean_absolute_error(y2, y2_pred_DT)
MBE_DT = np.mean(y2 - y2_pred_DT)
MSE_DT = metrics.mean_squared_error(y2, y2_pred_DT)
RMSE_DT = np.sqrt(metrics.mean_squared_error(y2, y2_pred_DT))
cvRMSE_DT = RMSE_DT / np.mean(y2)
NMBE_DT = MBE_DT / np.mean(y2)

#                                           Bootstraping Model
with open('BT_model.pkl', 'rb') as file:
    BT_model2 = pickle.load(file)

y2_pred_BT = BT_model2.predict(X2)

# Evaluate errors
MAE_BT = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_BT = np.mean(y2 - y2_pred_RF)
MSE_BT = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_BT = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_BT = RMSE_RF / np.mean(y2)
NMBE_BT = MBE_RF / np.mean(y2)


#                          Criar quadros de dados com resultados de previsão e métricas de erro

d = {'Methods': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'Decision Tree', 'BootStraping'],
     'MAE': [MAE_LR, MAE_RF, MAE_GB, MAE_DT, MAE_BT],
     'MBE': [MBE_LR, MBE_RF, MBE_GB, MBE_DT, MBE_BT],
     'MSE': [MSE_LR, MSE_RF, MSE_GB, MSE_DT, MSE_BT],
     'RMSE': [RMSE_LR, RMSE_RF, RMSE_GB, RMSE_DT, RMSE_BT],
     'cvMSE': [cvRMSE_LR, cvRMSE_RF, cvRMSE_GB, cvRMSE_DT, cvRMSE_BT],
     'NMBE': [NMBE_LR, NMBE_RF, NMBE_GB, NMBE_DT, NMBE_BT]}

df_metrics = pd.DataFrame(data=d)

d = {'Date': real_data['Date'].values,
     'LinearRegression': y2_pred_LR,
     'RandomForest': y2_pred_RF,
     'Gradient Boosting': y2_pred_GB,
     'Decision Tree': y2_pred_DT,
     'BootStraping': y2_pred_BT}

df_forecast = pd.DataFrame(data=d)


df_results = pd.merge(real_data, df_forecast, on='Date')
df = pd.merge(real_data, df_data, on='Date')


fig = px.line(df, x='Date', y=df.columns[1])

fig2 = px.line(df_results, x=df_results.columns[0], y=df_results.columns[1:7])

df_data.reset_index(drop=True, inplace=True)


# Funcões utilizads para a 4 Tab, na qual servem para mudar os forecast que vão ser escolhidos pelos utilizadores
def perform_forecast(X_train, Y_train, forecast_method, X_test):

    y_pred = None

    if forecast_method == 'Linear Regression':

        regr = linear_model.LinearRegression()

        regr.fit(X_train, Y_train)

        y_pred = regr.predict(X_test)

    elif forecast_method == 'Decision Tree':

        DT_regr_model = DecisionTreeRegressor()

        DT_regr_model.fit(X_train, Y_train)

        y_pred = DT_regr_model.predict(X_test)

    elif forecast_method == 'Gradient Boosting':

        GB_model = GradientBoostingRegressor()

        GB_model.fit(X_train, Y_train)

        y_pred = GB_model.predict(X_test)

    elif forecast_method == 'BootStraping':

        BT_model = BaggingRegressor()

        BT_model.fit(X_train, Y_train)

        y_pred = BT_model.predict(X_test)

    elif forecast_method == 'Random Forrest':

        parameters = {'bootstrap': True,
                      'min_samples_leaf': 3,
                      'n_estimators': 200,
                      'min_samples_split': 15,
                      'max_features': 'sqrt',
                      'max_depth': 20,
                      'max_leaf_nodes': None}

        RF_model = RandomForestRegressor(**parameters)

        RF_model.fit(X_train, Y_train)
        y_pred = RF_model.predict(X_test)

    return y_pred

def generate_table(dataframe, max_rows=11):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


#                                               Layout Do DashBoard
app = Dash(__name__, external_stylesheets=external_stylesheets)
server =app.server

app.layout = html.Div(children=[
    html.H1(children='2nd Project '),
    html.Div(children='''
        André Ferreira, Nº93574
    '''),
    dcc.Tabs([

        dcc.Tab(label='Raw variables', children=[
            html.H2(children='Data '),
            html.Label('Raw Variables'),
            dcc.Dropdown(
                id='Raw Variables',
                options=[{'label': i, 'value': i} for i in variables_19],
                value='Central (kWh)',
                multi=True
            ),

            html.Div([
                dcc.Graph(id='2019-data')
            ], style={'width': '100%', 'display': 'inline-block', 'padding': '0 20'}),
        ], style={'columnCount': 1}),

        dcc.Tab(label='Forecast', children=[
            html.Div([
                html.H4('IST Electricity Forecast (kWh)'),
                dcc.Graph(
                    id='yearly-data',
                    figure=fig2,
                ),

            ])

        ]),

        dcc.Tab(label='Metrics', children=[
            html.Div([
                html.H2(children='Metrics'),
                html.H4('IST Electricity Forecast Error Metrics'),
                generate_table(df_metrics)
            ])
        ]),

        dcc.Tab(label='Make your own forecast', children=[
            html.Div([
                html.H2(children='Make your own forecast'),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': col, 'value': col} for col in df_data.columns[1:]],
                    value=[df_data.columns[1]],
                    multi=True
                ),
                dcc.Dropdown(
                    id='table-shape-dropdown',
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'IQR', 'value': 'IQR'},
                        {'label': 'Z_Score', 'value': 'Z_Score'},
                        {'label': 'Interpole', 'value': 'Interpole'}
                    ],
                    value='None',  # Default value is 'Original'
                    clearable=False
                ),
                dcc.Dropdown(
                    id='Forecast-dropdown',
                    options=[
                        {'label': 'None', 'value': 'None'},
                        {'label': 'Linear Regression', 'value': 'Linear Regression'},
                        {'label': 'Decision Tree', 'value': 'Decision Tree'},
                        {'label': 'Random Forrest', 'value': 'Random Forrest'},
                        {'label': 'Gradient Boosting', 'value': 'Gradient Boosting'},
                        {'label': 'BootStraping', 'value': 'BootStraping'}
                    ],
                    value='None',
                    clearable=False
                ),
                # Table and Graph
                html.Div([
                    # Table Container
                    html.Div([
                        html.Div(id='table-container',
                             style={'overflowY': 'scroll', 'height': '400px', 'padding': '10px'}),
                        html.Div(id='table-info',
                             style={'padding': '10px'})],
                        style={'flex': '50%', 'padding': '10px'}),
                    # Graph Container
                    # Side for graph
                    html.Div([
                        dcc.Graph(id='figure-shower', figure=fig),  # Include fig here
                        html.Div(id='Metrics', style={'padding': '10px'})]
                    , style={'flex': '50%', 'padding': '10px'}),

                ], style={'display': 'flex'})
            ])
        ])
    ])
])


# parte que faz com que o Dashboard seja interativo para com o utilizador
# 1-Mostra o gráfico dos dados iniciais sem tratamento
@app.callback(
    dash.dependencies.Output('2019-data', 'figure'),
    [dash.dependencies.Input('Raw Variables', 'value'),
     ]
)
def Raw_data_graphs (variables_19):
    dff2 = data_X_19[variables_19]
    dff4 = pd.DataFrame(data={'Year': Date_19, 'Demand': dff2})
    fig = px.line(dff4, x='Year', y=[dff2])

    fig.update_layout(yaxis_title='')

    return fig

#2- Atualza o Forecast pedido pelo o utilizador e tipo de tratamento de dados para fazer a previsão

@app.callback(
    [Output('table-container', 'children'),
     Output('table-info', 'children'),
     Output('figure-shower', 'figure'),
     Output('Metrics','children')],
    [Input('feature-dropdown', 'value'),
     Input('table-shape-dropdown', 'value'),
     Input('Forecast-dropdown', 'value')]
)
def update_table(selected_features, table_shape, forecast_method):
    predictions_df = None

    if not selected_features:
        return [], [], {}, {}

    X_test_index = df_2019[selected_features].index
    X_test = df_2019[selected_features].values

    Y_train = df_data[df_data.columns[0]].dropna().values

    selected_df = df_data[selected_features].copy().dropna()

    if table_shape == 'IQR':
        first_column = df_data.columns[0]
        Q1 = df_data[first_column].quantile(0.25)
        Q3 = df_data[first_column].quantile(0.75)
        IQR = Q3 - Q1

        df_clean_IQR = df_data[((df_data[first_column] > (Q1 - 1.5 * IQR)) & (df_data[first_column] < (Q3 + 1.5 * IQR)))]
        selected_df = df_clean_IQR[selected_features]
        Y_train = df_data[(df_data[first_column] > (Q1 - 1.5 * IQR)) & (df_data[first_column] < (Q3 + 1.5 * IQR))][first_column]

    elif table_shape == 'Z_Score':
        first_column = df_data.columns[0]
        z = np.abs(stats.zscore(df_data[first_column]))
        selected_df['Z_Score'] = z
        selected_df = selected_df[(z < 3)].drop(columns=['Z_Score'])
        Y_train = df_data[(z < 3)][first_column].values

    elif table_shape == 'Interpole':
        first_column = df_data.columns[1]
        df_Interpolate = selected_df.dropna()
        df_Interpolate.index = pd.to_datetime(df_Interpolate.index)
        df_Interpolate[first_column] = df_Interpolate[first_column].mask(df_Interpolate[first_column] <= 100).interpolate(method='time')
        Y_train = df_Interpolate[first_column].values
        selected_df = df_Interpolate

    if forecast_method != 'None':
        predictions_df = perform_forecast(selected_df, Y_train, forecast_method, X_test)

        if predictions_df is not None:
            if not isinstance(predictions_df, pd.DataFrame):
                predictions_df = pd.DataFrame(predictions_df, columns=['Prediction'])

            if not predictions_df.empty:
                predictions_df.index = X_test_index

    table_header = html.Div([
        html.H4('Head of DataFrame'),
        html.Table([
            html.Tr([html.Th(col) for col in selected_df.columns]),
            *[html.Tr([html.Td(selected_df.iloc[i][col]) for col in selected_df.columns]) for i in range(min(5, len(selected_df)))]
        ])
    ])

    X_train = selected_df.values

    table_info = html.Div([
        html.H4('DataFrame Information'),
        html.P(f'Number of Columns: {selected_df.shape[1]}'),
        html.P(f'Number of Rows: {selected_df.shape[0]}'),
        html.Pre(selected_df.info()),
    ])

    if forecast_method != 'None':
        if predictions_df is not None:
            d = {'Date': real_data['Date'].values.ravel(),
                 'Ytest': predictions_df.values.ravel(),
                 'Real_Values_19': y2.values.ravel()}

            fig = px.line(d, x='Date', y=['Ytest', 'Real_Values_19'])
            fig.update_layout(yaxis_title='')

            MAE = metrics.mean_absolute_error(y2, predictions_df)
            MBE = np.mean(y2 - predictions_df)
            MSE = metrics.mean_squared_error(y2, predictions_df)
            RMSE = np.sqrt(metrics.mean_squared_error(y2, predictions_df))
            cvRMSE = RMSE_RF / np.mean(y2)
            NMBE = MBE_RF / np.mean(y2)

            d = {'Methods': forecast_method,
                 'MAE': MAE,
                 'MBE': MBE,
                 'MSE': MSE,
                 'RMSE': RMSE,
                 'cvMSE': cvRMSE,
                 'NMBE': NMBE}

            Metrics = html.Div([
                html.H4('Metrics'),
                html.Table([
                    html.Thead(html.Tr([html.Th(key) for key in d.keys()])),
                    html.Tbody([html.Tr([html.Td(d[key]) for key in d.keys()])])
                ])
            ])
            return table_header, table_info, fig, Metrics

    # Return default table and graph when forecast_method is 'None'
    d = {'Date': real_data['Date'].values.ravel(), 'Real_Values_19': y2.values.ravel()}
    default_fig = px.line(d, x='Date', y='Real_Values_19')
    default_fig.update_layout(yaxis_title='Y2')

    return table_header, table_info, default_fig, {}

if __name__ == '__main__':
    app.run(debug=True)
