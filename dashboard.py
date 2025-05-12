import os
import json
import numpy as np
import dash
from dash import Dash, dcc, html, Output, Input
import plotly.graph_objs as go
import plotly.express as px
import plotly.subplots as sp
import yfinance as yf
import pandas as pd
from simulacion import MonteCarloVaR
from arch import arch_model
from sklearn.linear_model import LinearRegression

# Diccionario de nombres
nombres_tickers = {
    "1398.HK": "ICBC",
    "0939.HK": "China Construction Bank",
    "1288.HK": "Agricultural Bank of China",
    "3988.HK": "Bank of China",
    "3968.HK": "China Merchants Bank",
    "6030.HK": "CITIC Securities",
    "2611.HK": "Haitong Securities",
    "2318.HK": "Ping An Insurance",
    "2628.HK": "China Life Insurance",
    "1339.HK": "PICC"
}

# --- Parte 1: Ratios ---
class DataDownloader:
    def __init__(self, tickers):
        self.tickers = tickers
        self.balances = {}

    def descargar_balances(self):
        for ticker in self.tickers:
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            balance_sheet = balance_sheet.loc[:, balance_sheet.columns >= '2021-01-01']
            self.balances[ticker] = balance_sheet
        return self.balances

class RatioCalculator:
    def __init__(self, balances):
        self.balances = balances
        self.ratios = {}

    def calcular_ratios(self):
        for ticker, bs in self.balances.items():
            try:
                total_debt = bs.loc["Total Debt"]
                total_equity = bs.loc["Total Equity Gross Minority Interest"]
                total_assets = bs.loc["Total Assets"]

                df_ratios = pd.DataFrame({
                    'Debt_to_Equity': total_debt / total_equity,
                    'Debt_to_Assets': total_debt / total_assets,
                    'Equity_to_Assets': total_equity / total_assets
                })

                self.ratios[ticker] = df_ratios
            except KeyError as e:
                print(f"No se pudieron calcular ratios para {ticker}: {e}")
        df = pd.concat(self.ratios, axis=0)
        df.index.names = ['Ticker', 'Fecha']
        return df

class GraphGenerator:
    @staticmethod
    def generar_subplots(df_riesgo_capital):
        colores_ratios = {
            'Debt_to_Equity': '#7F3FD6',
            'Debt_to_Assets': '#E3000F',
            'Equity_to_Assets': '#0D3B78'
        }

        unique_tickers = df_riesgo_capital.index.get_level_values(0).unique()
        fig = sp.make_subplots(rows=5, cols=2, shared_xaxes=True, 
                               subplot_titles=[nombres_tickers[t] for t in unique_tickers])

        for i, ticker in enumerate(unique_tickers):
            row = i // 2 + 1
            col = i % 2 + 1
            subset = df_riesgo_capital.loc[ticker].sort_index()

            for ratio, color in colores_ratios.items():
                fig.add_trace(go.Scatter(
                    x=subset.index, y=subset[ratio],
                    mode='lines+markers', name=ratio.replace('_', '/'),
                    legendgroup=f"{ticker}", showlegend=(i == 0),
                    line=dict(color=color)
                ), row=row, col=col)

        fig.update_layout(height=1800, width=1000, showlegend=True)
        return fig


class GARCHModel:
    def __init__(self, tickers, start_date, end_date, p=1, q=1):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.p = p
        self.q = q
        self.data = {}
        self.models = {}
        self.model_fits = {}

    def load_data(self):
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            self.data[ticker] = df.dropna()

    def train_models(self):
        for ticker in self.tickers:
            returns = self.data[ticker]['Returns']
            model = arch_model(returns, vol='GARCH', p=self.p, q=self.q)
            self.model_fits[ticker] = model.fit(disp="off")

    def plot_results(self):
        fig = go.Figure()
        for ticker in self.tickers:
            vol = self.model_fits[ticker].conditional_volatility
            fig.add_trace(go.Scatter(x=vol.index, y=vol, mode='lines', name=f'Volatilidad {ticker}'))

        fig.update_layout(title='Volatilidad Condicional estimada por el modelo GARCH',
                          xaxis_title='Fecha', yaxis_title='Volatilidad')
        return fig

    def get_forecasts(self, steps=10):
        forecasts = {}
        for ticker in self.tickers:
            forecast = self.model_fits[ticker].forecast(horizon=steps)
            forecasts[ticker] = forecast.variance.values[-1, :]
        return forecasts
    
garch_model = GARCHModel(
    tickers=list(nombres_tickers.keys()), 
    start_date="2021-01-01", 
    end_date="2024-12-31"
)

garch_model.load_data()
garch_model.train_models()
# --- Parte 2: VaR ---
SIMULATIONS_FOLDER = "simulaciones"

def load_simulations():
    data = {}
    for filename in os.listdir(SIMULATIONS_FOLDER):
        if filename.endswith(".json"):
            name = filename.replace("_var_results.json", "").replace("_", " ").title()
            with open(os.path.join(SIMULATIONS_FOLDER, filename), 'r') as f:
                vars = json.load(f)
                data[name] = np.array(vars)
    return data

simulation_data = load_simulations()

# --- Procesamiento de datos ---
tickers = list(nombres_tickers.keys())
descargador = DataDownloader(tickers)
balances = descargador.descargar_balances()
calculador = RatioCalculator(balances)
df_riesgo_capital = calculador.calcular_ratios()
df_reset = df_riesgo_capital.reset_index()
df_reset['Fecha'] = pd.to_datetime(df_reset['Fecha']).dt.year
for col in ['Debt_to_Equity', 'Debt_to_Assets', 'Equity_to_Assets']:
    df_reset[col] = pd.to_numeric(df_reset[col], errors='coerce')
df_reset = df_reset.dropna()
df_reset['Empresa'] = df_reset['Ticker'].map(nombres_tickers)

# --- Métricas individuales de precios ---
start_date = "2024-02-14"
end_date = "2025-02-14"
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

precios_cierre = pd.DataFrame({ticker: data[ticker]["Close"] for ticker in tickers})
precios_volumen = pd.DataFrame({ticker: data[ticker]["Volume"] for ticker in tickers})

metricas = []
for ticker in tickers:
    if ticker in data:
        print(f"Calculando métricas para {nombres_tickers[ticker]}...")
        try:
            precio_prom = data[ticker]["Close"].mean()
            volumen_prom = data[ticker]["Volume"].mean()
            volatilidad = data[ticker]["Close"].std()

            print(f"Precio promedio: {precio_prom}, Volumen promedio: {volumen_prom}, Volatilidad: {volatilidad}")

            metricas.append({
                "Empresa": nombres_tickers[ticker],
                "Precio promedio": round(precio_prom, 2),
                "Volumen promedio": round(volumen_prom, 2),
                "Volatilidad": round(volatilidad, 2)
            })
        except Exception as e:
            print(f"Error calculando métricas para {nombres_tickers[ticker]}: {e}")
    else:
        print(f"No hay datos para {nombres_tickers[ticker]}")

# Descargar datos
start_date = "2024-01-01"
end_date = "2024-03-30"
df = yf.download(list(nombres_tickers.keys()), start=start_date, end=end_date, auto_adjust=False)

# Extraer métricas
closing_prices = df['Close']
volume = df['Volume']
mean_prices = closing_prices.mean()
mean_volume = volume.mean()
volatility = closing_prices.pct_change().std() * np.sqrt(len(closing_prices))

# Predicción de cierre a 30 días
X = np.arange(len(closing_prices)).reshape(-1, 1)
future_days = np.arange(len(closing_prices), len(closing_prices) + 30).reshape(-1, 1)
predictions = {}
for ticker in nombres_tickers.keys():
    y = closing_prices[ticker].dropna()
    model = LinearRegression()
    model.fit(X[:len(y)], y)
    predictions[ticker] = model.predict(future_days)

# --- Dash App ---
app = Dash("Proyecto Final")

app.layout = html.Div([
    html.H1("Dashboard Financiero - Ratios de Estructura de Capital", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Selecciona un ratio:"),
        dcc.Dropdown(
            id='dropdown-ratio',
            options=[{'label': col.replace("_", " "), 'value': col} for col in ['Debt_to_Equity', 'Debt_to_Assets', 'Equity_to_Assets']],
            value='Debt_to_Equity',
            style={'width': '50%'}
        )
    ], style={'padding': '20px'}),

    dcc.Graph(id='grafico-barras'),

    html.H2("Volatilidad estimada por modelo GARCH", style={'textAlign': 'center', 'marginTop': '40px'}),
    dcc.Graph(id='grafico-garch'),  

    html.H2("Evolución de ratios de estructura de capital (4 años)", style={'textAlign': 'center'}),
    dcc.Graph(id='grafico-subplots', figure=GraphGenerator.generar_subplots(df_riesgo_capital)),

    html.Hr(),

    html.H1("Simulación Monte Carlo y VaR - Sector Financiero Chino"),

    html.Label("Selecciona una institución financiera (simulada):"),
    dcc.Dropdown(
        id='bank-selector',
        options=[{'label': name, 'value': name} for name in simulation_data.keys()],
        value=list(simulation_data.keys())[0],
        style={'width': '60%'}
    ),

    html.Div(id='stats-output', style={'fontSize': 18, 'marginBottom': 20}),
    dcc.Graph(id='var-histogram'),
    html.Div(id='metricas-individuales', style={'fontSize': 18, 'marginTop': 40, 'marginBottom': 40}),
    dcc.Graph(
        id='price-chart',
        figure={
            'data': [go.Scatter(x=closing_prices.index, y=closing_prices[ticker],
                                mode='lines', name=ticker) for ticker in nombres_tickers.keys()],
            'layout': go.Layout(title='Precio de Cierre', xaxis_title='Fecha', yaxis_title='Precio')
        }
    ),

    dcc.Graph(
        id='volume-chart',
        figure={
            'data': [go.Bar(x=volume.index, y=volume[ticker], name=ticker) for ticker in nombres_tickers.keys()],
            'layout': go.Layout(title='Número de Acciones Negociadas', xaxis_title='Fecha', yaxis_title='Volumen')
        }
    ),

    dcc.Graph(
        id='mean-price-chart',
        figure={
            'data': [go.Bar(x=list(nombres_tickers.keys()), y=mean_prices, name='Precio Promedio')],
            'layout': go.Layout(title='Precio Promedio', xaxis_title='Acción', yaxis_title='Precio')
        }
    ),

    dcc.Graph(
        id='mean-volume-chart',
        figure={
            'data': [go.Bar(x=list(nombres_tickers.keys()), y=mean_volume, name='Número de Acciones Promedio')],
            'layout': go.Layout(title='Número de Acciones Promedio', xaxis_title='Acción', yaxis_title='Volumen')
        }
    ),

    dcc.Graph(
        id='volatility-chart',
        figure={
            'data': [go.Bar(x=list(nombres_tickers.keys()), y=volatility, name='Volatilidad Histórica')],
            'layout': go.Layout(title='Volatilidad Histórica', xaxis_title='Acción', yaxis_title='Volatilidad')
        }
    )
])


# Callbacks
# Callback para mostrar el gráfico GARCH
@app.callback(
    Output('grafico-garch', 'figure'),
    Input('dropdown-ratio', 'value')  # Puedes usar cualquier input que se relacione
)
def actualizar_grafico_garch(_):
    return garch_model.plot_results()
@app.callback(
    Output('grafico-barras', 'figure'),
    Input('dropdown-ratio', 'value')
)
def actualizar_grafico(ratio):
    fig = px.bar(
        df_reset,
        x='Fecha',
        y=ratio,
        color='Empresa',
        barmode='group',
        title=f'{ratio.replace("_", " ")} por Año y Empresa'
    )
    fig.update_layout(xaxis_title='Año', yaxis_title=ratio.replace("_", " "))
    return fig

@app.callback(
    [Output('var-histogram', 'figure'),
     Output('stats-output', 'children'),
     Output('metricas-individuales', 'children')],
    Input('bank-selector', 'value')
)
def update_var_graph(bank_name):
    # Obtener las simulaciones de VaR para la empresa seleccionada
    vars = simulation_data[bank_name]
    
    # Crear el gráfico de histograma de VaR
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vars, nbinsx=50, marker_color='indianred'))
    fig.update_layout(
        title=f'Histograma de VaR Simulado - {bank_name}',
        xaxis_title='Valor en Riesgo (VaR)',
        yaxis_title='Frecuencia',
        bargap=0.1
    )

    # Calcular estadísticas del VaR
    avg_var = np.mean(vars)
    perc_5 = np.percentile(vars, 5)
    stats_text = f"VaR Promedio: {avg_var:.4f} | VaR Percentil 5%: {perc_5:.4f}"

    # Limpiar y comparar los nombres de las empresas
    bank_name_clean = bank_name.strip().lower()

    # Buscar las métricas de la empresa seleccionada
    metricas_empresa = None
    for m in metricas:
        empresa_clean = m['Empresa'].strip().lower()
        print(f"Comparando: {bank_name_clean} con {empresa_clean}")  # Depuración
        if bank_name_clean == empresa_clean:
            metricas_empresa = m
            break

    # Mostrar las métricas si se encontraron
    if metricas_empresa:
        lista = html.Ul([
            html.Li(f"Precio promedio: ${metricas_empresa['Precio promedio']}"),
            html.Li(f"Volumen promedio: {metricas_empresa['Volumen promedio']:,}"),
            html.Li(f"Volatilidad: {metricas_empresa['Volatilidad']}")
        ])
    else:
        lista = html.Div("No hay métricas disponibles para esta empresa.")

    
    return fig, stats_text, lista


if __name__ == '__main__':
    app.run(debug=True)