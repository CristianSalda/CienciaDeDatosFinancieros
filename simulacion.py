import yfinance as yf
import numpy as np
import json
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

class StockpriceSimulator:
    def __init__(self, initial_price, mu, sigma, dt, steps):
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.steps = steps
        self.prices = None

    def simulate(self):
        prices = [self.initial_price]
        for _ in range(self.steps):
            dWt = np.random.normal(0, np.sqrt(self.dt))
            dS = self.mu * prices[-1] * self.dt + self.sigma * prices[-1] * dWt
            new_price = prices[-1] + dS
            prices.append(new_price)
        self.prices = np.array(prices).flatten()

        if np.any(np.isnan(self.prices)) or len(self.prices) < 2:
            print("Error: precios simulados inválidos", self.prices)

    def plot_prices(self):
        plt.plot(self.prices)
        plt.title('Simulación de Precios de Acción')
        plt.xlabel('Tiempo')
        plt.ylabel('Precio')
        plt.show()

class VaRCalculator:
    def __init__(self, prices):
        self.prices = prices

    def calculate_var(self, confidence_level):
        if self.prices is None or len(self.prices) < 2:
            raise ValueError("No hay suficientes datos de precios para calcular el VaR.")
        returns = np.diff(self.prices) / self.prices[:-1]
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index]
        return var

class MonteCarloVaR:
    def __init__(self, initial_price, mu, sigma, dt, steps, simulations, confidence_level):
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.steps = steps
        self.simulations = simulations
        self.confidence_level = confidence_level

    def run_simulation(self):
        vars = []
        for _ in range(self.simulations):
            simulator = StockpriceSimulator(self.initial_price, self.mu, self.sigma, self.dt, self.steps)
            simulator.simulate()
            if simulator.prices is None or len(simulator.prices) < 2:
                continue
            var_calculator = VaRCalculator(simulator.prices)
            try:
                var = var_calculator.calculate_var(self.confidence_level)
                vars.append(var)
            except Exception as e:
                print("Error en cálculo de VaR:", e)

        return np.array(vars)

    def save_results(self, vars, filename):
        os.makedirs("simulaciones", exist_ok=True)
        with open(f"simulaciones/{filename}", 'w') as f:
            json.dump(vars.tolist(), f)

    @staticmethod
    def get_stock_data(ticker, start_date, end_date):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data


# Punto 8
class LiquidezDashboard:
    def __init__(self, nombres_tickers, activos, pasivos):
        # Inicializamos los datos
        self.nombres_tickers = nombres_tickers
        self.activos = activos
        self.pasivos = pasivos
        
        # Crear DataFrame
        self.df_liquidez = self.crear_dataframe()

    def crear_dataframe(self):
        """Crear el DataFrame con los datos de activos y pasivos"""
        data = []
        for i, empresa in enumerate(self.nombres_tickers):
            for año in [2021, 2022, 2023, 2024]:
                data.append({
                    "Empresa": empresa,
                    "Año": año,
                    "Activos Corrientes": self.activos[año][i],
                    "Pasivos Corrientes": self.pasivos[año][i]
                })
        
        df_liquidez = pd.DataFrame(data)
        df_liquidez["Razón Corriente"] = df_liquidez["Activos Corrientes"] / df_liquidez["Pasivos Corrientes"]
        return df_liquidez

    def generar_grafico(self):
        """Generar el gráfico de la razón corriente por empresa y por año"""
        años = sorted(self.df_liquidez['Año'].unique())
        empresas = self.df_liquidez['Empresa'].unique()

        colores = px.colors.qualitative.Set3
        color_empresa = {empresa: colores[i % len(colores)] for i, empresa in enumerate(empresas)}

        # Crear la figura con subgráficos por año
        fig = make_subplots(
            rows=len(años), cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Año {a}" for a in años]
        )

        # Agregar barras para cada empresa por año
        for i, año in enumerate(años, start=1):
            df_año = self.df_liquidez[self.df_liquidez["Año"] == año]
            for empresa in empresas:
                valor = df_año[df_año["Empresa"] == empresa]["Razón Corriente"].values
                if len(valor) > 0:
                    fig.add_trace(
                        go.Bar(
                            x=[empresa],  
                            y=[valor[0]],
                            name=self.nombres_tickers[empresa], 
                            marker=dict(color=color_empresa[empresa]),
                            showlegend=(i == 1)  # Mostrar leyenda solo en el primer gráfico
                        ),
                        row=i, col=1
                    )

        fig.update_layout(
            height=300 * len(años),
            title="Razón Corriente por Empresa - Gráficos Separados por Año",
            barmode="group",
            xaxis_tickangle=-45,
            bargap=0.3
        )

        return fig

# Mapa de Henon

class HestonModel:
    def __init__(self, S0, K, T, r, sigma, kappa, theta, rho, v0, N=10000):
        self.S0 = S0      # Precio inicial del activo.
        self.K = K        # Precio de ejercicio de la opción.
        self.T = T        # Tiempo de vencimiento (años).
        self.r = r        # Tasa de interés.
        self.sigma = sigma  # Volatilidad del activo.
        self.kappa = kappa  # Tasa de reversión a la media.
        self.theta = theta  # Varianza a largo plazo.
        self.rho = rho      # Correlación entre los procesos de cierre y volatilidad.
        self.v0 = v0        # Varianza inicial.
        self.N = N        # Número de simulaciones de Monte Carlo.

    def simulate(self):
        dt = 1/252  # Paso temporal (usando 252 días hábiles al año)
        M = int(self.T / dt)
        S = np.zeros((self.N, M))
        V = np.zeros((self.N, M))
        S[:, 0] = self.S0
        V[:, 0] = self.v0

        for t in range(1, M):
            Z1 = np.random.normal(size=self.N)
            Z2 = np.random.normal(size=self.N)
            dW1 = np.sqrt(dt) * Z1
            # Mantener la correlación entre los procesos
            dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.sqrt(dt) * Z2
            V[:, t] = np.maximum(V[:, t-1] + self.kappa * (self.theta - V[:, t-1]) * dt + self.sigma * np.sqrt(V[:, t-1]) * dW2, 0)
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * V[:, t-1]) * dt + np.sqrt(V[:, t-1]) * dW1)
        return S, V

    def calculate_option_price(self):
        S, _ = self.simulate()
        option_payoff = np.maximum(S[:, -1] - self.K, 0)
        option_price = np.exp(-self.r * self.T) * np.mean(option_payoff)
        return option_price

    def plot_simulation(self, ax=None, num_paths=10):
        S, _ = self.simulate()
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_paths):
            ax.plot(S[i], lw=1)
        ax.set_title('Trayectorias simuladas del Precio del Activo')
        ax.set_xlabel('Pasos de tiempo')
        ax.set_ylabel('Precio del activo')
        if ax is None:
            plt.show()

class HestonDashboard(HestonModel):
    def generate_simulation_figure(self, num_paths=10):
        S, _ = self.simulate()
        fig = go.Figure()
        for i in range(num_paths):
            fig.add_trace(go.Scatter(y=S[i], mode='lines', name=f'Sim {i+1}'))
        fig.update_layout(
            title='Trayectorias simuladas del Precio del Activo (Modelo de Heston)',
            xaxis_title='Pasos de tiempo',
            yaxis_title='Precio del activo',
            template='plotly_white'
        )
        return fig


acciones = [
    {"nombre": "CITIC Securities", "S0": 20, "K": 20, "T": 1, "r": 0.05, "sigma": 0.35, "kappa": 1.4, "theta": 0.05, "rho": -0.5, "v0": 0.05},
    {"nombre": "Haitong Securities", "S0": 15, "K": 15, "T": 1, "r": 0.05, "sigma": 0.30, "kappa": 1.5, "theta": 0.045, "rho": -0.45, "v0": 0.045},
    {"nombre": "Ping An Insurance", "S0": 50,  "K": 50, "T": 1, "r": 0.05, "sigma": 0.25, "kappa": 1.7, "theta": 0.04, "rho": -0.5, "v0": 0.04},
    {"nombre": "China Life Insurance", "S0": 60, "K": 60, "T": 1, "r": 0.05, "sigma": 0.20, "kappa": 1.6, "theta": 0.035, "rho": -0.55, "v0": 0.035},
    {"nombre": "People’s Insurance Company of China", "S0": 40, "K": 40, "T": 1, "r": 0.05, "sigma": 0.22, "kappa": 1.8, "theta": 0.04, "rho": -0.5, "v0": 0.04}
]


def generar_figuras_acciones(acciones):
    figuras = []
    precios = []
    for params in acciones:
        nombre = params["nombre"]
        parametros_modelo = {k: v for k, v in params.items() if k != "nombre"}
        modelo = HestonDashboard(**parametros_modelo)
        precio = modelo.calculate_option_price()
        fig = modelo.generate_simulation_figure(num_paths=10)
        fig.update_layout(title=f'Trayectorias simuladas - {nombre}')
        figuras.append((nombre, fig))
        precios.append({"Acción": nombre, "Precio Opción": round(precio, 4)})
    return figuras, precios

#comportamiendo del sector 

# ---------- EJECUCIÓN PARA TODOS LOS TICKERS ---------- #

tickers = {
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

if __name__ == '__main__':
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)

    for ticker, name in tickers.items():
        print(f"\nProcesando {name} ({ticker})...")
        try:
            stock_data = MonteCarloVaR.get_stock_data(ticker, start_date, end_date)

            if stock_data.empty:
                print(f"No se pudieron obtener datos para {ticker}.")
                continue

            log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1)).dropna()
            mu = log_returns.mean()
            sigma = log_returns.std()
            initial_price = stock_data['Close'].iloc[-1]

            mc_var = MonteCarloVaR(
                initial_price=initial_price,
                mu=mu,
                sigma=sigma,
                dt=1/252,
                steps=10,
                simulations=1000,
                confidence_level=0.95
            )

            vars = mc_var.run_simulation()
            filename = f"{name.replace(' ', '_').lower()}_var_results.json"
            mc_var.save_results(vars, filename)
            print(f"Simulación completada y guardada: simulaciones/{filename}")

        except Exception as e:
            print(f"Error procesando {ticker} ({name}): {e}")

    
