from flask import Flask, render_template, request
import pandas as pd

from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

datos = pd.read_csv("ventas.csv", sep=';', parse_dates=['fecha'])

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("pronosticos.html")

@app.route('/pronosticos', methods=['GET', 'POST'])
def pronosticar():
    if request.method == 'POST':
        p = request.form['p']
        metodo = request.form.get('metodo', 'promedio_movil')

        producto_selecionado = datos[['fecha', p]].copy()

        # Promedio móvil (se ejecuta siempre para comparación)
        N = request.form.get('N', type=int)
        if N is not None and N > 0:
            producto_selecionado['pronostico_pm'] = producto_selecionado[p].rolling(window=N).mean().shift(1)
            producto_selecionado['pronostico_pm'] = producto_selecionado['pronostico_pm'].round(0)
        else:
            producto_selecionado['pronostico_pm'] = pd.NA

        producto_selecionado['Error_pm'] = producto_selecionado['pronostico_pm'] - producto_selecionado[p]
        producto_selecionado['Error_abs_pm'] = producto_selecionado['Error_pm'].abs()
        producto_selecionado['APE_pm'] = producto_selecionado['Error_abs_pm'] / producto_selecionado[p]
        producto_selecionado["APE'_pm"] = producto_selecionado['Error_abs_pm'] / producto_selecionado['pronostico_pm']
        producto_selecionado['error_cuadrado_pm'] = producto_selecionado['Error_pm'] ** 2

        MAPE_pm = round(producto_selecionado['APE_pm'].mean(skipna=True) * 100, 2) if N else None
        MAPE_prima_pm = round(producto_selecionado["APE'_pm"].mean(skipna=True) * 100, 2) if N else None
        MsE_pm = round(producto_selecionado['error_cuadrado_pm'].mean(skipna=True), 2) if N else None
        MAE_pm = round(producto_selecionado['Error_pm'].abs().mean(skipna=True), 2) if N else None
        RMSE_pm = round(MsE_pm ** 0.5, 2) if MsE_pm is not None else None

        # Horizonte de pronóstico para suavización exponencial
        h = request.form.get('h', type=int)
        if h is None or h < 1:
            h = 10

        # Suavización exponencial simple
        modelo = ETSModel(producto_selecionado[p])
        ajuste = modelo.fit(maxiter=1000)
        producto_selecionado['pronostico_se'] = ajuste.fittedvalues.round(0)
        producto_selecionado['Error_se'] = producto_selecionado['pronostico_se'] - producto_selecionado[p]
        producto_selecionado['Error_abs_se'] = producto_selecionado['Error_se'].abs()
        producto_selecionado['APE_se'] = producto_selecionado['Error_abs_se'] / producto_selecionado[p]
        producto_selecionado["APE'_se"] = producto_selecionado['Error_abs_se'] / producto_selecionado['pronostico_se']
        producto_selecionado['error_cuadrado_se'] = producto_selecionado['Error_se'] ** 2

        MAPE_se = round(producto_selecionado['APE_se'].mean(skipna=True) * 100, 2)
        MAPE_prima_se = round(producto_selecionado["APE'_se"].mean(skipna=True) * 100, 2)
        MsE_se = round(producto_selecionado['error_cuadrado_se'].mean(skipna=True), 2)
        MAE_se = round(producto_selecionado['Error_se'].abs().mean(skipna=True), 2)
        RMSE_se = round(MsE_se ** 0.5, 2)

        # Pronóstico futuro, solo para suavización exponencial
        try:
            forecast_se = ajuste.forecast(h).round(0)
            pronostico_se_final = int(forecast_se.iloc[-1])
        except Exception:
            forecast_se = None
            pronostico_se_final = None

        # Metodo prophet
        try:
            # Usa la misma fecha y frecuencia de la serie original
            first_date = producto_selecionado['fecha'].iloc[0]
            freq = pd.infer_freq(producto_selecionado['fecha'])
            if freq is None:
                freq = 'MS'

            datos_prophet = pd.DataFrame({
                'ds': pd.date_range(start=first_date, periods=len(producto_selecionado), freq=freq),
                'y': producto_selecionado[p].values
            })
            modelo_prophet = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            modelo_prophet.fit(datos_prophet)
            future_prophet = modelo_prophet.make_future_dataframe(periods=h, freq=freq)
            forecast_prophet = modelo_prophet.predict(future_prophet)

            producto_selecionado['pronostico_prophet'] = forecast_prophet['yhat'].iloc[:len(producto_selecionado)].values
            producto_selecionado['pronostico_prophet'] = producto_selecionado['pronostico_prophet'].round(0)
            forecast_prophet_futuro = forecast_prophet['yhat'].iloc[len(producto_selecionado):]
            pronostico_prophet_final = round(forecast_prophet_futuro.iloc[-1], 0) if not forecast_prophet_futuro.empty else None
        except Exception as e:
            producto_selecionado['pronostico_prophet'] = pd.NA
            forecast_prophet_futuro = pd.Series()
            pronostico_prophet_final = None

        producto_selecionado['Error_prophet'] = producto_selecionado['pronostico_prophet'] - producto_selecionado[p]
        producto_selecionado['Error_abs_prophet'] = producto_selecionado['Error_prophet'].abs()
        producto_selecionado['APE_prophet'] = producto_selecionado['Error_abs_prophet'] / producto_selecionado[p]
        producto_selecionado["APE'_prophet"] = producto_selecionado['Error_abs_prophet'] / producto_selecionado['pronostico_prophet']
        producto_selecionado['error_cuadrado_prophet'] = producto_selecionado['Error_prophet'] ** 2

        MAPE_prophet = round(producto_selecionado['APE_prophet'].mean(skipna=True) * 100, 2)
        MAPE_prima_prophet = round(producto_selecionado["APE'_prophet"].mean(skipna=True) * 100, 2)
        MsE_prophet = round(producto_selecionado['error_cuadrado_prophet'].mean(skipna=True), 2)
        MAE_prophet = round(producto_selecionado['Error_prophet'].abs().mean(skipna=True), 2)
        RMSE_prophet = round(MsE_prophet ** 0.5, 2)

        # Gráfico según opción seleccionada
        if metodo == 'suav_exponencial':
            pronostico_col = 'pronostico_se'
            titulo = "Suavización Exponencial Simple"
            pronostico_valor = pronostico_se_final
        elif metodo == 'prophet':
            pronostico_col = 'pronostico_prophet'
            titulo = "Prophet"
            pronostico_valor = pronostico_prophet_final
        else:
            pronostico_col = 'pronostico_pm'
            titulo = "Promedio Móvil"
            pronostico_valor = int(round(producto_selecionado['pronostico_pm'].iloc[-1], 0)) if N and not pd.isna(producto_selecionado['pronostico_pm'].iloc[-1]) else None

        producto_selecionado['pronostico'] = producto_selecionado[pronostico_col]

        display_hist = producto_selecionado[['fecha', p, 'pronostico']].copy()
        display_hist = display_hist.rename(columns={p:'historico'})
        display_rows = display_hist.to_dict('records')

        future_rows = []
        if metodo == 'suav_exponencial' and forecast_se is not None:
            start_date = producto_selecionado['fecha'].iloc[-1] + pd.offsets.MonthBegin(1)
            future_dates = pd.date_range(start=start_date, periods=h, freq='MS')
            future_rows = [{'fecha': d, 'pronostico': int(round(v, 0))} for d, v in zip(future_dates, forecast_se)]

        elif metodo == 'prophet':
            prophet_future = forecast_prophet.iloc[len(producto_selecionado):]
            future_rows = [{'fecha': d, 'pronostico': int(round(v, 0))} for d, v in zip(prophet_future['ds'], prophet_future['yhat'])]

        # Agrega los rows futuros a la misma tabla de pronósticos
        for row in future_rows:
            display_rows.append({'fecha': row['fecha'], 'historico': None, 'pronostico': row['pronostico']})

        plt.figure(figsize=(10, 5))
        plt.plot(producto_selecionado['fecha'], producto_selecionado[p], label="Ventas Históricas", marker='o')
        plt.plot(producto_selecionado['fecha'], producto_selecionado['pronostico'], label="Pronóstico (fit)", marker='x')

        if metodo == 'suav_exponencial' and forecast_se is not None:
            plt.plot(future_dates, [r['pronostico'] for r in future_rows], label=f"Pronóstico futuro SE (h={h})", marker='o', color='red')
        if metodo == 'prophet' and future_rows:
            plt.plot([r['fecha'] for r in future_rows], [r['pronostico'] for r in future_rows], label=f"Pronóstico futuro Prophet (h={h})", marker='o', linestyle='--', color='green')

        plt.xlabel("Período")
        plt.ylabel("Ventas")
        plt.title(f"Ventas Históricas vs Pronóstico ({titulo})")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/grafico.png")
        plt.close()

        return render_template(
            "pronosticos.html",
            metodo=metodo,
            MAPE_pm=MAPE_pm, MAPE_prima_pm=MAPE_prima_pm, MsE_pm=MsE_pm, MAE_pm=MAE_pm, RMSE_pm=RMSE_pm,
            MAPE_se=MAPE_se, MAPE_prima_se=MAPE_prima_se, MsE_se=MsE_se, MAE_se=MAE_se, RMSE_se=RMSE_se,
            MAPE_prophet=MAPE_prophet, MAPE_prima_prophet=MAPE_prima_prophet, MsE_prophet=MsE_prophet, MAE_prophet=MAE_prophet, RMSE_prophet=RMSE_prophet,
            pronostico_pm=int(round(producto_selecionado['pronostico_pm'].iloc[-1], 0)) if N and not pd.isna(producto_selecionado['pronostico_pm'].iloc[-1]) else None,
            pronostico_se=pronostico_se_final,
            pronostico_prophet=pronostico_prophet_final,
            h=h,
            display_rows=display_rows,
            future_rows=future_rows
        )

    return render_template("pronosticos.html")

if __name__ == "__main__":
    app.run(debug=True)
