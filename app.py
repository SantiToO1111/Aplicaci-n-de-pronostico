from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
datos=pd.read_csv("ventas.csv", sep=';')
#función para ejecutar  el pronostico

app = Flask(__name__)
@app.route("/")
def home():
  return render_template("pronosticos.html")

@app.route('/pronosticos', methods=['GET', 'POST'])

def pronosticar():
    if request.method == 'POST':
        N = int(request.form['N'])
        p = request.form['p']

        producto_selecionado=datos[[p]].copy()
        producto_selecionado["pronostico"]=datos[p].rolling(window=N).mean().shift(1)
        producto_selecionado["Error"]=producto_selecionado['pronostico']-producto_selecionado[p]
        producto_selecionado['Error_abs']=producto_selecionado['Error'].abs()
        producto_selecionado['APE']=producto_selecionado['Error_abs']/producto_selecionado[p]
        producto_selecionado["APE'"]=producto_selecionado['Error_abs']/producto_selecionado['pronostico']
        producto_selecionado["error_cuadrado"]=producto_selecionado["Error"]*producto_selecionado["Error"]

        #medidas de errors
        MAPE=round(producto_selecionado["APE"].mean()*100, 2)
        MAPE_prima=round(producto_selecionado["APE'"].mean()*100, 2)
        MsE=round(producto_selecionado["error_cuadrado"].mean(), 2)
        MAE_prima=round(producto_selecionado["Error"].mean(), 2)
        RMSE=round(MsE**0.5, 2)

        # Gráfico
        plt.figure(figsize=(10,5))
        plt.plot(producto_selecionado.index, producto_selecionado[p], label="Ventas Históricas", marker='o')
        plt.plot(producto_selecionado.index, producto_selecionado["pronostico"], label="Pronóstico", marker='x')
        plt.xlabel("Período")
        plt.ylabel("Ventas")
        plt.title("Ventas Históricas vs Pronóstico")
        plt.legend()
        plt.grid(True)
        plt.savefig("static/grafico.png")
        plt.close()

        return render_template("pronosticos.html", 
                               MAPE=MAPE, 
                               MAPE_prima=MAPE_prima, 
                               MsE=MsE, MAE_prima=MAE_prima, 
                               RMSE=RMSE, 
                               pronostico=round(producto_selecionado["pronostico"].iloc[-1], 0))

    return render_template("pronosticos.html")


if __name__=="__main__":
    app.run(debug=True)
