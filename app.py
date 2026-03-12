import pandas as pd
datos=pd.read_csv("venta_historicas.csv")
#función para ejecutar un pronostico

def pronosticar(historia,N):
  datos["pronostico"]=datos["ventas"].rolling(window=N).mean().shift(1)
  datos["Error"]=datos['pronostico']-datos['ventas']
  datos['Error_abs']=datos['Error'].abs()
  datos['APE']=datos['Error_abs']/datos['ventas']
  datos["APE'"]=datos['Error_abs']/datos['pronostico']
  datos["error_cuadrado"]=datos["Error"]*datos["Error"]

  #medidas de error
  MAPE=datos["APE"].mean()
  MAPE_prima=datos["APE'"].mean()
  MsE=datos["error_cuadrado"].mean()
  MAE_prima=datos["Error"].mean()
  RMSE=MsE**0.5



  return datos,MAPE,MAPE_prima,MsE,MAE_prima,RMSE
print(pronosticar(datos,3))