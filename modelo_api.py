from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# Definir la estructura del cuerpo de la solicitud (Delta,High-alpha,High-beta,Low-alpha,Low-beta,Low_gamma,Mid-gamma,Theta)
class DatosDePrueba(BaseModel):
    delta: float
    highAlpha: float
    highBeta: float
    lowAlpha: float
    lowBeta: float
    lowGamma: float
    midGamma: float
    tetha: float

app = FastAPI()

# Cargar el modelo desde el archivo .pkl
def cargar_modelo(ruta_archivo):
    modelo = load(ruta_archivo)
    return modelo

# Hacer predicciones usando el modelo cargado
def hacer_prediccion(modelo, datos_de_prueba):
    prediccion = modelo.predict(datos_de_prueba)
    return prediccion

# Mapeo de clases a etiquetas
def obtener_etiqueta_clase(clase):
    etiquetas_clases = {
        0: "Usted no ha sido detectado con ninguna enfermedad de Alzheimer o Parkinson.",
        1: "Usted ha sido diagnosticado con Alzheimer",
        2: "Usted ha sido diagnosticado con Parkinson"
        # Añade más clases si es necesario
    }
    return etiquetas_clases.get(clase, "Clase Desconocida")

# Ruta al archivo .pkl que contiene el modelo
ruta_modelo_pkl = r'svm_model.pkl'  # Cambia esto al nombre del archivo que guardaste

# Cargar el modelo
modelo_cargado = cargar_modelo(ruta_modelo_pkl)

@app.post("/predecir/")
async def predecir(datos: DatosDePrueba):
    datos_lista = [[datos.delta, datos.highAlpha, datos.highBeta, datos.lowAlpha, datos.lowBeta, datos.lowGamma, datos.midGamma, datos.tetha]]
    predicciones = hacer_prediccion(modelo_cargado, datos_lista)
    etiquetas_predicciones = [obtener_etiqueta_clase(clase) for clase in predicciones]

    # Construir la respuesta con el formato deseado
    response = {
        "statusCode": 200,
        "success": True,
        "messages": ["OK"],
        "predicciones": etiquetas_predicciones
    }
    return response

# Iniciar el servidor cuando se ejecute el script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("modelo_api:app", host="127.0.0.1", port=8000, reload=True)
