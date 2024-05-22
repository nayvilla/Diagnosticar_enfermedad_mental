import pickle

# Cargar el modelo desde el archivo .pkl
def cargar_modelo(ruta_archivo):
    with open(ruta_archivo, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

# Hacer predicciones usando el modelo cargado
def hacer_prediccion(modelo, datos_de_prueba):
    prediccion = modelo.predict(datos_de_prueba)
    return prediccion

# Mapeo de clases a etiquetas
def obtener_etiqueta_clase(clase):
    etiquetas_clases = {
        0: "Clase 1",
        1: "Clase 2",
        2: "Clase 3"
        # Añade más clases si es necesario
    }
    return etiquetas_clases.get(clase, "Clase Desconocida")

# Ruta al archivo .pkl que contiene el modelo
ruta_modelo_pkl = r'svm_model.pkl'

# Cargar el modelo
modelo_cargado = cargar_modelo(ruta_modelo_pkl)

# Datos de prueba para hacer predicciones
datos_de_prueba = [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]  # Reemplaza esto con tus propios datos de prueba

# Hacer predicciones usando el modelo cargado
predicciones = hacer_prediccion(modelo_cargado, datos_de_prueba)

# Mapear las clases predichas a etiquetas
etiquetas_predicciones = [obtener_etiqueta_clase(clase) for clase in predicciones]

print("Predicciones:", etiquetas_predicciones)
