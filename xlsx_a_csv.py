import pandas as pd

# Ruta del archivo Excel (.xlsx) y del archivo CSV de salida
xlsx_file_path = 'dataset_brain.xlsx'
csv_file_path = 'dataset_brain.csv'

# Leer el archivo Excel
data = pd.read_excel(xlsx_file_path)

# Escribir el archivo CSV
data.to_csv(csv_file_path, index=False)

print(f'Se ha convertido {xlsx_file_path} a {csv_file_path}')
