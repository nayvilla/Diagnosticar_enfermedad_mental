import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from joblib import dump

try:
    # Cargar datos
    data = pd.read_csv('dataset_brain.csv')

    # Rellenar datos faltantes con ceros
    data.fillna(0, inplace=True)

    # Convertir columnas numéricas a tipo float
    numeric_cols = ['Delta', 'High-alpha', 'High-beta', 'Low-alpha', 'Low-beta', 'Low_gamma', 'Mid-gamma', 'Theta']
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Eliminar filas con datos faltantes después de la conversión
    data.dropna(subset=numeric_cols, inplace=True)

    X = data.drop('DIAGNOSTICO', axis=1)
    y = data['DIAGNOSTICO']

    # Oversampling de las clases minoritarias
    class_counts = y.value_counts()
    max_class_count = class_counts.max()
    class_names = class_counts.index

    X_resampled = pd.DataFrame(columns=X.columns)
    y_resampled = pd.Series(dtype=y.dtype)

    for class_name in class_names:
        class_data = data[data['DIAGNOSTICO'] == class_name]
        resampled_data = resample(class_data, replace=True, n_samples=max_class_count, random_state=42)
        X_resampled = pd.concat([X_resampled, resampled_data.drop('DIAGNOSTICO', axis=1)])
        y_resampled = pd.concat([y_resampled, resampled_data['DIAGNOSTICO']])

    X = X_resampled
    y = y_resampled

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Diccionario de modelos con parámetros ajustados
    models = {
        'SVM': SVC(kernel='linear', C=1),
        'Random Forest': RandomForestClassifier(n_estimators=100),
    }

    # Comparar modelos con validación cruzada estratificada
    results = {}
    for name, model in models.items():
        print(f"Entrenando modelo {name}...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
        results[name] = np.mean(cv_scores)
        print(f'{name}: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')

    # Determinar el mejor modelo
    best_model = max(results, key=results.get)
    print(f"Mejor modelo: {best_model} con una precisión de {results[best_model]:.3f}")

    # Entrenar el mejor modelo y evaluarlo en el conjunto de prueba
    final_model = models[best_model]
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión en el conjunto de prueba: {test_accuracy:.3f}")

    # Guardar el modelo final
    dump(final_model, f'{best_model.lower().replace(" ", "_")}_model.pkl')
    print(f"Modelo {best_model} guardado correctamente.")

except Exception as e:
    print(f"Ocurrió un error: {e}")
