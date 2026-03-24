# SAD-Clasificacion-Automatizada

Este repositorio contiene la implementación de un entorno de experimentación para la asignatura de **Sistemas de Ayuda a la Decisión (SAD)** de la **Universidad del País Vasco (UPV/EHU)**. El objetivo principal es la transición de prototipos básicos de Dataiku a un entorno robusto en Python capaz de realizar barridos de hiperparámetros y evaluaciones automáticas.

## Miembros
* **Lou Gómez Foucher**
* **Aimar Larriba**

---

## Estructura del Proyecto
El proyecto se organiza en torno a scripts funcionales y archivos de persistencia siguiendo la "receta" oficial:

* **`train.py`**: Script encargado de cargar los datos, realizar el preproceso dinámico, ejecutar el barrido de parámetros y seleccionar el modelo ganador.
* **`test.py`**: Programa diseñado para cargar el modelo salvado y clasificar nuevas instancias.
* **`configuration.json`**: Fichero centralizado donde se definen las estrategias de preproceso (nulos, escalado, balanceo) y los rangos de hiperparámetros (Estructura exlicada más abajo).
* **`bestmodel.sav`**: Archivo binario que contiene el mejor modelo entrenado mediante `pickle`.
* **`preprocessing_objects.sav`**: Diccionario persistido con los objetos necesarios para que el test sea consistente con el entrenamiento (Scaler, Imputer, etc.).
* **`resultados_entrenamiento.csv`**: Informe generado automáticamente con las métricas Accuracy, Precision, Recall y F-score de todas las combinaciones probadas.

---

## Estructura de `configuration.json`
El archivo de configuración, el cual se muestra a continuación, actúa como el motor del experimento, permitiendo modificar el comportamiento de los scripts sin necesidad de editar el código fuente. 
```json
{
  "algorithm": "todos",
  "preprocessing": {
    "target_variable": "Target",
    "drop_features": [],
    "missing_values": "impute",
    "impute_strategy": "mean",
    "scaling": "standard",
    "sampling": "none"
  },
  "hyperparameters": {
    "knn": {
      "k_min": 1,
      "k_max": 5,
      "p_min": 1,
      "p_max": 2,
      "weights": ["uniform", "distance"]
    },
    "trees": {
      "max_depth": [3, 6, 9],
      "min_samples_leaf": [1, 2]
    },
    "naive_bayes": {
      "n_bins": 5
    }
  }
}
```
Este divide en tres bloques principales:

#### 1. Control de Ejecución
* **`algorithm`**: Determina el alcance del entrenamiento.
    * **Valores**: `"knn"`, `"tree"`, `"nb"` o `"todos"`.
    * **Función**: Permite aislar un experimento o ejecutar la comparativa completa entre algoritmos para seleccionar el mejor modelo global.

#### 2. Preprocesado (`preprocessing`)
Configura las transformaciones que aseguran la calidad de los datos antes del entrenamiento:

* **`target_variable`**: 
    * **Valores**: String (ej. `"Especie"` o `"Target"`).
    * **Función**: Debe coincidir exactamente con el nombre de la columna objetivo en el archivo `.csv`.
* **`drop_features`**: 
    * **Valores**: Lista de String `[]`. 
    * **Función**: Permite eliminar columnas irrelevantes o identificadores únicos para evitar el sobreajuste.
* **`missing_values`**: 
    * **Valores**: `"impute"` o `"none"`. 
    * **Función**: Activa o desactiva la gestión de datos faltantes en el dataset.
* **`impute_strategy`**: 
    * **Valores**: `"mean"`, `"median"` o `"most_frequent"`. 
    * **Función**: Define el criterio estadístico para rellenar los valores nulos.
* **`scaling`**: 
    * **Valores**: `"standard"` o `"none"`. 
    * **Función**: Activa o desactiva el escalado $Z$-score, fundamental para algoritmos basados en distancia como KNN.
* **`sampling`**: 
    * **Valores**: `"undersampling"`, `"smote"` o `"none"`. 
    * **Función**: Balancea las clases en el conjunto de entrenamiento para evitar sesgos hacia la clase mayoritaria.

#### 3. Hiperparámetros (`hyperparameters`)
Define los rangos para el barrido automático (Grid Search) y la optimización de los modelos:

* **`knn`**:
    * **`k_min` / `k_max`**: Valores enteros (ej. 1 y 5). Define el rango de vecinos $k$ para el barrido.
    * **`p_min` / `p_max`**: Valores enteros (1 o 2). Define la métrica de distancia de Minkowski ($p=1$: Manhattan, $p=2$: Euclídea).
    * **`weights`**: Lista `["uniform", "distance"]`. Determina la influencia de los vecinos según su cercanía.
* **`trees`**:
    * **`max_depth`**: Lista de enteros (ej. `[3, 6, 9]`). Controla la profundidad máxima del árbol para evitar el *overfitting*.
    * **`min_samples_leaf`**: Lista de enteros (ej. `[1, 2]`). Define el número mínimo de muestras requerido en un nodo terminal.
* **`naive_bayes`**:
    * **`n_bins`**: Valor entero (ej. 5). Determina el número de intervalos para la discretización de variables continuas necesaria para `CategoricalNB`.
---

## Requisitos
Para garantizar el correcto funcionamiento del pipeline, es necesario contar con el siguiente entorno:

* **Entorno**: Se recomienda el uso de **Anaconda** con Python 3.7 o superior.
* **Librerías principales**:
    * `scikit-learn`: Para algoritmos de clasificación y métricas.
    * `pandas`: Para la manipulación de los juegos de datos.
    * `imbalanced-learn (imblearn)`: Para técnicas de balanceo.
    * `mixed-naive-bayes`: Para la implementación de Naive Bayes mixto.
    * `pickle` / `pickle5`: Para la serialización de modelos en disco.

---

## Modo de Empleo

### 1. Entrenamiento y Barrido
El script de entrenamiento requiere dos argumentos por línea de comandos: el archivo de datos y el fichero de configuración.
```bash
python train.py TrainDev.csv configuration.json
```
Este proceso evalúa todas las combinaciones de KNN, Árboles de Decisión y Naive Bayes definidas, guardando el mejor resultado basado en la figura de mérito F-score.

### 2. Clasificación de Instancias
Para predecir la clase de nuevas muestras, se utiliza el script de test con el modelo previamente guardado:
```bash
python test.py Test.csv configuration.json
```
**Nota:** El script de test aplica automáticamente el preproceso (escalado, imputación) utilizando los parámetros aprendidos durante el entrenamiento, pero nunca aplica balanceo a los datos de test.
