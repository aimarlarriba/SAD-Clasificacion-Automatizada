
# SAD-Clasificacion-Automatizada
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Este repositorio contiene la implementación de un entorno de experimentación para la asignatura de **Sistemas de Ayuda a la Decisión (SAD)** de la **Universidad del País Vasco (UPV/EHU)**. El objetivo principal es la transición de prototipos básicos de Dataiku a un entorno robusto en Python capaz de realizar barridos de hiperparámetros y evaluaciones automáticas.

## Desarrollado por:
* **Lou Gómez**
* **Aimar Larriba**

---

## Estructura del Proyecto
El proyecto se organiza en torno a scripts funcionales y archivos de persistencia siguiendo la "receta" oficial:

* **`train.py`**: Script que realiza la carga de datos, preproceso dinámico, partición estratificada, barrido de parámetros (Grid Search) y selección del mejor modelo.
* **`test.py`**: Programa para cargar el modelo ganador y clasificar nuevas instancias, manteniendo la consistencia del preproceso.
* **`configuration.json`**: Fichero centralizado de configuración (estrategias de preproceso y rangos de hiperparámetros).
* **`proyectos/{project_name}/`**: 
  * **`datos/`**: Copias de seguridad de los datasets utilizados y tests automáticos.
  * **`best_model/`**: Contiene `bestmodel.sav`, `preprocessing_objects.sav` y el informe de `ultimos_resultados.csv`.
  * **`archivo_versiones/`**: Histórico de modelos previos archivados al encontrar una mejora en el F-score.
  * **`predicciones_generadas/`**: CSVs resultantes de las ejecuciones de `test.py`.
  
```
      .
      ├── train.py                # Script de entrenamiento y optimización
      ├── test.py                 # Script de inferencia y evaluación
      ├── configuration.json      # Configuración de experimentos
      └── proyectos/
          └── {project_name}/     # Carpeta creada automáticamente
              ├── datos/          # Copias de seguridad y tests estratificados
              │   ├── entrenamiento_dataset.csv
              │   └── test_automatico_Iris.csv
              ├── best_model/     # El modelo con mejor F1-score hasta la fecha
              │   ├── bestmodel.sav
              │   ├── preprocessing_objects.sav
              │   ├── ultimos_resultados.csv
              │   └── predicciones_generadas/
              │       └── pred_KNN_F1_0.98_dataset.csv
              └── archivo_versiones/ # Historial de modelos superados
                  └── v_F1_0.9200_2026-03-20_10-30/
                      ├── bestmodel.sav
                      └── preprocessing_objects.sav
```

---

## Estructura de `configuration.json`
El archivo de configuración, el cual se muestra a continuación, actúa como el motor del experimento, permitiendo modificar el comportamiento de los scripts sin necesidad de editar el código fuente. 
```json
{
  "project_name": "Project",
  "algorithm": "knn",
  "average_strategy": "macro",
  "preprocessing": {
    "test_split": 0,
    "target_variable": "Target",
    "drop_features": [],
    "missing_values": "impute",
    "impute_strategy": "median",
    "scaling": "standard",
    "sampling": "undersampling"
  },
  "hyperparameters": {
    "knn": {
      "k_min": 1,
      "k_max": 9,
      "p_min": 1,
      "p_max": 2,
      "weights": ["uniform", "distance"]
    },
    "trees": {
      "max_depth": [5, 10, 15],
      "min_samples_leaf": [2, 5]
    },
    "random_forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, null]
    },
    "naive_bayes": {
      "n_bins": [5, 10],
      "alphas": [0.5, 1.0],
      "min_categories": null
    }
  }
}
```
Este se divide en tres bloques principales:

#### 1. Control de Ejecución
* **`project_name`**: Determina el nombre del proyecto.
* **`algorithm`**: Permite aislar un experimento o ejecutar la comparativa completa entre algoritmos para seleccionar el mejor modelo global.
    * **Valores**: `"knn"`, `"tree"`, `"rf"`, `"nb"` o `"todos"`.
* **`average_strategy`**: Determina el tipo de F-Score a usar.
    * **Valores**: `"micro"`, `"macro"`, `"weighted"`, `"binary"` o `"auto"`.

#### 2. Preprocesado (`preprocessing`)
Configura las transformaciones que aseguran la calidad de los datos antes del entrenamiento:

* **`test_split`**: Extrae un porcentaje de la muestra inicial para generar un set de evaluación.
    * **Valores**: Float entre 0 y 1.
* **`target_variable`**: Nombre del atributo a predecir. Debe coincidir exactamente con el nombre de la columna objetivo en el archivo `.csv`.
    * **Valores**: String.
* **`drop_features`**: Nombres de las columnas irrelevantes o identificadores únicos a eliminar para evitar el sobreajuste. Deben coincidir exactamente con el nombre de la columna en el archivo `.csv`.
    * **Valores**: Lista de String `[]`.
* **`missing_values`**: Activa o desactiva la gestión de datos faltantes en el dataset
    * **Valores**: `"impute"` o `"none"`.
* **`impute_strategy`**: Define el criterio estadístico para rellenar los valores nulos.
    * **Valores**: `"mean"`, `"median"` o `"most_frequent"`.
* **`scaling`**: Activa o desactiva el escalado $Z$-score, fundamental para algoritmos basados en distancia como KNN.
    * **Valores**: `"standard"` o `"none"`.
* **`sampling`**: Define el método de balanceo de clases en el conjunto de entrenamiento para evitar sesgos hacia la clase mayoritaria.
    * **Valores**: `"undersampling"`, `"smote"`, `"adasyn"` o `"none"`.

#### 3. Hiperparámetros (`hyperparameters`)
Define los rangos para el barrido automático (Grid Search) y la optimización de los modelos:

* **`knn`**:
    * **`k_min` / `k_max`**: Define el rango de vecinos $k$ para el barrido.
        * **Valores**: Integer.
    * **`p_min` / `p_max`**: Define la métrica de distancia de Minkowski ($p=1$: Manhattan, $p=2$: Euclídea).
        * **Valores**: Integer.
    * **`weights`**: Determina la influencia de los vecinos según su cercanía. Se puede indicar sólo un método o ambos.
        * **Valores**: Lista de String `["uniform", "distance"]`. 
* **`trees`**: 
    * **`max_depth`**: Controla la profundidad máxima del árbol para evitar el *overfitting*. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
    * **`min_samples_leaf`**: Define el número mínimo de muestras requerido en un nodo terminal. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`.
* **`random_forest`**:
    * **`n_estimators`**: Define el número de estimadores (árboles) que componen el Random Forest. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
    * **`max_depth`**: Controla la profundidad máxima del árbol para evitar el *overfitting*. Se deben indicar los valores a probar, no el rango.
        * **Valores**: Lista de Integer `[]`. 
* **`naive_bayes`**:
    * **`n_bins`**: Determina el número de intervalos para la discretización de variables continuas al usar la versión CategoricalNB. Se deben indicar los valores a probar.
        * **Valores**: Lista de Integer `[]`.
    * **`alphas`**: Define los valores del parámetro de suavizado de Laplace (Laplace smoothing) a probar en el barrido para evitar probabilidades nulas.
        * **Valores**: Lista de Float `[]`.
    * **`min_categories`**: Número mínimo de categorías esperadas por atributo. Si se desconoce, se deja vacío para que el algoritmo lo calcule automáticamente.
        * **Valores**: Integer o `null`.

---

## Requisitos
El proyecto está desarrollado en Python 3.12. Para replicar el entorno de ejecución de forma sencilla, se recomienda el uso de un entorno virtual:
```bash
# Instalación de dependencias
pip install -r requirements.txt
```

---

## Modo de Empleo

### 1. Entrenamiento y Barrido
El script de entrenamiento requiere dos argumentos por línea de comandos: el archivo de datos y el fichero de configuración.
```bash
python train.py TrainDev.csv configuration.json
```
El script comparará el F-score del mejor modelo actual en `best_model/`. Si el nuevo entrenamiento lo supera, el anterior se mueve a `archivo_versiones/` y se guarda el nuevo modelo.

***Nota:** El archivo `.csv` se puede indicar mediante su ruta directa o, si ya está `datos/` del proyecto correspondiente creado y se encuentra en esta, únicamente mediante su nombre*.

### 2. Clasificación de Instancias
Para predecir la clase de nuevas muestras, se utiliza el mejor modelo guardado en el proyecto correspondiente. Se debe indicar el archivo de datos (cargado con datos nuevos) y el nombre del proyecto. 
```bash
python test.py Test.csv NombreDelProyecto
```
***Nota:** El script de test aplica automáticamente el preproceso (escalado, imputación) utilizando los parámetros aprendidos durante el entrenamiento, pero nunca aplica balanceo a los datos de test.*

***Nota:** El archivo `.csv` se puede indicar mediante su ruta directa o, si ya está `datos/` del proyecto correspondiente creado y se encuentra en esta, únicamente mediante su nombre*.

---

## Licencia
Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## Declaración de Asistencia de IA
Se ha hecho uso de herramientas de IA Generativa (Gemini) como asistente para:

* **Estructuración técnica**: Diseño de la lógica de persistencia, partición de datos y gestión de carpetas por proyecto.
* **Depuración de código**: Resolución de problemas en el preprocesado dinámico (One-Hot Encoding con `get_dummies`) y flujos de variables.
* **Documentación**: Redacción y formato Markdown de este archivo README.

***Nota:** Todo el código ha sido validado y testeado manualmente para asegurar su integridad y cumplimiento con los objetivos de la asignatura.*
