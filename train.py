# ==========================================
# ENTRENAMIENTO
# ==========================================

import pandas as pd  # pandas (pd) es la librería principal para leer CSVs y manejar datos en forma de tabla (DataFrames).
import json  # json nos permite leer el archivo configuration.json donde guardamos los parámetros sin tocar el código.
import sys
import pickle  # pickle es la herramienta para "empaquetar" y guardar nuestro modelo ya entrenado (.sav).
import os
import shutil
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler  # Herramienta para balancear los datos si hay muchas más filas de una clase que de otra (recorta la clase mayoritaria).
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Partir datos en dos trozos: uno para estudiar train y otro para test/dev.
from sklearn.impute import SimpleImputer  # Herramienta que busca celdas vacías (nulos o NaN) y las rellena (con "median" "mean" "most_frequent").
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from mixed_naive_bayes import MixedNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score  # Las fórmulas matemáticas para ponerle nota a nuestro modelo.
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder  # Herramientas de preprocesado: escalar números (Z-score), hacer cajas (bins) y pasar texto a números (LabelEncoder).


# ---------------------------------------------------------
# FUNCIÓN PARA CALCULAR LAS NOTAS DEL MODELO
# ---------------------------------------------------------
def registrar_metrica(y_true, y_pred, nom, pars, avg):
    # y_true: Las respuestas correctas reales.
    # y_pred: Las respuestas que ha adivinado el modelo.
    # nom: Nombre del algoritmo (ej. "KNN").
    # pars: Los hiperparámetros usados (ej. "k=3, p=1").
    # avg: Forma de hacer la media ('binary' si son 2 clases, 'macro' si son 3 o más).

    # Accuracy: % total de aciertos
    acc = accuracy_score(y_true, y_pred)
    # Precision: De todos los que el modelo dijo que eran clase X, ¿cuántos lo eran de verdad? (zero_division=0 evita errores si da 0).
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    # Recall: De todos los que de verdad eran clase X, ¿cuántos logró encontrar el modelo?
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    # F1 Score: Es una mezcla (media armónica) entre Precision y Recall. Es la nota en la que más nos fijamos.
    f1 = f1_score(y_true, y_pred, average=avg)

    # Devuelve un diccionario con toda la fila de resultados para el CSV y, aparte, el número suelto del F1 para comparar.
    return {
        "Combinación": f"{nom} ({pars})",
        "Accuracy": acc, "Precisión": prec, "Recall": rec, "F_score": f1
    }, f1


# ---------------------------------------------------------
# FUNCIONES DE ENTRENAMIENTO ESPECÍFICAS POR ALGORITMO
# ---------------------------------------------------------

def entrenar_knn(hp, X_train, y_train, X_dev, y_dev, avg):
    # hp: contiene los hiperparámetros que configuramos en el JSON.
    resultados = []  # Aquí iremos apuntando las notas de cada intento.

    # Estas tres variables forman el "podio". Empiezan vacías o en -1.
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Probamos distintos números de vecinos (K). El range salta de 2 en 2 para probar solo impares (1, 3, 5...).
    for k in range(hp["knn"]["k_min"], hp["knn"]["k_max"] + 1, 2):

        # Probamos la forma de medir la distancia (p=1 es distancia Manhattan, p=2 es Euclídea).
        for p in range(hp["knn"].get("p_min", 1), hp["knn"].get("p_max", 2) + 1):

            # Probamos si todos los vecinos valen igual (uniform) o si el más cercano tiene más peso (distance).
            for w in hp["knn"]["weights"]:

                # 1. ENTRENAMIENTO: Creamos el modelo con los parámetros actuales y le damos los datos para que aprenda (.fit)
                clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w).fit(X_train, y_train)

                # 2. EXAMEN: Le pedimos que adivine los datos de validación (.predict) y calculamos su nota (res, val).
                res, val = registrar_metrica(y_dev, clf.predict(X_dev), "KNN", f"k={k},p={p},w={w}", avg)

                # Guardamos  en la lista general.
                resultados.append(res)

                # 3.Si esta nota (val) supera a la mejor que teníamos guardada, actualizamos.
                if val > mejor_f1_local:
                    mejor_f1_local = val
                    mejor_clf_local = clf
                    mejor_comb_local = res["Combinación"]

    # Devolvemos todas las notas y al mejor
    # (El 'None' está ahí porque KNN no usa discretizador, al contrario que Naive Bayes).
    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_arboles(hp, X_train, y_train, X_dev, y_dev, avg):
    resultados = []
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Probamos hasta qué nivel de profundidad dejamos crecer al árbol (para que no memorice demasiado).
    for d in hp["trees"]["max_depth"]:

        # Probamos cuántos datos como mínimo tienen que quedar en la última rama para dar por válida una regla.
        for ml in hp["trees"]["min_samples_leaf"]:

            # ENTRENAMIENTO: Creamos el árbol.
            # (El random_state=42 es una semilla para que, si hay empates o decisiones aleatorias, siempre elija lo mismo y sea repetible).
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=ml, random_state=42).fit(X_train, y_train)

            # EXAMEN: Calculamos la nota del árbol con los datos de prueba.
            res, val = registrar_metrica(y_dev, clf.predict(X_dev), "Tree", f"d={d},ml={ml}", avg)
            resultados.append(res)

            # Si este árbol es el mejor hasta ahora, lo guardamos.
            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf
                mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_rf(hp, X_train, y_train, X_dev, y_dev, avg):
    resultados = []

    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_comb_local = ""

    # Bucle 1: Elegimos cuántos árboles vamos a plantar (ej. 50, 100, 200).
    # Random Forest funciona poniendo a muchos árboles distintos a votar la respuesta final.
    for n in hp["random_forest"]["n_estimators"]:
        # Bucle 2: Elegimos hasta qué nivel de profundidad dejamos crecer a todos esos árboles.
        for d in hp["random_forest"]["max_depth"]:
            # 1. ENTRENAMIENTO: Creamos el bosque con 'n' árboles y profundidad 'd', y lo ponemos a estudiar (.fit).
            # (El random_state=42 vuelve a ser para que el factor "aleatorio" del bosque sea siempre igual si lo repites).
            clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(X_train, y_train)
            # 2. EXAMEN: Hacemos que el bosque intente adivinar los datos de prueba (.predict) y le calculamos la nota.
            res, val = registrar_metrica(y_dev, clf.predict(X_dev), "Random Forest", f"n={n},d={d}", avg)

            # Guardamos la nota en la lista general
            resultados.append(res)

            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf
                mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, None, mejor_comb_local


def entrenar_nb(hp, X_train_ns, y_train_ns, X_dev_imp, y_dev, avg, cat_indices):
    resultados = []
    mejor_f1_local = -1
    mejor_clf_local = None
    mejor_prep_local = None
    mejor_comb_local = ""

    # Extraemos min_categories (por defecto None si no está en el JSON)
    min_cat = hp["naive_bayes"].get("min_categories", None)

    # Bucle 1: Barrido del parámetro alpha (Laplace smoothing)
    for a in hp["naive_bayes"].get("alphas", [1.0]):

        # --- VERSIÓN 1: Categorical Naive Bayes ---
        for bins in hp["naive_bayes"].get("n_bins", [5]):
            disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
            X_train_nb = disc.fit_transform(X_train_ns)

            # Instanciamos aplicando alpha y min_categories
            clf_cat = CategoricalNB(alpha=a, min_categories=min_cat).fit(X_train_nb, y_train_ns)

            res, val = registrar_metrica(y_dev, clf_cat.predict(disc.transform(X_dev_imp)), "CategoricalNB",
                                         f"bins={bins},alpha={a}", avg)
            resultados.append(res)

            if val > mejor_f1_local:
                mejor_f1_local = val
                mejor_clf_local = clf_cat
                mejor_prep_local = disc
                mejor_comb_local = res["Combinación"]

        # --- VERSIÓN 2: Mixed Naive Bayes ---
        # Le pasamos los índices detectados automáticamente y el alpha actual
        clf_mix = MixedNB(categorical_features=cat_indices, alpha=a).fit(X_train_ns, y_train_ns)

        res, val = registrar_metrica(y_dev, clf_mix.predict(X_dev_imp), "MixedNB", f"alpha={a}", avg)
        resultados.append(res)

        if val > mejor_f1_local:
            mejor_f1_local = val
            mejor_clf_local = clf_mix
            mejor_prep_local = None
            mejor_comb_local = res["Combinación"]

    return resultados, mejor_f1_local, mejor_clf_local, mejor_prep_local, mejor_comb_local


# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ---------------------------------------------------------
def train():
    # 1. VERIFICAR ARGUMENTOS DE TERMINAL
    # sys.argv es una lista para consola
    # sys.argv[0] es "train.py", sys.argv[1] es el "csv", sys.argv[2] es "config.json".
    if len(sys.argv) < 3:
        print("\n[!] Uso: python train.py <ruta_csv_o_nombre> <config.json>")
        sys.exit(1)  # Corta la ejecución si no le has dado los archivos

    f_data_input = sys.argv[1]
    f_conf = sys.argv[2]

    # 2. LEER JSON DE CONFIGURACIÓN
    with open(f_conf, 'r') as f:
        config = json.load(f)

    proyecto = config.get("project_name", "Proyecto_Generico")

    # --- AJUSTE DE RUTA DINÁMICA ---
    # Si el archivo no existe en la ruta actual, lo busca en la carpeta 'datos' del proyecto
    if not os.path.exists(f_data_input):
        posible_ruta = os.path.join("proyectos", proyecto, "datos", f_data_input)
        if os.path.exists(posible_ruta):
            f_data_input = posible_ruta
        else:
            print(f"[ERROR] No se encuentra el archivo: {f_data_input}")
            sys.exit(1)

    # Extraemos las variables para usarlas más fácil. Si no existe en el json, coge el valor por defecto (ej. "Proyecto_Generico")
    conf_pre = config['preprocessing']
    hp = config['hyperparameters']
    target = conf_pre['target_variable']  # Esta es la columna que queremos adivinar (ej. "Species" o "Clase").
    algoritmo_elegido = config.get('algorithm', 'todos')  # Permite correr solo knn, solo tree, o todos.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")  # Para crear carpetas con la fecha actual.

    # 3. CREAR CARPETAS
    base_path = os.path.join("proyectos", proyecto)
    data_path = os.path.join(base_path, "datos")
    best_path = os.path.join(base_path, "best_model")
    archive_path = os.path.join(base_path,
                                "archivo_versiones")  # Aquí guardamos modelos viejos si encontramos uno mejor.

    for folder in [data_path, best_path, archive_path]:
        os.makedirs(folder, exist_ok=True)  # exist_ok=True evita que de error si la carpeta ya existe.

    # 4. CARGAR DATOS
    df_full = pd.read_csv(f_data_input)  # Lee el excel/csv

    # LabelEncoder coge las clases en texto (ej. "Gato", "Perro") y las vuelve números (0, 1) que la máquina entiende.
    le = LabelEncoder()
    y_full = le.fit_transform(df_full[target].astype(str))  # target es la columna objetivo del JSON.

    # Si hay 2 clases se puntúa como 'binary', si hay más de 2 (ej. Iris tiene 3) se puntúa como 'macro'.
    avg = 'binary' if len(le.classes_) == 2 else 'macro'

    # En el JSON podemos decir si queremos separar un trozo automático para Test. Si es 0, no separa nada.
    split_pct = conf_pre.get('test_split', 0)

    # 5. DIVISIÓN INICIAL (Opcional según JSON)
    if split_pct > 0:
        # stratify=y_full: asegura que si hay 80% gatos y 20% perros en el archivo original,
        # en el trozo de test haya también 80% gatos y 20% perros.
        df, df_test_auto = train_test_split(df_full, test_size=split_pct, random_state=42, stratify=y_full)
        # Guarda ese test a un lado para usarlo en el examen con test.py
        f_test_auto = os.path.join(data_path, f"test_automatico_{proyecto}.csv")
        df_test_auto.to_csv(f_test_auto, index=False)
    else:
        df = df_full  # Si split_pct era 0, usamos todo el archivo para entrenar.

    # Guarda una copia de seguridad exacta de lo que vamos a usar para entrenar.
    nombre_csv = os.path.basename(f_data_input)
    f_data_final = os.path.join(data_path, f"entrenamiento_{nombre_csv}")
    df.to_csv(f_data_final, index=False)

    # 6. LIMPIEZA Y PREPROCESADO
    # Si en el json pusiste "drop_features": ["ID", "Nombre"], aquí borra esas columnas porque no sirven para predecir.
    df_clean = df.drop(columns=[c for c in conf_pre.get('drop_features', []) if c in df.columns])

    # get_dummies es el One-Hot Encoding: convierte columnas de texto (ej. Color: Rojo, Verde) en varias columnas binarias (Color_Rojo: 1 o 0). drop_first=True evita redundancias matemáticas.
    X_cols = pd.get_dummies(df_clean.drop(columns=[target]), drop_first=True)

    # Esto borra las columnas que sean todas iguales (si una colmna no aporta información, se borra).
    X_cols = X_cols.loc[:, (X_cols != X_cols.iloc[0]).any()]

    # La 'y' es nuestra columna de soluciones pasada a números
    y = le.transform(df_clean[target].astype(str))

    # 7. DIVISIÓN TRAIN / (DEV)
    # Partimos los datos: 80% (train) para estudiar y 20% (dev) para hacer simulacros de examen internos
    X_train, X_dev, y_train, y_dev = train_test_split(X_cols, y, test_size=0.2, random_state=42, stratify=y)

    # 8. IMPUTACIÓN (Rellenar huecos vacíos)
    imputer = None
    X_train_imp, X_dev_imp = X_train, X_dev  # Por defecto, los datos se quedan igual

    # Solo imputamos si el JSON lo pide explícitamente
    if conf_pre.get('missing_values') == "impute":
        imputer = SimpleImputer(strategy=conf_pre.get('impute_strategy', 'mean')).fit(X_train)
        X_train_imp = imputer.transform(X_train)
        X_dev_imp = imputer.transform(X_dev)

    # 9. ESCALADO (Z-Score)
    scaler = None  # Por defecto nada
    X_train_prep, X_dev_prep = X_train_imp, X_dev_imp  # Copiamos los datos tal cual

    # Si el json pide "standard", aplica la fórmula matemática para que todos los números estén en la misma escala (para knn)
    if conf_pre.get('scaling') == "standard":
        scaler = StandardScaler().fit(X_train_imp)
        X_train_prep = scaler.transform(X_train_imp)
        X_dev_prep = scaler.transform(X_dev_imp)

    # 10. BALANCEO DE CLASES
    # Si hay 90 sanos y 10 enfermos, undersampling borra 80 sanos al azar para que quede 10 y 10.
    estrategia_balanceo = conf_pre.get('sampling')

    # Opción 1: Undersampling (recorta la clase mayoritaria)
    if estrategia_balanceo == "undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train_model, y_train_model = rus.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = rus.fit_resample(X_train_imp, y_train)

    # Opción 2: Oversampling/SMOTE (crea datos sintéticos para la clase minoritaria)
    elif estrategia_balanceo == "oversampling":
        smote = SMOTE(random_state=42)
        X_train_model, y_train_model = smote.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = smote.fit_resample(X_train_imp, y_train)

    # Opción 3: Ningún balanceo
    else:
        X_train_model, y_train_model = X_train_prep, y_train
        X_train_ns, y_train_ns = X_train_imp, y_train

    # 11.Llamada a algoritmos
    # Aquí vamos guardando al mejor de entre todos los algoritmos.
    resultados_globales = []
    mejor_f1_global = -1;
    mejor_clf_global = None;
    mejor_prep_global = None;
    nombre_mejor_global = "";
    mejor_comb_global = ""

    # Si en el json pone "knn" o "todos", entra aquí. (Sigue la misma lógica para el resto).
    if algoritmo_elegido in ["knn", "todos"]:
        res, f1, clf, prep, comb = entrenar_knn(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:  # Si este KNN superó la mejor nota global actual, se pone la corona.
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "KNN", comb

    if algoritmo_elegido in ["tree", "todos"]:
        res, f1, clf, prep, comb = entrenar_arboles(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "Tree", comb

    if algoritmo_elegido in ["rf", "todos"]:
        res, f1, clf, prep, comb = entrenar_rf(hp, X_train_model, y_train_model, X_dev_prep, y_dev, avg)
        resultados_globales.extend(res)
        if f1 > mejor_f1_global:
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "Random Forest", comb

    if algoritmo_elegido in ["nb", "todos"]:
        # Detectamos automáticamente qué columnas son categóricas (las que tienen 2 o menos valores únicos, ej. 0 y 1)
        indices_categoricos = [i for i, col in enumerate(X_cols.columns) if X_cols[col].nunique() <= 2]

        # Convertimos a DataFrame si son arrays de numpy para manejar mejor las columnas
        if isinstance(X_train_ns, pd.DataFrame):
            X_train_nb_df = X_train_ns.copy()
            X_dev_nb_df = X_dev_imp.copy()
        else:
            X_train_nb_df = pd.DataFrame(X_train_ns, columns=X_cols.columns)
            X_dev_nb_df = pd.DataFrame(X_dev_imp, columns=X_cols.columns)

        for col_idx in indices_categoricos:
            col_name = X_cols.columns[col_idx]

            # Forzamos a entero para evitar problemas de float64 vs int
            # Y usamos fit con el UNION de ambos sets para que no haya "unseen labels"
            le_temp = LabelEncoder()
            combined_values = pd.concat([X_train_nb_df[col_name], X_dev_nb_df[col_name]]).astype(str)
            le_temp.fit(combined_values)

            X_train_nb_df[col_name] = le_temp.transform(X_train_nb_df[col_name].astype(str))
            X_dev_nb_df[col_name] = le_temp.transform(X_dev_nb_df[col_name].astype(str))

        # Pasamos los datos limpios a la función de entrenamiento
        res, f1, clf, prep, comb = entrenar_nb(hp, X_train_nb_df.values, y_train_ns, X_dev_nb_df.values, y_dev, avg, indices_categoricos)

        resultados_globales.extend(res)
        if f1 > mejor_f1_global:
            mejor_f1_global, mejor_clf_global, mejor_prep_global, nombre_mejor_global, mejor_comb_global = f1, clf, prep, "Naive Bayes", comb

    # 12. GUARDADO DEL MEJOR
    ruta_obj_best_model = os.path.join(best_path, "preprocessing_objects.sav")
    ruta_best_model = os.path.join(best_path, "bestmodel.sav")

    # Revisa en la carpeta si ya existía un modelo guardado de días anteriores y mira su nota F1.
    f1_actual = 0.0
    if os.path.exists(ruta_obj_best_model):
        with open(ruta_obj_best_model, 'rb') as f:
            f1_actual = pickle.load(f).get('f1_score', 0.0)

    # Si nuestro campeón de hoy tiene mejor F1 que el campeón antiguo...
    if mejor_f1_global > f1_actual:
        if os.path.exists(ruta_obj_best_model):
            # Coge al campeón antiguo y lo manda a la carpeta "archivo_versiones" cambiándole el nombre.
            nombre_archivo = f"v_F1_{f1_actual:.4f}_{timestamp}"
            folder_archivo = os.path.join(archive_path, nombre_archivo)
            os.makedirs(folder_archivo, exist_ok=True)
            shutil.move(ruta_obj_best_model, os.path.join(folder_archivo, "preprocessing_objects.sav"))
            shutil.move(ruta_best_model, os.path.join(folder_archivo, "bestmodel.sav"))

        # Guarda el cerebro del nuevo algoritmo usando pickle.
        pickle.dump(mejor_clf_global, open(ruta_best_model, 'wb'))

        # Guarda todas las herramientas que usamos (imputer, scaler, columnas originales, etc).
        # Esto es vital: si escalamos el train, el script de Test necesita ESTE MISMO escalador para aplicar la misma transformación.
        obj_final = {
            'target_variable': target,
            'imputer': imputer, 'scaler': scaler, 'label_encoder': le,
            'columns': X_cols.columns, 'discretizer': mejor_prep_global,
            'algoritmo': nombre_mejor_global, 'f1_score': mejor_f1_global,
            'combinacion_exacta': mejor_comb_global, 'fecha': timestamp,
            'project_name': proyecto
        }
        pickle.dump(obj_final, open(ruta_obj_best_model, 'wb'))

        # Guarda un excel (csv) con TODAS las notas de TODAS las combinaciones probadas.
        pd.DataFrame(resultados_globales).to_csv(os.path.join(best_path, "ultimos_resultados.csv"), index=False)
        print(f"\n[!] NUEVO MEJOR MODELO: {mejor_comb_global} con F1: {mejor_f1_global:.4f}")
    else:
        print(f"\n[-] No hay mejora respecto al mejor modelo.")


# Esto asegura que la función train() solo arranque si ejecutas el archivo directamente.
if __name__ == "__main__":
    train()
