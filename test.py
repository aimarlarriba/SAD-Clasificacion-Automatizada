# ==========================================
# SCRIPT DE EVALUACIÓN / PREDICCIÓN
# ==========================================

import shutil  # Para copiar el CSV de test dentro de nuestra estructura de carpetas si está fuera
import pandas as pd  # Para manejar las tablas
import sys  # Para leer qué CSV desde la terminal
import pickle  # Para cargar el modelo ganador guardado en train.py
import os  # Para las rutas de carpetas
import string
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score  # Fórmulas para comprobar qué tal lo ha hecho adivinando
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def limpiar_texto_libre(texto, idioma):
    try:
        stop_words = set(stopwords.words(idioma))
    except:
        stop_words = set()
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(texto).lower())
    tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def test():
    # 1. COMPROBAR TERMINAL
    # Necesitamos el comando de python, el nombre del csv y el nombre del proyecto.
    if len(sys.argv) < 3:
        print("\n[!] Uso: python test.py <ruta_o_nombre_csv> <nombre_proyecto>")
        sys.exit(1)

    input_csv, proyecto = sys.argv[1], sys.argv[2]

    # 2. LOCALIZAR CARPETAS
    # Entra en la carpeta /proyectos/ y luego en la carpeta de nuestro proyecto en concreto.
    base_path = os.path.join("proyectos", proyecto)
    data_path = os.path.join(base_path, "datos")
    best_model_path = os.path.join(base_path, "best_model")

    # Si el archivo no existe en la ruta actual, lo busca en la carpeta 'datos' del proyecto
    if not os.path.exists(input_csv):
        posible_ruta = os.path.join(data_path, input_csv)
        if os.path.exists(posible_ruta):
            input_csv = posible_ruta
        else:
            print(f"[ERROR] No se encuentra el archivo de entrada: {input_csv}")
            sys.exit(1)

    # Chequea que el proyecto existió y se entrenó (debe haber una carpeta 'best_model').
    if not os.path.exists(best_model_path):
        print(f"[ERROR] No existe el proyecto '{proyecto}' o no tiene modelos en 'best_model'.")
        sys.exit(1)

    # 3. ORGANIZAR DATOS
    nombre_csv = os.path.basename(
        input_csv)  # Coge solo el nombre, ignorando la ruta entera (ej. C:/descargas/test.csv -> test.csv)
    ruta_final_datos = os.path.join(data_path,
                                    f"evaluacion_{nombre_csv}")  # Crea la ruta donde se va a guardar una copia.

    # Si el archivo CSV lo tenemos en el escritorio, lo copia automáticamente a la carpeta /datos/ del proyecto.
    if os.path.abspath(input_csv) != os.path.abspath(ruta_final_datos):
        os.makedirs(data_path, exist_ok=True)
        shutil.copy2(input_csv, ruta_final_datos)

    # 4. CARGAR HERRAMIENTAS Y MODELO (Deserializar)
    try:
        # Abrimos (en modo 'rb' -> read binary) los archivos .sav y recuperamos los objetos.
        pre_obj = pickle.load(open(os.path.join(best_model_path, "preprocessing_objects.sav"), 'rb'))
        clf = pickle.load(open(os.path.join(best_model_path, "bestmodel.sav"), 'rb'))
        target_col = pre_obj['target_variable']
        vectorizador = pre_obj.get('vectorizador_texto')
        text_columns = pre_obj.get('text_columns_original', [])
    except FileNotFoundError:
        print("[ERROR] Faltan archivos .sav en la carpeta del modelo.")
        sys.exit(1)

    # 5. PREPROCESADO IGUAL QUE EN EL TRAIN
    df = pd.read_csv(ruta_final_datos)  # Leemos los datos nuevos

    # Si hay un vectorizador guardado, procesamos el texto automáticamente
    if vectorizador is not None and len(text_columns) > 0:
        print(f"[*] Detectado modelo de texto. Procesando columnas: {text_columns}")

        # El idioma también lo podemos sacar del objeto si lo guardaste, o fijarlo a 'spanish'
        idioma = pre_obj.get('language', 'spanish')

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: limpiar_texto_libre(x, idioma))

        texto_unido = df[text_columns].apply(lambda x: ' '.join(x), axis=1)
        X_text_transformed = vectorizador.transform(texto_unido)

        df_text_num = pd.DataFrame(X_text_transformed.toarray(),
                                   columns=vectorizador.get_feature_names_out(),
                                   index=df.index)

        df_proc = df.drop(columns=text_columns)
        df_proc = pd.concat([df_proc, df_text_num], axis=1)
    else:
        df_proc = df.copy()

    # Recuperamos de pre_obj las variables que el train consideró importantes
    le = pre_obj['label_encoder']  # Letras -> Números de la clase
    target_col = pre_obj['target_variable']  # El nombre de la columna que queremos predecir

    # ¿Viene la solución en este CSV?
    # Si existe la columna objetivo, guardamos las respuestas correctas en y_true transformándolas a números. Si no, le damos valor None.
    y_true = le.transform(df[target_col].astype(str)) if target_col in df.columns else None

    # X son los datos que le enseñamos al modelo. Usamos df_proc que ya tiene el texto procesado
    X = df_proc.drop(columns=[target_col]) if target_col in df_proc.columns else df_proc
    # get_dummies transforma categorías a binario.
    # .reindex(columns=pre_obj['columns']) Si el train vio "Color_Rojo" y "Color_Verde", pero en el CSV de test
    # no hay ningún rojo, get_dummies no crearía esa columna y el modelo reventaría porque le faltan datos
    # reindex fuerza a que las columnas sean idénticas al train, rellenando con ceros si falta algo.
    X_p = pd.get_dummies(X, drop_first=True).reindex(columns=pre_obj['columns'], fill_value=0)

    # Aplicamos el imputador solo si el train nos guardó uno
    if pre_obj['imputer'] is not None:
        X_p = pre_obj['imputer'].transform(X_p)

    # Vemos qué algoritmo ganó en train para saber qué último preprocesado aplicar.
    alg = pre_obj['algoritmo']

    # Si ganó uno de estos y encima había escalador guardado...
    if alg in ["KNN", "Tree", "Random Forest"] and pre_obj['scaler'] is not None:
        X_p = pre_obj['scaler'].transform(X_p)  # ... los escala con el Z-score del train.

    # Si ganó Naive Bayes y guardó un discretizador (cajas)...
    elif alg == "Naive Bayes" and pre_obj['discretizer'] is not None:
        X_p = pre_obj['discretizer'].transform(X_p)  # ... empaqueta los datos de test en esas mismas cajas.

    # 6. EL MOMENTO DE LA VERDAD: PREDICCIÓN
    # clf (clasificador) suelta sus predicciones basándose en los datos nuevos ya preprocesados.
    preds = clf.predict(X_p)


    print("\n" + "=" * 50)
    print(f"PROYECTO: {proyecto} | Algoritmo: {alg}")
    print(f"Combinación: {pre_obj.get('combinacion_exacta', 'N/A')}")
    print("=" * 50)

    # 7. EXAMINAR RESULTADOS (Si tenemos las respuestas reales)
    if y_true is not None:
        # Vuelve a verificar cómo calcular la nota media (binary o macro)
        avg = pre_obj['average_strategy']
        print(f"F-Score (Val): {f1_score(y_true, preds, average=avg):.4f}")
        print(f"Accuracy: {accuracy_score(y_true, preds):.4f}")
        print(f"Precisión: {precision_score(y_true, preds, average=avg, zero_division=0):.4f}")
        print(f"Recall: {recall_score(y_true, preds, average=avg, zero_division=0):.4f}")
        # Matriz de confusión: Muestra visualmente dónde ha acertado y dónde se ha liado.
        # Filas: Lo que era en realidad. Columnas: Lo que predijo el modelo.
        print("\nMatriz de Confusión:")
        conf_df = pd.DataFrame(confusion_matrix(y_true, preds), index=le.classes_, columns=le.classes_)
        print(conf_df)

    # 8. GUARDAR LAS PREDICCIONES
    # Creamos carpeta si no existe
    preds_dir = os.path.join(best_model_path, "predicciones_generadas")
    os.makedirs(preds_dir, exist_ok=True)

    # Formateamos el nombre del archivo para que incluya qué modelo es y qué nota sacó en train.
    nombre_alg = pre_obj['algoritmo'].replace(" ", "_")
    f1_val = f"{pre_obj.get('f1_score', 0):.4f}"
    nombre_archivo_final = f"pred_{nombre_alg}_F1_{f1_val}_{nombre_csv}"
    output_path = os.path.join(preds_dir, nombre_archivo_final)

    # Añadimos las predicciones al excel original.
    # le.inverse_transform(preds) coge los [0, 1] y los vuelve a transformar en ["Gato", "Perro"] para que lo leamos los humanos.
    df['Prediccion_Label'] = le.inverse_transform(preds)

    # Guardamos el CSV definitivo
    df.to_csv(output_path, index=False)

    print(f"\n[*] Predicciones guardadas exitosamente en:\n    {output_path}\n")


if __name__ == "__main__":
    test()
