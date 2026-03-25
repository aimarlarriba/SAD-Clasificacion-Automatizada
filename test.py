import shutil
import pandas as pd
import sys
import pickle
import os
from sklearn.metrics import confusion_matrix, f1_score


def test():
    if len(sys.argv) < 3:
        print("\n[!] Uso: python test.py <ruta_o_nombre_csv> <nombre_proyecto>")
        sys.exit(1)

    input_csv, proyecto = sys.argv[1], sys.argv[2]

    base_path = os.path.join("proyectos", proyecto)
    data_path = os.path.join(base_path, "datos")
    best_model_path = os.path.join(base_path, "best_model")

    # SE VERIFICA QUE EXISTE EL MODELO
    if not os.path.exists(best_model_path):
        print(f"[ERROR] No existe el proyecto '{proyecto}' o no tiene modelos en 'best_model'.")
        sys.exit(1)

    # --- LOCALIZACIÓN Y COPIA DE DATOS ---
    if not os.path.exists(input_csv):
        print(f"[ERROR] El archivo de entrada no existe: {input_csv}")
        sys.exit(1)

    nombre_csv = os.path.basename(input_csv)
    # Definimos la ruta destino dentro de la carpeta del proyecto
    ruta_final_datos = os.path.join(data_path, f"evaluacion_{nombre_csv}")

    # Si el archivo no está en la carpeta de datos, lo copiamos
    if os.path.abspath(input_csv) != os.path.abspath(ruta_final_datos):
        os.makedirs(data_path, exist_ok=True)
        shutil.copy2(input_csv, ruta_final_datos)
        print(f"[*] Archivo copiado a la estructura del proyecto: {ruta_final_datos}")

    # --- CARGA DE OBJETOS ---
    try:
        pre_obj = pickle.load(open(os.path.join(best_model_path, "preprocessing_objects.sav"), 'rb'))
        clf = pickle.load(open(os.path.join(best_model_path, "bestmodel.sav"), 'rb'))
    except FileNotFoundError:
        print("[ERROR] Faltan archivos .sav en la carpeta del modelo.")
        sys.exit(1)

    # --- PROCESAMIENTO ---
    df = pd.read_csv(ruta_final_datos)
    le = pre_obj['label_encoder']
    target_col = pre_obj['target_variable']

    # Separar Target y Features
    y_true = le.transform(df[target_col].astype(str)) if target_col in df.columns else None
    X = df.drop(columns=[target_col]) if target_col in df.columns else df

    # Reconstrucción de columnas (One-Hot Encoding)
    X_p = pd.get_dummies(X, drop_first=True).reindex(columns=pre_obj['columns'], fill_value=0)

    # Imputación
    X_p = pre_obj['imputer'].transform(X_p)

    # APLICACIÓN DE TRANSFORMACIONES SEGÚN EL MODELO
    alg = pre_obj['algoritmo']

    if alg in ["KNN", "Tree", "Random Forest"] and pre_obj['scaler'] is not None:
        X_p = pre_obj['scaler'].transform(X_p)
    elif "CategoricalNB" in alg and pre_obj['discretizer'] is not None:
        X_p = pre_obj['discretizer'].transform(X_p)

    # Realizar predicción
    preds = clf.predict(X_p)

    # --- SALIDA ---
    print("\n" + "=" * 50)
    print(f"PROYECTO: {proyecto} | Algoritmo: {alg}")
    print(f"Combinación: {pre_obj.get('combinacion_exacta', 'N/A')}")
    print("=" * 50)

    if y_true is not None:
        avg = 'binary' if len(le.classes_) == 2 else 'macro'
        print(f"F-Score (Val): {f1_score(y_true, preds, average=avg):.4f}")
        print("\nMatriz de Confusión:")
        conf_df = pd.DataFrame(confusion_matrix(y_true, preds), index=le.classes_, columns=le.classes_)
        print(conf_df)

    # Guardar predicciones
    preds_dir = os.path.join(best_model_path, "predicciones_generadas")
    os.makedirs(preds_dir, exist_ok=True)

    nombre_alg = pre_obj['algoritmo'].replace(" ", "_")
    f1_val = f"{pre_obj.get('f1_score', 0):.4f}"

    # Usamos nombre_csv (el original) para el nombre del archivo de salida
    nombre_archivo_final = f"pred_{nombre_alg}_F1_{f1_val}_{nombre_csv}"
    output_path = os.path.join(preds_dir, nombre_archivo_final)

    # Guardar
    df['Prediccion_Label'] = le.inverse_transform(preds)
    df.to_csv(output_path, index=False)

    print(f"\n[*] Predicciones guardadas exitosamente en:\n    {output_path}\n")


if __name__ == "__main__":
    test()