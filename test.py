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

    # 1. RUTA CORREGIDA: Debe coincidir con 'best_model' de tu train.py
    campeon_path = os.path.join("proyectos", proyecto, "best_model")

    if not os.path.exists(campeon_path):
        print(f"[ERROR] No existe el proyecto '{proyecto}' o no tiene modelos en 'best_model'.")
        sys.exit(1)

    # CARGA DE OBJETOS
    try:
        pre_obj = pickle.load(open(os.path.join(campeon_path, "preprocessing_objects.sav"), 'rb'))
        clf = pickle.load(open(os.path.join(campeon_path, "bestmodel.sav"), 'rb'))
    except FileNotFoundError:
        print("[ERROR] Faltan archivos .sav en la carpeta del modelo.")
        sys.exit(1)

    # 2. LOCALIZACIÓN DE DATOS
    if os.path.exists(input_csv):
        ruta_final_datos = input_csv
    else:
        ruta_final_datos = os.path.join("proyectos", proyecto, "datos", os.path.basename(input_csv))

    if not os.path.exists(ruta_final_datos):
        print(f"[ERROR] No se encuentra el archivo: {ruta_final_datos}")
        sys.exit(1)

    # 3. PROCESAMIENTO
    df = pd.read_csv(ruta_final_datos)
    le = pre_obj['label_encoder']

    # Intentamos obtener el target dinámicamente
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

    # Si el modelo necesita escalado (KNN, Trees, RF)
    if alg in ["KNN", "Tree", "Random Forest"] and pre_obj['scaler'] is not None:
        X_p = pre_obj['scaler'].transform(X_p)

    # Si es Naive Bayes Categórico y tenemos el discretizador guardado
    elif "CategoricalNB" in alg and pre_obj['discretizer'] is not None:
        X_p = pre_obj['discretizer'].transform(X_p)

    # Realizar predicción
    preds = clf.predict(X_p)

    # 4. SALIDA
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
    preds_dir = os.path.join(campeon_path, "predicciones_generadas")
    os.makedirs(preds_dir, exist_ok=True)

    # Creamos un nombre descriptivo: pred_Algoritmo_F1_Fecha_NombreOriginal.csv
    nombre_alg = pre_obj['algoritmo'].replace(" ", "_")
    f1_val = f"{pre_obj.get('f1_score', 0):.4f}"
    fecha_modelo = pre_obj.get('fecha', 'sin_fecha')

    nombre_archivo_final = f"pred_{nombre_alg}_F1_{f1_val}_{os.path.basename(ruta_final_datos)}"
    output_path = os.path.join(preds_dir, nombre_archivo_final)

    # Guardar
    df['Prediccion_Label'] = le.inverse_transform(preds)
    df.to_csv(output_path, index=False)

    print(f"\n[*] Predicciones guardadas exitosamente en:\n    {output_path}")
    print(f"\n")

if __name__ == "__main__":
    test()