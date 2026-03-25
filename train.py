import pandas as pd
import json
import sys
import pickle
import os
import shutil
from datetime import datetime
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from mixed_naive_bayes import MixedNB
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder


def train():
    if len(sys.argv) < 3:
        print("\n[!] Uso: python train.py <ruta_csv_o_nombre> <config.json>")
        sys.exit(1)

    f_data_input = sys.argv[1]
    f_conf = sys.argv[2]

    with open(f_conf, 'r') as f:
        config = json.load(f)

    # --- CONFIGURACIÓN DE PROYECTO ---
    proyecto = config.get("project_name", "Proyecto_Generico")
    conf_pre = config['preprocessing']
    hp = config['hyperparameters']
    target = conf_pre['target_variable']
    algoritmo_elegido = config.get('algorithm', 'todos')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # --- ESTRUCTURA DE CARPETAS PROFESIONAL ---
    base_path = os.path.join("proyectos", proyecto)
    data_path = os.path.join(base_path, "datos")
    best_path = os.path.join(base_path, "best_model")
    archive_path = os.path.join(base_path, "archivo_versiones")

    for folder in [data_path, best_path, archive_path]:
        os.makedirs(folder, exist_ok=True)

    # --- CARGA Y DIVISIÓN ESTRATIFICADA ---
    df_full = pd.read_csv(f_data_input)

    le = LabelEncoder()
    y_full = le.fit_transform(df_full[target].astype(str))

    # Definición de AVG: 'binary' si hay 2 clases, 'macro' si hay más
    avg = 'binary' if len(le.classes_) == 2 else 'macro'
    split_pct = conf_pre.get('test_split', 0)

    if split_pct > 0:
        print(f"[*] Generando partición de test estratificada ({split_pct * 100}%)...")
        # Aseguramos que la distribución de clases sea idéntica en train y test
        df, df_test_auto = train_test_split(
            df_full,
            test_size=split_pct,
            random_state=42,
            stratify=y_full
        )
        # Guardamos el archivo de test para uso posterior
        f_test_auto = os.path.join(data_path, f"test_automatico_{proyecto}.csv")
        df_test_auto.to_csv(f_test_auto, index=False)
    else:
        df = df_full

    # Guardar copia de seguridad de los datos de entrenamiento
    nombre_csv = os.path.basename(f_data_input)
    f_data_final = os.path.join(data_path, f"entrenamiento_{nombre_csv}")
    df.to_csv(f_data_final, index=False)

    # --- PREPROCESADO ---
    df_clean = df.drop(columns=[c for c in conf_pre.get('drop_features', []) if c in df.columns])

    X_cols = pd.get_dummies(df_clean.drop(columns=[target]), drop_first=True)
    X_cols = X_cols.loc[:, (X_cols != X_cols.iloc[0]).any()]

    y = le.transform(df_clean[target].astype(str))

    # Split interno para validación durante el entrenamiento
    X_train, X_dev, y_train, y_dev = train_test_split(X_cols, y, test_size=0.2, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy=conf_pre.get('impute_strategy', 'mean')).fit(X_train)
    X_train_imp = imputer.transform(X_train)
    X_dev_imp = imputer.transform(X_dev)

    scaler = None
    X_train_prep, X_dev_prep = X_train_imp, X_dev_imp
    if conf_pre.get('scaling') == "standard":
        scaler = StandardScaler().fit(X_train_imp)
        X_train_prep = scaler.transform(X_train_imp)
        X_dev_prep = scaler.transform(X_dev_imp)

    # Balanceo (Undersampling)
    if conf_pre.get('sampling') == "undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_train_model, y_train_model = rus.fit_resample(X_train_prep, y_train)
        X_train_ns, y_train_ns = rus.fit_resample(X_train_imp, y_train)
    else:
        X_train_model, y_train_model = X_train_prep, y_train
        X_train_ns, y_train_ns = X_train_imp, y_train

    # --- MÉTRICAS DETALLADAS ---
    resultados = []
    mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = -1, None, None, "", ""

    def registrar_metrica(y_true, y_pred, nom, pars):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=avg)
        return {
            "Combinación": f"{nom} ({pars})",
            "Accuracy": acc, "Precisión": prec, "Recall": rec, "F_score": f1
        }, f1

    # --- BUCLES DE ENTRENAMIENTO ---

    # 1. KNN (Con Tuning Completo)
    if algoritmo_elegido in ["knn", "todos"]:
        for k in range(hp["knn"]["k_min"], hp["knn"]["k_max"] + 1, 2):
            for p in range(hp["knn"].get("p_min", 1), hp["knn"].get("p_max", 2) + 1):
                for w in hp["knn"]["weights"]:
                    clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w).fit(X_train_model, y_train_model)
                    res, val = registrar_metrica(y_dev, clf.predict(X_dev_prep), "KNN", f"k={k},p={p},w={w}")
                    resultados.append(res)
                    if val > mejor_f1: mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = val, clf, None, "KNN", \
                    res["Combinación"]

    # 2. Árboles
    if algoritmo_elegido in ["tree", "todos"]:
        for d in hp["trees"]["max_depth"]:
            for ml in hp["trees"]["min_samples_leaf"]:
                clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=ml, random_state=42).fit(X_train_model,
                                                                                                    y_train_model)
                res, val = registrar_metrica(y_dev, clf.predict(X_dev_prep), "Tree", f"d={d},ml={ml}")
                resultados.append(res)
                if val > mejor_f1: mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = val, clf, None, "Tree", \
                res["Combinación"]

    # 3. Random Forest
    if algoritmo_elegido in ["rf", "todos"]:
        for n in hp["rf"]["n_estimators"]:
            for d in hp["rf"]["max_depth"]:
                clf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(X_train_model,
                                                                                               y_train_model)
                res, val = registrar_metrica(y_dev, clf.predict(X_dev_prep), "Random Forest", f"n={n},d={d}")
                resultados.append(res)
                if val > mejor_f1: mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = val, clf, None, "Random Forest", \
                    res["Combinación"]

    # 4. Naive Bayes (Categorical + Mixed)
    if algoritmo_elegido in ["nb", "todos"]:
        bins = hp["naive_bayes"]["n_bins"]
        disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        X_train_nb = disc.fit_transform(X_train_ns)

        # CategoricalNB
        clf_cat = CategoricalNB().fit(X_train_nb, y_train_ns)
        res, val = registrar_metrica(y_dev, clf_cat.predict(disc.transform(X_dev_imp)), "CategoricalNB",
                                     f"bins={bins}")
        resultados.append(res)
        if val > mejor_f1: mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = val, clf_cat, disc, "CategoricalNB", \
        res["Combinación"]

        # MixedNB (Lo recuperamos del programa 1)
        clf_mix = MixedNB(categorical_features=[]).fit(X_train_ns, y_train_ns)
        res, val = registrar_metrica(y_dev, clf_mix.predict(X_dev_imp), "MixedNB", "default")
        resultados.append(res)
        if val > mejor_f1: mejor_f1, mejor_clf, mejor_prep, nombre_mejor, mejor_comb = val, clf_mix, None, "MixedNB", \
        res["Combinación"]

    # --- LÓGICA DE BEST MODEL ---
    ruta_obj_best_model = os.path.join(best_path, "preprocessing_objects.sav")
    ruta_best_model = os.path.join(best_path, "bestmodel.sav")

    f1_actual = 0.0
    if os.path.exists(ruta_obj_best_model):
        with open(ruta_obj_best_model, 'rb') as f:
            f1_actual = pickle.load(f).get('f1_score', 0.0)

    if mejor_f1 > f1_actual:
        if os.path.exists(ruta_obj_best_model):
            # Movemos el antiguo mejor modelo al archivo histórico
            nombre_archivo = f"v_F1_{f1_actual:.4f}_{timestamp}"
            folder_archivo = os.path.join(archive_path, nombre_archivo)
            os.makedirs(folder_archivo, exist_ok=True)
            shutil.move(ruta_obj_best_model, os.path.join(folder_archivo, "preprocessing_objects.sav"))
            shutil.move(ruta_best_model, os.path.join(folder_archivo, "bestmodel.sav"))
            print(f"\n[+] Versión anterior archivada en: {nombre_archivo}")

        # Guardamos el nuevo mejor modelo
        pickle.dump(mejor_clf, open(ruta_best_model, 'wb'))
        obj_final = {
            'target_variable': target,
            'imputer': imputer, 'scaler': scaler, 'label_encoder': le,
            'columns': X_cols.columns, 'discretizer': mejor_prep,
            'algoritmo': nombre_mejor, 'f1_score': mejor_f1,
            'combinacion_exacta': mejor_comb, 'fecha': timestamp,
            'project_name': proyecto
        }
        pickle.dump(obj_final, open(ruta_obj_best_model, 'wb'))

        # Guardar CSV de resultados de la ejecución actual
        pd.DataFrame(resultados).to_csv(os.path.join(best_path, "ultimos_resultados.csv"), index=False)
        print(f"\n[!] NUEVO MEJOR MODELO: {mejor_comb} con F1: {mejor_f1:.4f}")
    else:
        print(f"\n[-] No hay mejora respecto al mejor modelo.")

    print(f"\n")


if __name__ == "__main__":
    train()