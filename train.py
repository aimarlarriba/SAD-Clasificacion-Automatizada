import pandas as pd
import json
import sys
import pickle
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from mixed_naive_bayes import MixedNB
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder


def train():
    if len(sys.argv) < 3:
        sys.exit(1)

    f_data, f_conf = sys.argv[1], sys.argv[2]
    with open(f_conf, 'r') as f:
        config = json.load(f)

    conf_pre = config['preprocessing']
    hp = config['hyperparameters']
    target = conf_pre['target_variable']
    algoritmo_elegido = config.get('algorithm', 'todos')

    # --- CARGA Y FILTRADO DINÁMICO ---
    df = pd.read_csv(f_data)

    # Eliminar atributos según el JSON
    columnas_borrar = [c for c in conf_pre.get('drop_features', []) if c in df.columns]
    df = df.drop(columns=columnas_borrar)

    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    X = X.loc[:, (X != X.iloc[0]).any()]

    le = LabelEncoder()
    y = le.fit_transform(df[target].astype(str))

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preproceso base
    imputer = SimpleImputer(strategy=conf_pre.get('impute_strategy', 'mean'))
    X_train = imputer.fit_transform(X_train)
    X_dev = imputer.transform(X_dev)

    # --- Preproceso base ---
    scaler = None
    X_train_prep = X_train  # Datos imputados
    X_dev_prep = X_dev
    if conf_pre.get('scaling') == "standard":
        scaler = StandardScaler()
        X_train_prep = scaler.fit_transform(X_train)
        X_dev_prep = scaler.transform(X_dev)

    # --- Balanceo (Undersampling) ---
    if conf_pre.get('sampling') == "undersampling":
        rus = RandomUnderSampler(random_state=42)
        # Resampleamos los datos preparados (para KNN/Trees)
        X_train_model, y_train_model = rus.fit_resample(X_train_prep, y_train)
        # Resampleamos los datos NO escalados (para Naive Bayes)
        X_train_ns, y_train_ns = rus.fit_resample(X_train, y_train)
    else:
        X_train_model, y_train_model = X_train_prep, y_train
        X_train_ns, y_train_ns = X_train, y_train

    resultados = []
    mejor_f1, mejor_clf, mejor_prep, nombre_mejor = -1, None, None, ""
    mejor_comb = ""
    avg = 'binary' if len(le.classes_) == 2 else 'macro'

    # Matriz de confusión
    def registrar_metrica(y_true, y_pred, nombre_alg, params):
        return {
            "Combinación": f"{nombre_alg} ({params})",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precisión": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "F_score(Mac/Mic/Avg/None)": f1_score(y_true, y_pred, average=avg)
        }

    # 1. KNN
    if algoritmo_elegido in ["knn", "todos"]:
        config_weights = hp["knn"]["weights"]
        if isinstance(config_weights, str): config_weights = [config_weights]

        for k in range(hp["knn"]["k_min"], hp["knn"]["k_max"] + 1, 2):
            for p in range(hp["knn"]["p_min"], hp["knn"]["p_max"] + 1):
                for w in config_weights:
                    clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
                    clf.fit(X_train_model, y_train_model)
                    y_pred = clf.predict(X_dev_prep)

                    m = registrar_metrica(y_dev, y_pred, "KNN", f"k={k}, p={p}, w={w}")
                    resultados.append(m)
                    if m["F_score(Mac/Mic/Avg/None)"] > mejor_f1:
                        mejor_f1 = m["F_score(Mac/Mic/Avg/None)"]
                        mejor_clf, mejor_prep, nombre_mejor = clf, None, "KNN"
                        mejor_comb = m["Combinación"]

    # 2. Árboles
    if algoritmo_elegido in ["tree", "todos"]:
        for d in hp["trees"]["max_depth"]:
            for ml in hp["trees"]["min_samples_leaf"]:
                clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=ml, random_state=42)
                clf.fit(X_train_model, y_train_model)
                y_pred = clf.predict(X_dev_prep)

                m = registrar_metrica(y_dev, y_pred, "Tree", f"d={d}, ml={ml}")
                resultados.append(m)
                if m["F_score(Mac/Mic/Avg/None)"] > mejor_f1:
                    mejor_f1 = m["F_score(Mac/Mic/Avg/None)"]
                    mejor_clf, mejor_prep, nombre_mejor = clf, None, "Tree"
                    mejor_comb = m["Combinación"]

    # 3. Naive-Bayes
    if algoritmo_elegido in ["nb", "todos"]:
        # Versión Discretizada
        bins = hp.get("naive_bayes", {}).get("n_bins", 5)
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        X_train_disc = discretizer.fit_transform(X_train_ns)
        X_dev_disc = discretizer.transform(X_dev)

        clf_cat = CategoricalNB(alpha=1.0)  # Alpha según receta
        clf_cat.fit(X_train_disc, y_train_ns)
        # Obtener matriz de confusión
        m_cat = registrar_metrica(y_dev, clf_cat.predict(X_dev_disc), "CategoricalNB", f"bins={bins}")
        resultados.append(m_cat)
        if m_cat["F_score(Mac/Mic/Avg/None)"] > mejor_f1:
            mejor_f1, mejor_clf, mejor_prep, nombre_mejor = m_cat[
                "F_score(Mac/Mic/Avg/None)"], clf_cat, discretizer, "CategoricalNB"
            mejor_comb = m["Combinación"]

        # Versión MixedNB
        clf_mixed = MixedNB(categorical_features=[])
        clf_mixed.fit(X_train_ns, y_train_ns)
        # Obtener matriz de confusión
        m_mix = registrar_metrica(y_dev, clf_mixed.predict(X_dev), "MixedNB", "default")
        resultados.append(m_mix)
        if m_mix["F_score(Mac/Mic/Avg/None)"] > mejor_f1:
            mejor_f1, mejor_clf, mejor_prep, nombre_mejor = m_mix[
                "F_score(Mac/Mic/Avg/None)"], clf_mixed, None, "MixedNB"
            mejor_comb = m["Combinación"]

    # Guardado siguiendo la "Receta"
    pd.DataFrame(resultados).to_csv("resultados_entrenamiento.csv", index=False)
    pickle.dump(mejor_clf, open("bestmodel.sav", 'wb'))

    obj_guardar = {
        'imputer': imputer,
        'scaler': scaler,
        'label_encoder': le,
        'columns': X.columns,
        'discretizer': mejor_prep,
        'algoritmo': nombre_mejor,
        'combinacion_exacta': mejor_comb
    }
    pickle.dump(obj_guardar, open("preprocessing_objects.sav", 'wb'))
    print(f"Mejor modelo global: {mejor_comb} con F1: {mejor_f1}")


if __name__ == "__main__":
    train()