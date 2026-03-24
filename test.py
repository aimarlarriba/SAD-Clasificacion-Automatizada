import pandas as pd
import sys
import pickle


def test():
    if len(sys.argv) < 2:
        sys.exit(1)

    f_test = sys.argv[1]
    clf = pickle.load(open("bestmodel.sav", 'rb'))
    pre_obj = pickle.load(open("preprocessing_objects.sav", 'rb'))

    df_test = pd.read_csv(f_test)

    if 'Especie' in df_test.columns:
        df_test = df_test.drop(columns=['Especie'])

    X_test = pd.get_dummies(df_test, drop_first=True)
    X_test = X_test.reindex(columns=pre_obj['columns'], fill_value=0)

    # 1. Imputación básica (aplica a todos)
    if pre_obj['imputer']:
        X_test = pre_obj['imputer'].transform(X_test)

    # 2. Transformaciones específicas según el modelo ganador
    algoritmo = pre_obj['algoritmo']

    if algoritmo in ["KNN", "Tree"]:
        if pre_obj['scaler']:
            X_test_final = pre_obj['scaler'].transform(X_test)
    elif algoritmo == "CategoricalNB":
        X_test_final = pre_obj['discretizer'].transform(X_test)
    elif algoritmo == "MixedNB":
        X_test_final = X_test
    else:
        X_test_final = X_test

    predicciones = clf.predict(X_test_final)

    df_resultado = pd.read_csv(f_test)
    df_resultado['Clase_Predicha'] = predicciones
    df_resultado.to_csv("predicciones_finales.csv", index=False)
    print("Predicciones guardadas en 'predicciones_finales.csv'")


if __name__ == "__main__":
    test()