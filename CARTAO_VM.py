import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from pickle import dump

# Carregando os dados
dados = pd.read_csv("default_of_credit_card_clients.csv", sep=";")

dados_atributos = dados.drop(columns=["default payment next month", "ID"])
target = dados["default payment next month"]

# pro xgboost precisa one hotar 
dados_atributos = pd.get_dummies(dados_atributos, drop_first=True)

#splitando
atributos_treino, atributos_teste, target_treino, target_teste = train_test_split(
    dados_atributos, target, test_size=0.3, random_state=42
)

#balanceamento especialmente e bom pq o extreme gradient e mais sensivel a dado desbalanceado
smoter = SMOTE(random_state=42)
atributos_treino_smotados, target_treino_smotado = smoter.fit_resample(atributos_treino, target_treino)

#essa grade eu tenho usado pra praticamente tudo em xgboost
grade_de_parametros = {
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=500, num=5)],
    "max_depth": [int(x) for x in np.linspace(start=3, stop=15, num=5)],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5]
}

# Modelo XGBoost base
xgb_model = xgb.XGBClassifier(
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',  # xgb precisa dessa metrica
    use_label_encoder=False  #isso e um bo com a versao mas eu prefiro usar pra n chover erro
)

# Randomized Search
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=grade_de_parametros,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)


random_search.fit(atributos_treino_smotados, target_treino_smotado)

print("\nmelhor parametro:", random_search.best_params_)
print("melhor score (validação cruzada):", random_search.best_score_)

#melhor treinamento como antes
melhores_parametros = random_search.best_params_
start_time = time.time()
xgb_final = xgb.XGBClassifier(
    **melhores_parametros,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
)
xgb_final.fit(atributos_treino_smotados, target_treino_smotado)
tempo_treino = time.time() - start_time

print(f"\ntempo de treino com os melhores parâmetros: {tempo_treino:.2f} segundos")

# Predição e avali
pred_teste = xgb_final.predict(atributos_teste)
print("acuracia:", accuracy_score(target_teste, pred_teste))
print("matriz de confusao:", confusion_matrix(target_teste, pred_teste))
print("relatorio de classificaçao:", classification_report(target_teste, pred_teste, target_names=["nao default", "default"]))

# Salvando o model
dump(xgb_final, open("CARTAO_xgboost.pkl", "wb"))
