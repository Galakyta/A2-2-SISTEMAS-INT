import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
from imblearn.over_sampling import SMOTE 

# ja vou abrir falando que considerei fortemente usar um xgboost aqui, pro
# treinamento ser mais rapido e simplesmente pq eu gostei dele, mas a base ta meio stranha
# ent achei melhor deixar quieto e n inventar mt, e ir com rf ja q eu ja sei usar melhor

dados = pd.read_csv("default_of_credit_card_clients.csv", sep = ";")

dados_atributos = dados.drop(columns=["default payment next month", "ID"])
target = dados["default payment next month"]



dados_atributos = pd.get_dummies(dados_atributos, drop_first=True)
#detalhe importante q eu esqueci q o smote n tanka dado string ent precisa dummar

#enfim, depois de olhar o csv eu considerei q eu precisaria dropar o id, pra n criar uma hierarquia estranha
# e separar o target, agora o resto me pareceu bem legitimo, entao optei por manter

atributos_treino, atributos_teste, target_treino, target_teste = train_test_split(
    dados_atributos, target, test_size=0.3, random_state=42
)

#sobre o split eu usei a proporcao de 70 pra 30 % que a gente usou em sala msm
#pois é o q eu me acostumei a usar e achei um ponto bom em relacao ao segundo mais usado
# pelo q eu vi de ser o 20 80

smoter = SMOTE(random_state=42) 

atributos_treino_smotados, target_treino_smotado = smoter.fit_resample(atributos_treino, target_treino)


#como sempre smotar só depois do split se n da um data leakeage lascado se aplicar no modelo todo


grade_de_parametros = {
    "n_estimators": [int(x) for x in np.linspace(start=100, stop=1000, num=10)],
    "criterion": ["gini", "entropy"],
    "min_samples_split": [int(x) for x in np.linspace(start=2, stop=10, num=2)],
    "max_depth": [int(x) for x in np.linspace(start=10, stop=100, num=20)],
    "max_features": ["sqrt", "log2"]
}

#sobre os parametros, como é um arquivo grandinho de 30k linhas (grandinho pra mim pelo menos)
#eu usei os que eu tenho usado de forma mais geral ultimamente, pro pimas eu acho que até ficou meio estourado
# o numero de arvores, mas aqui acho q ta na media


AraucariasRF = RandomForestClassifier(random_state=42, n_jobs=1)

#me acostumei a usar esse nome de floresta em tudo ent deixei

random_search = RandomizedSearchCV(
    AraucariasRF,
    param_distributions=grade_de_parametros,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# pois bem, aqui eu ja diferenciei pq eu fiquei com trauma da ultima prova em relacao ao tempo 
# de treinamento, ent eu comecei a usar o random search q funciona mt bem e mt mais rapido do que o grid
# e como eu to em um ambente onde o tempo é sensivel achei justificiavel 

#tem um outro detalhe que eu vi que mt gente usa f1 ou accuracy no random search, mas como a gente só usou accuracy preferi manter normal tmbm

random_search.fit(atributos_treino_smotados, target_treino_smotado)

print("melhor parametro:", random_search.best_params_)
print("melhor score", random_search.best_score_)

#aq eu faço um ultimo treinamento final do modelo com o time pq eu tenho gostado de medir
# nos meus exercicios eu fiquei muito focado em diminuir ao maximo o tempo de treino pra brincar com o gradient

melhores_parametros = random_search.best_params_
start_time = time.time()
rf_final = RandomForestClassifier(**melhores_parametros, random_state=42, n_jobs=-1)
rf_final.fit(atributos_treino_smotados, target_treino_smotado)
tempo_treino = time.time() - start_time

print("tempo de treino com os mior parametro:  " + str(tempo_treino) + " segundos")

pred_teste = rf_final.predict(atributos_teste)
print("acuracia:", accuracy_score(target_teste, pred_teste))
print("matriz de confusao:\n", confusion_matrix(target_teste, pred_teste))
print("relatorio de classificaçao:\n", classification_report(target_teste, pred_teste, target_names=["nao default", "default"]))

from pickle import dump # esqueci o pickle 

dump(rf_final, open("CARTAO_rf.pkl", "wb"))


#no final saiu isso aq, e meu deus do ceu demorou mt esse segundos ai ta errado foram 3 minutos e 45, tudo bem um i5 a 4,5ghz nn é la
# uma workstation mas xesus amado, pra mim isso em da dimensao daquela parada q vc falou q demorou seculos pra treinar o csv da huwawei em uma maquina de 150k

'''PS C:\Users\galak\OneDrive\Desktop\A2-2-SISTEMAS-INT> & C:/Users/galak/AppData/Local/Python/pythoncore-3.14-64/python.exe c:/Users/galak/OneDrive/Desktop/A2-2-SISTEMAS-INT/CARTAO_rf.py
Fitting 5 folds for each of 20 candidates, totalling 100 fits
melhor parametro: {'n_estimators': 400, 'min_samples_split': 2, 'max_features': 'log2', 'max_depth': 43, 'criterion': 'entropy'}
melhor score 0.8616490920353697
tempo de treino com os mior parametro:  3.4595658779144287 segundos
acuracia: 0.7968888888888889
matriz de confusao:
 [[6305  735]
 [1093  867]]
relatorio de classificaçao:
               precision    recall  f1-score   support

 nao default       0.85      0.90      0.87      7040
     default       0.54      0.44      0.49      1960

    accuracy                           0.80      9000
   macro avg       0.70      0.67      0.68      9000
weighted avg       0.78      0.80      0.79      9000

PS C:\Users\galak\OneDrive\Desktop\A2-2-SISTEMAS-INT> '''
