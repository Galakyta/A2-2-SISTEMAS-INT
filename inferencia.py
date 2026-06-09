from pickle import load
import pandas as pd

modelo_rf = load(open("CARTAO_rf.pkl", "rb"))
modelo_xgb = load(open("CARTAO_xgboost.pkl", "rb"))

# loadando o modelo normalmente 

colunas = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# cuidado pra nn colocar id de novo aq pq a gente dropo antes pra n gerar a hierarquia

# aq da pra substituir por qualquer dado, mas eu gerei esses ai pra fatura pra ficar mais facil
#na teoria com essa quantidade de atraso deve atrasar bem 
novo_cliente = pd.DataFrame([{
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913,
    "BILL_AMT2": 3102,
    "BILL_AMT3": 689,
    "BILL_AMT4": 0,
    "BILL_AMT5": 0,
    "BILL_AMT6": 0,
    "PAY_AMT1": 0,
    "PAY_AMT2": 689,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0
}], columns=colunas)

novo_cliente = pd.get_dummies(novo_cliente, drop_first=True)
#precisa dummar pq a gente dummou no treino, se nn fizer isso nem roda
novo_cliente = novo_cliente.reindex(columns=modelo_rf.feature_names_in_, fill_value=0)
#aq tmbm so reindexando igual a gente fazia 

# Random Forest
pred_rf = modelo_rf.predict(novo_cliente)[0]
proba_rf = modelo_rf.predict_proba(novo_cliente)[0]

print("Resultado Random Forest:")
print(f"Predicao:{'default' if pred_rf == 1 else 'nao default'}")
print(f"Probabilidade nao default (pagar em dia):  {proba_rf[0]:.2%}")
print(f"Probabilidade default (defaultar):  {proba_rf[1]:.2%}")

# XGBoost
pred_xgb = modelo_xgb.predict(novo_cliente)[0]
proba_xgb = modelo_xgb.predict_proba(novo_cliente)[0]

print("\nResultado XGBoost:")
print(f"Predicao:{'default' if pred_xgb == 1 else 'nao default'}")
print(f"Probabilidade nao default (pagar em dia):  {proba_xgb[0]:.2%}")
print(f"Probabilidade default (defaultar):  {proba_xgb[1]:.2%}")

print("\n")
if pred_rf == pred_xgb:
    print(f"ambos previram {'default' if pred_rf == 1 else 'nao default'}.")
else:
    print(f"RF disse {'default' if pred_rf == 1 else 'nao default'}, XGB disse {'default' if pred_xgb == 1 else 'nao default'}.")
    
#entre as duas, xgboost pelos dados aparenta ser bem mais confiavel, levando em conta que arbitrariamente eu coloquei alguem que
#nitidamente nunca iria pagar em dia, o xgboost aparentou ter muito mais certeza