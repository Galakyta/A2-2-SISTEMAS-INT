from pickle import load
import pandas as pd

modelo = load(open("CARTAO_rf.pkl", "rb"))

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

novo_cliente = novo_cliente.reindex(columns=modelo.feature_names_in_, fill_value=0)

pred = modelo.predict(novo_cliente)[0]
proba = modelo.predict_proba(novo_cliente)[0]

print(f"Predicao:{'default' if pred == 1 else 'nao default'}")
print(f"Probabilidade nao default (pagar em dia):  {proba[0]:.2%}")
print(f"Probabilidade default (defaultar):  {proba[1]:.2%}") #atrasar e nunca mais sair do cheque especial

# Predicao:default
#Probabilidade nao default (pagar em dia):  11.00%
#Probabilidade default (defaultar):  89.00%
#PS C:\Users\galak\OneDrive\Desktop\A2-2-SISTEMAS-INT>'''