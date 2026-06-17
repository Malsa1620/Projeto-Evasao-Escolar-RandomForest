import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Lê os dados
dados = pd.read_csv("evasao.csv")

# Variáveis de entrada
X = dados[["Faltas", "Media", "Participacao"]]

# Resposta correta
y = dados["Evasao"]

# Divide os dados
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Cria o modelo
modelo = RandomForestClassifier()

# Treina o modelo
modelo.fit(X_treino, y_treino)

# Faz previsões
previsoes = modelo.predict(X_teste)

# Calcula acurácia
acuracia = accuracy_score(y_teste, previsoes)

print("Acurácia:", round(acuracia * 100, 2), "%")

# Novos alunos
novos_alunos = [
    [18, 4.5, 1],
    [3, 9.0, 3],
    [12, 6.0, 2]
]

resultados = modelo.predict(novos_alunos)

for i, resultado in enumerate(resultados):
    print(f"Aluno {i+1}: {resultado}")