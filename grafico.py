import pandas as pd
import matplotlib.pyplot as plt

dados = pd.read_csv("evasao.csv")

dados["Evasao"].value_counts().plot(kind="bar")

plt.title("Quantidade de alunos com e sem evasão")
plt.xlabel("Evasão")
plt.ylabel("Quantidade")

plt.show()