import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Coleta de Dados
# Simulação de dados de humor (em uma escala de 1 a 10) ao longo de 100 dias
np.random.seed(42)
dias = np.arange(1, 101)
humor = np.random.normal(loc=5, scale=2, size=100)  # Média 5, desvio padrão 2
humor = np.clip(humor, 1, 10)  # Limitar os valores entre 1 e 10

df = pd.DataFrame({'Dia': dias, 'Humor': humor})

# 2. Análise Estatística
stats_descriptive = df['Humor'].describe()

print("Análise Estatística Descritiva:")
print(stats_descriptive)

# 3. Visualização de Dados
plt.figure(figsize=(12, 6))
plt.plot(df['Dia'], df['Humor'], marker='o', linestyle='-', color='b')
plt.title('Variação do Humor ao Longo do Tempo')
plt.xlabel('Dia')
plt.ylabel('Nível de Humor')
plt.ylim(1, 10)
plt.grid(True)
plt.show()

# Histograma do humor
plt.figure(figsize=(12, 6))
plt.hist(df['Humor'], bins=10, color='c', edgecolor='black')
plt.title('Distribuição dos Níveis de Humor')
plt.xlabel('Nível de Humor')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# 4. Previsão de Humor Futuro
# Preparar os dados
X = df[['Dia']]
y = df['Humor']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Erro Quadrático Médio (MSE) do modelo: {mse:.2f}")

# Visualização das previsões
plt.figure(figsize=(12, 6))
plt.plot(df['Dia'], df['Humor'], marker='o', linestyle='-', color='b', label='Dados Reais')
plt.scatter(X_test, y_pred, color='r', label='Previsões')
plt.title('Previsão do Nível de Humor')
plt.xlabel('Dia')
plt.ylabel('Nível de Humor')
plt.ylim(1, 10)
plt.legend()
plt.grid(True)
plt.show()

# Função para prever o humor futuro
def prever_humor(dia_futuro):
    return model.predict([[dia_futuro]])[0]

# Previsão para o dia 101
humor_dia_101 = prever_humor(101)
print(f"Previsão do nível de humor para o dia 101: {humor_dia_101:.2f}")

# Relatório
relatorio = f"""
Relatório de Análise de Humor

1. Análise Estatística Descritiva:
{stats_descriptive}

2. Avaliação do Modelo de Regressão Linear:
- Erro Quadrático Médio (MSE): {mse:.2f}

3. Previsão de Humor Futuro:
- Previsão do nível de humor para o dia 101: {humor_dia_101:.2f}

Conclusão:
Através da análise dos dados de humor ao longo de 100 dias, foi possível identificar padrões e prever o nível de humor futuro com um modelo de regressão linear. Esse tipo de análise pode ser utilizado no HAP para monitorar e prever o estado emocional dos usuários, auxiliando na criação de planos de intervenção personalizados.

Relação com outras disciplinas:
- Design de Interfaces: A visualização dos dados e a apresentação dos resultados de forma clara e intuitiva são essenciais para uma boa experiência do usuário. Gráficos interativos e interfaces amigáveis podem ajudar tanto os pacientes quanto os profissionais de saúde a entenderem melhor os dados e tomarem decisões informadas.
- Inovação em Tecnologia da Informação: A utilização de técnicas de machine learning, como a regressão linear, exemplifica como a inovação tecnológica pode ser aplicada na área de saúde mental. Ao integrar essas tecnologias na plataforma HAP, podemos fornecer insights valiosos e personalizados para os usuários, promovendo um cuidado mais eficaz e acessível.
"""

print(relatorio)
