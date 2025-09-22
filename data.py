import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# URL do dataset HateBR no GitHub
DATASET_URL = 'https://raw.githubusercontent.com/franciellevargas/HateBR/main/dataset/HateBR.csv'

# 1. Carregar o dataset
print("Carregando o dataset...")
try:
    df = pd.read_csv(DATASET_URL)
    print("Dataset carregado com sucesso!")
    # Mostra as primeiras linhas e a distribuição das classes
    print("\nDistribuição das classes:")
    print(df['label_final'].value_counts(normalize=True))
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    exit()

# 2. Pré-processamento e Definição das variáveis
print("\nIniciando pré-processamento...")
# Para este MVP, a única limpeza será converter para minúsculas.
# O TfidfVectorizer já lida com muita coisa.
X = df['comentario'].str.lower()
y = df['label_final']

# 3. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Dados divididos: {len(X_train)} para treino, {len(X_test)} para teste.")

# 4. Vetorização do texto usando TF-IDF
print("Vetorizando o texto...")
vectorizer = TfidfVectorizer(max_features=5000) # Usamos as 5000 palavras mais relevantes

# Aprende o vocabulário com os dados de treino e transforma os dados de treino
X_train_vect = vectorizer.fit_transform(X_train)

# Apenas transforma os dados de teste com o vocabulário já aprendido
X_test_vect = vectorizer.transform(X_test)

# 5. Treinamento do modelo de Regressão Logística
print("Treinando o modelo de classificação...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)
print("Modelo treinado com sucesso!")

# 6. Avaliação do modelo
print("\nAvaliando o modelo nos dados de teste...")
y_pred = model.predict(X_test_vect)

print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Não Odioso', 'Odioso']))

# 7. Salvando o modelo e o vetorizador
print("\nSalvando o modelo e o vetorizador em arquivos...")

joblib.dump(model, 'joblib/modelo_odio.joblib')
joblib.dump(vectorizer, 'joblib/vetorizador_odio.joblib')

print("Modelo e vetorizador salvos com sucesso!")