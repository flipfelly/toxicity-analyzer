import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# URL do dataset HateBR no GitHub
DATASET_URL = 'https://raw.githubusercontent.com/franciellevargas/HateBR/main/dataset/HateBR.csv'

# 1. Carregar o dataset
print("Carregando o dataset...")
try:
    df = pd.read_csv(DATASET_URL)
    print("Dataset carregado com sucesso!")
    # Mostra as primeiras linhas e a distribuição das classes
    print(df.head())
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

# Agora que o modelo está treinado, podemos usá-lo.
# Os objetos que precisamos salvar/usar para novas previsões são:
# - `model` (o classificador)
# - `vectorizer` (o vetorizador)

print("\n--- MVP PRONTO PARA USO ---")

# --- INÍCIO DO SCRIPT MVP ---

def avaliar_toxicidade(comentario: str, model, vectorizer) -> dict:
    """
    Recebe um comentário e retorna a classificação de toxicidade
    e a probabilidade de ser discurso de ódio.
    """
    # 1. Aplicar o mesmo pré-processamento (minúsculas)
    comentario_processado = comentario.lower()
    
    # 2. Vetorizar o comentário usando o vetorizador JÁ TREINADO
    comentario_vect = vectorizer.transform([comentario_processado])
    
    # 3. Fazer a predição
    predicao = model.predict(comentario_vect)
    probabilidades = model.predict_proba(comentario_vect)
    
    # A probabilidade de ser discurso de ódio é a probabilidade da classe "1"
    prob_odio = probabilidades[0][1]
    
    if predicao[0] == 1:
        classificacao = "Discurso de Ódio"
    else:
        classificacao = "Não é Discurso de Ódio"
        
    return {
        "classificacao": classificacao,
        "nivel_toxicidade": f"{prob_odio:.2%}" # Formata como porcentagem
    }

# --- EXEMPLOS DE USO ---
print("\n--- Testando o MVP com novos comentários ---")

# Exemplo 1: Comentário potencialmente tóxico
comentario1 = "Esses políticos são todos uns bandidos, tinham que sumir do mapa!"
resultado1 = avaliar_toxicidade(comentario1, model, vectorizer)
print(f"Comentário: '{comentario1}'")
print(f"Resultado: {resultado1}\n")

# Exemplo 2: Comentário neutro
comentario2 = "O jogo de futebol ontem foi muito emocionante, gostei bastante do resultado."
resultado2 = avaliar_toxicidade(comentario2, model, vectorizer)
print(f"Comentário: '{comentario2}'")
print(f"Resultado: {resultado2}\n")

# Exemplo 3: Comentário inserido pelo usuário
print("Digite um comentário para ser analisado (ou 'sair' para terminar):")
while True:
    meu_comentario = input("> ")
    if meu_comentario.lower() == 'sair':
        break
    resultado = avaliar_toxicidade(meu_comentario, model, vectorizer)
    print(f"Resultado: {resultado}\n")