# Guia Completo — Machine Learning & IA com Python
> Material de revisão baseado nos exemplos do curso. Cobre fundamentos de Python, bibliotecas, algoritmos de ML e Deep Learning com explicação linha a linha dos códigos.

---

## Sumário
1. [Por que Python para IA?](#1-por-que-python-para-ia)
2. [Bibliotecas essenciais](#2-bibliotecas-essenciais)
3. [Fundamentos de Python para ML](#3-fundamentos-de-python-para-ml)
4. [Regressão Linear](#4-regressão-linear)
5. [K-Nearest Neighbors (KNN)](#5-k-nearest-neighbors-knn)
6. [SVM — Support Vector Machine](#6-svm--support-vector-machine)
7. [K-Means — Agrupamento (Clustering)](#7-k-means--agrupamento-clustering)
8. [Random Forest](#8-random-forest)
9. [Redes Neurais com Keras/TensorFlow](#9-redes-neurais-com-kerastensorflow)
10. [Redes Neurais com PyTorch](#10-redes-neurais-com-pytorch)
11. [Detecção de Fraude — os 5 conceitos juntos](#11-detecção-de-fraude--os-5-conceitos-juntos)
12. [MNIST — Reconhecimento de Dígitos](#12-mnist--reconhecimento-de-dígitos)
13. [Iris — Classificação Multiclasse](#13-iris--classificação-multiclasse)
14. [Previsão de Ações com Random Forest](#14-previsão-de-ações-com-random-forest)
15. [LLMs com Ollama e LangChain](#15-llms-com-ollama-e-langchain)
16. [API REST com FastAPI](#16-api-rest-com-fastapi)
17. [Pré-processamento de Dados](#17-pré-processamento-de-dados)
18. [Otimização de Hiperparâmetros](#18-otimização-de-hiperparâmetros)
19. [Hugging Face — Modelos Pré-treinados](#19-hugging-face--modelos-pré-treinados)
20. [Glossário rápido](#20-glossário-rápido)

### PARTE II — Curso de IA & Machine Learning (Aulas Práticas)
21. [Aula 1 — Conceitos Básicos e Aplicações de ML](#21-aula-1--conceitos-básicos-e-aplicações-de-ml)
22. [Aula 2 — Regressão Linear Simples (Sorvete)](#22-aula-2--regressão-linear-simples-sorvete)
23. [Aula 2 — Regressão Linear Múltipla (Casas da Califórnia, projeto end-to-end)](#23-aula-2--regressão-linear-múltipla-casas-da-califórnia-projeto-end-to-end)
24. [Aula 3 — PCA (Análise de Componentes Principais)](#24-aula-3--pca-análise-de-componentes-principais)
25. [Aula 4 — Feature Scaling (Normalização e Padronização)](#25-aula-4--feature-scaling-normalização-e-padronização)
26. [Desafio — Insurance (Regressão de custos de seguro)](#26-desafio--insurance-regressão-de-custos-de-seguro)

### PARTE III — Machine Learning Avançado (Pós Tech DTAT — Classificação)
27. [Aula 1 — Introdução à Classificação + Pré-processamento de Categóricas](#27-aula-1--introdução-à-classificação--pré-processamento-de-categóricas)
28. [Aula 2 — Classificação na Prática (Fraude no Cartão + Recrutamento)](#28-aula-2--classificação-na-prática-fraude-no-cartão--recrutamento)
29. [Aula 3 — Aprendizado Não Supervisionado (Clusterização + Segmentação de Imagens)](#29-aula-3--aprendizado-não-supervisionado-clusterização--segmentação-de-imagens)
30. [Aula 4 — Modelos Baseados em Árvores (Decision Tree, Random Forest, SMOTE)](#30-aula-4--modelos-baseados-em-árvores-decision-tree-random-forest-smote)
31. [Aula 5 — Validação Cruzada e Busca de Hiperparâmetros](#31-aula-5--validação-cruzada-e-busca-de-hiperparâmetros)
32. [Aula 6 — Avaliação: Matriz de Confusão e Classification Report](#32-aula-6--avaliação-matriz-de-confusão-e-classification-report)
33. [Aula 7 — Curva ROC e AUC](#33-aula-7--curva-roc-e-auc)
34. [Desafio — HR Analytics (previsão de abandono/churn)](#34-desafio--hr-analytics-previsão-de-abandonochurn)

### PARTE IV — Visão Computacional (Computer Vision)
35. [Aula 1 — Fundamentos de Visão Computacional com OpenCV](#35-aula-1--fundamentos-de-visão-computacional-com-opencv)
36. [Aula 2 — OCR (Reconhecimento de Texto) com Tesseract e PaddleOCR](#36-aula-2--ocr-reconhecimento-de-texto-com-tesseract-e-paddleocr)
37. [Aula 3 — Detecção de Faces com Haar Cascades](#37-aula-3--detecção-de-faces-com-haar-cascades)
38. [Aula 4 — Redes Neurais Convolucionais (CNN)](#38-aula-4--redes-neurais-convolucionais-cnn)
39. [Aula 5 — Detecção de Objetos com YOLOv5](#39-aula-5--detecção-de-objetos-com-yolov5)
40. [Aula 6 — GANs (Redes Generativas Adversárias)](#40-aula-6--gans-redes-generativas-adversárias)
41. [Projetos com MediaPipe — Hand Tracking (Libras) e Análise de Agachamento](#41-projetos-com-mediapipe--hand-tracking-libras-e-análise-de-agachamento)
42. [Rastreamento de Objetos (Object Tracking) — KCF, CSRT e a família do OpenCV](#42-rastreamento-de-objetos-object-tracking--kcf-csrt-e-a-família-do-opencv)
43. [Aprofundando: Classificador em Cascata e Haar Cascade](#43-aprofundando-classificador-em-cascata-e-haar-cascade)
44. [Glossário da Parte IV (Visão Computacional)](#44-glossário-da-parte-iv-visão-computacional)

---

## 1. Por que Python para IA?

Python domina o mundo de Machine Learning e IA por razões práticas:

| Motivo | Explicação |
|--------|-----------|
| **Sintaxe simples** | Código legível, próximo do inglês. Foco no problema, não na linguagem. |
| **Ecossistema rico** | NumPy, Pandas, scikit-learn, TensorFlow, PyTorch — todas as grandes ferramentas têm Python como primeira linguagem. |
| **Comunidade enorme** | Qualquer dúvida tem resposta. Qualquer algoritmo tem implementação. |
| **Interatividade** | Jupyter Notebooks permitem rodar e visualizar resultados célula a célula — ideal para exploração de dados. |
| **Integração fácil** | Conecta com APIs, bancos de dados, arquivos CSV, JSON, imagens — tudo com poucas linhas. |

**Analogia:** Python é como uma cozinha industrial bem equipada. Você não precisa fabricar as ferramentas, só usá-las.

---

## 2. Bibliotecas Essenciais

### NumPy — Computação Numérica

```python
import numpy as np

data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

mean = np.mean(data)  # média de todos os elementos = 5.0
print(mean)
```

**O que é:** NumPy dá ao Python vetores e matrizes de alta performance.  
**Por que importa:** Toda entrada de dados para modelos de ML é uma matriz (array). Sem NumPy, seria lento e verboso.  
**Analogia:** NumPy é a planilha do Python — mas muito mais rápida que o Excel.

---

### Pandas — Manipulação de Dados

```python
import pandas as pd

data = pd.read_csv('data.csv')         # lê um arquivo CSV
mean_value = data['column_name'].mean() # calcula a média de uma coluna
print(mean_value)
```

**O que é:** Pandas trabalha com tabelas (DataFrames), semelhante ao Excel.  
**Operações comuns:** filtrar linhas, calcular médias, agrupar dados, tratar valores nulos.  
**Analogia:** Se NumPy é a planilha crua, Pandas é o Excel com fórmulas automáticas.

---

### Matplotlib e Seaborn — Visualização

```python
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y)   # desenha linha conectando os pontos
plt.show()       # exibe o gráfico
```

```python
import seaborn as sns
import matplotlib.pyplot as plt

data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
sns.histplot(data)   # histograma: conta quantas vezes cada valor aparece
plt.show()
```

**Diferença:** Matplotlib é mais manual (controle total). Seaborn é mais automático e bonito para estatística.  
**Analogia:** Matplotlib é o Paint; Seaborn é o Canva.

---

### SciPy — Ciência e Matemática

```python
from scipy.integrate import solve_ivp

def dydt(t, y):
    return -0.5 * y   # equação diferencial: taxa de variação = -0.5 * y

solution = solve_ivp(dydt, [0, 10], [2])  # resolve de t=0 a t=10, condição inicial y=2
print(solution.y)
```

**O que é:** Resolve equações diferenciais, integrais, otimizações, estatísticas avançadas.  
**Na prática:** Usado em simulações físicas, modelagem científica.

---

## 3. Fundamentos de Python para ML

### Operações básicas com listas

```python
numeros = [10, 20, 30, 40, 50]

soma       = sum(numeros)          # 150
quantidade = len(numeros)          # 5
media      = soma / quantidade     # 30.0

print(f"Soma: {soma}")
print(f"Quantidade: {quantidade}")
print(f"Média: {media}")
```

**Por que importa para ML:** Antes de usar bibliotecas, entender que dados são listas de números é fundamental. O modelo não vê "maçã" — ele vê `[150, 7]`.

---

### Funções

```python
def saudacao(nome):           # define a função
    return f"Olá, {nome}!"   # retorna um valor

resultado = saudacao("Gabi") # chama a função
print(resultado)             # "Olá, Gabi!"
```

---

### Tratamento de Erros

```python
try:
    resultado = 10 / 0          # causa ZeroDivisionError
except ZeroDivisionError:
    print("Não pode dividir por zero!")
except TypeError:
    print("Tipo errado!")
```

**Por que importa para ML:** Dados do mundo real têm erros, valores ausentes e tipos errados. Saber tratar exceções evita que o programa quebre.

---

## 4. Regressão Linear

### Conceito

Regressão linear encontra a **melhor linha reta** que passa pelos dados.  
Fórmula: `y = a·x + b`  
- `a` = inclinação (quanto y muda por unidade de x)  
- `b` = intercepto (valor de y quando x = 0)

**Analogia:** Imagine marcar pontos num papel e depois desenhar a régua que mais se aproxima de todos eles. Isso é regressão linear.

**Quando usar:** Prever valores contínuos — preços, temperaturas, salários.

---

### Exemplo 1 — Preço de casas (`regressao-linear.py`)

```python
from sklearn.linear_model import LinearRegression

# Dados: tamanho da casa (m²) -> preço (mil R$)
X = [[50], [70], [90], [110], [130]]   # entrada: cada sublist é uma amostra
y = [200, 260, 330, 390, 450]          # saída esperada

modelo = LinearRegression()
modelo.fit(X, y)   # FASE DE TREINO: encontra o melhor 'a' e 'b'

print("Inclinação (a):", modelo.coef_[0])      # ~ preço por m²
print("Intercepto (b):", modelo.intercept_)    # valor base da casa

print("Previsão 100m²:", modelo.predict([[100]]))
```

**Linha a linha:**
- `X = [[50], ...]` — cada casa é representada como uma lista com 1 característica (m²). O formato `[[...]]` é obrigatório no scikit-learn: linhas = amostras, colunas = features.
- `modelo.fit(X, y)` — o algoritmo encontra matematicamente a inclinação e o intercepto que minimizam o erro quadrático médio.
- `modelo.coef_` — a inclinação. Se for ~2.5, cada m² adicional vale R$ 2.500.
- `modelo.predict([[100]])` — prevê o preço de uma casa de 100m² que nunca foi vista.

---

### Exemplo 2 — Múltiplas variáveis (`index7.py`)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 1],   # 4 amostras com 2 características cada
              [1, 2],
              [2, 2],
              [2, 3]])

# y = 1*x1 + 2*x2 + 3  (relação linear conhecida)
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression().fit(X, y)
print(model.coef_)   # deve retornar [1, 2] — achou os coeficientes reais!
```

**Detalhe importante:** `np.dot(X, [1,2])` faz multiplicação matricial — para cada linha de X, calcula `1*x1 + 2*x2`. O resultado é um exemplo sintético onde sabemos a resposta certa, ideal para validar que o modelo está funcionando.

---

### Métricas de Regressão

No notebook `ML_1_REGRESSAO.ipynb` (dataset com 5000 casas):
- **R² = 0.889** → o modelo explica 88.9% da variação nos preços
- **MAE = 93.564** → erro médio absoluto de ~R$ 93.000
- **MAPE = 8.58%** → erro médio percentual de 8.58%

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| R² | 1 - SS_res/SS_tot | Quanto da variação o modelo explica (0 a 1) |
| MAE | média de \|y_real - y_pred\| | Erro médio em unidades reais |
| MAPE | média de \|y_real - y_pred\| / y_real | Erro em percentual |

---

## 5. K-Nearest Neighbors (KNN)

### Conceito

KNN classifica um ponto novo olhando para os **K vizinhos mais próximos** e votando.  
Se K=3, os 3 vizinhos mais próximos "votam" na classe do novo ponto.

**Analogia:** Você se mudou para uma cidade nova e quer saber se o bairro é residencial ou comercial. Você olha para as 3 casas mais próximas e vê o que elas são.

**Quando usar:** Classificação quando os dados têm estrutura espacial clara; bom para protótipos.

---

### Exemplo — Classificar frutas (`k-nn.py`)

```python
from sklearn.neighbors import KNeighborsClassifier

# Dados de treino: [peso (g), doçura (1-10)]
X = [
    [150, 7], [170, 6], [140, 8],   # maçãs: pesadas e doces
    [130, 4], [120, 3], [160, 5],   # laranjas: mais leves e menos doces
]
y = ["maçã", "maçã", "maçã", "laranja", "laranja", "laranja"]

modelo = KNeighborsClassifier(n_neighbors=3)  # usa os 3 vizinhos mais próximos
modelo.fit(X, y)   # "treino" = apenas memoriza os dados (sem cálculo!)

fruta_nova = [[155, 7]]                        # nova fruta: 155g, doçura 7
print(modelo.predict(fruta_nova))              # -> ['maçã']
print(modelo.predict_proba(fruta_nova))        # probabilidade por classe
```

**Linha a linha:**
- `n_neighbors=3` — ao classificar, olha os 3 pontos mais próximos no espaço de features.
- `modelo.fit(X, y)` — aqui o KNN não "aprende" nada matematicamente; apenas guarda os dados.
- Na predição, calcula a distância euclidiana entre `[155, 7]` e todos os pontos, pega os 3 mais próximos e vota.
- `predict_proba` — retorna `[prob_laranja, prob_maçã]`. Se 2 dos 3 vizinhos são maçã, retorna 0.67 para maçã.

**Vantagem:** Simples de entender. **Desvantagem:** Lento com muitos dados (compara com todos).

---

## 6. SVM — Support Vector Machine

### Conceito

SVM encontra o **hiperplano** (linha em 2D, plano em 3D) que melhor **separa as classes** com a maior margem possível.

**Analogia:** Imagine duas turmas de alunos sentadas em uma sala. O SVM coloca uma mesa no meio da sala, posicionada de forma que haja o máximo de espaço possível dos dois lados.

**Quando usar:** Classificação binária, dados com muitas features, textos.

---

### Exemplo 1 — Solubilidade (`prev.py`)

```python
from sklearn.svm import LinearSVC

# Cada composto é representado por 3 características binárias
composto1 = [1, 1, 1]   # tem todas as 3 propriedades
composto2 = [0, 0, 0]   # não tem nenhuma

dados_treino = [composto1, composto2, composto3, composto4, composto5, composto6]
rotulos_treino = ['S', 'N', 'S', 'N', 'S', 'S']  # S=Solúvel, N=Não Solúvel

modelo = LinearSVC()
modelo.fit(dados_treino, rotulos_treino)

# Novos compostos para classificar
dados_teste = [[1, 0, 0], [0, 1, 1], [1, 0, 1]]
previsoes = modelo.predict(dados_teste)

mapeamento = {'S': 'Solúvel', 'N': 'Não Solúvel'}
for i, previsao in enumerate(previsoes):
    print(f"Teste {i+1}: {mapeamento[previsao]}")
```

---

### Exemplo 2 — Com Avaliação de Acurácia (`solubility.py`)

```python
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ... (mesmo setup acima) ...
rotulos_teste = ['S', 'N', 'S']    # respostas corretas conhecidas
previsoes = modelo.predict(dados_teste)

print(f"Taxa de acerto: {accuracy_score(rotulos_teste, previsoes) * 100:.2f}%")
```

**Diferença entre os dois arquivos:** `solubility.py` adiciona a avaliação com `accuracy_score`. Isso é o ciclo completo: treinar → prever → avaliar.

---

### Ações com SVM (`exemplo.py` — aula2)

```python
from sklearn.svm import LinearSVC

# Features: [preço, P/L, crescimento_receita, margem_lucro, volume]
X = [
    [150.0, 25.0, 0.1, 0.2, 1000000],   # AAPL — subiu
    [2800.0, 30.0, 0.15, 0.25, 500000],  # GOOGL — subiu
    # ...
]
y = [1, 1, 1, 0, 0, 0]   # 1=subiu, 0=caiu

modelo = LinearSVC()
modelo.fit(X, y)
```

**Por que é interessante:** Mostra como dados financeiros viram vetores numéricos — o modelo não sabe que é ação, só vê números.

---

## 7. K-Means — Agrupamento (Clustering)

### Conceito

K-Means **divide dados em K grupos** sem saber previamente as classes.  
Processo:
1. Coloca K centros aleatórios
2. Atribui cada ponto ao centro mais próximo
3. Recalcula os centros como a média dos pontos do grupo
4. Repete até estabilizar

**Analogia:** Você tem 300 clientes espalhados num mapa. Quer abrir 3 lojas. O K-Means encontra as 3 posições que minimizam a distância total dos clientes até a loja mais próxima.

**Quando usar:** Segmentação de clientes, compressão de imagens, detecção de anomalias. **Não tem rótulos (y)!**

---

### Exemplo — Clustering (`mod-agrup-clustering.py`)

```python
from sklearn.cluster import KMeans
import numpy as np

# Repare: NÃO existe um 'y' com rótulos. Só os dados.
X = np.array([
    [170, 120], [180, 90], [150, 140],   # grupo natural 1
    [440, 150], [420, 180], [460, 130],  # grupo natural 2
    [300, 320], [280, 340], [320, 300],  # grupo natural 3
])

modelo = KMeans(n_clusters=3, random_state=42)
modelo.fit(X)

print(modelo.labels_)            # qual grupo cada ponto foi parar: [0, 0, 0, 1, 1, 1, 2, 2, 2]
print(modelo.cluster_centers_)  # posição dos 3 centros encontrados

print(modelo.predict([[430, 160]]))   # novo ponto — qual grupo?
```

**Linha a linha:**
- `n_clusters=3` — você define quantos grupos quer (o algoritmo não sabe).
- `random_state=42` — garante que os centros iniciais sejam os mesmos toda vez (reprodutibilidade).
- `modelo.labels_` — array com o grupo de cada ponto (0, 1 ou 2).
- `modelo.cluster_centers_` — coordenadas dos centros finais após convergência.

**Diferença do KNN:** KNN é supervisionado (tem rótulos). K-Means é **não supervisionado** (descobre padrões sozinho).

---

## 8. Random Forest

### Conceito

Random Forest cria **várias árvores de decisão** (uma "floresta") e combina os votos de todas.  
Cada árvore vê uma amostra aleatória dos dados e features — isso reduz o overfitting.

**Analogia:** Você pergunta a opinião de 100 especialistas diferentes sobre o mesmo problema. A resposta final é a que a maioria escolheu. Um especialista pode errar; 100 erram junto com muito menos frequência.

**Quando usar:** Dados tabulares com muitas features, quando precisa de boa performance sem muito ajuste.

---

### Exemplo — Previsão de Preço de Ações (`exemplo1/app.py`)

```python
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_model(ticker):
    # 1. Baixa dados históricos da Apple de fev a jul 2024
    df = yf.download(ticker, start="2024-02-20", end="2024-07-12")

    # 2. Seleciona só o preço de fechamento
    df = df[['Close']].reset_index()
    df.columns = ['data', 'preco_fechamento']

    # 3. Converte data em número (dias desde 1970) — modelos não entendem datas
    df['data_ordinal'] = df['data'].map(pd.Timestamp.toordinal)

    X = df[['data_ordinal']]        # feature: data como número
    y = df['preco_fechamento']      # target: preço

    # 4. Divide: 80% treino, 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Treina a floresta com 100 árvores
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Prevê e avalia
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Root Mean Squared Error: {rmse}")

ticker = 'AAPL'
train_model(ticker)
```

**Conceitos-chave:**
- `train_test_split(test_size=0.2)` — 20% dos dados são reservados para testar; o modelo nunca os vê durante o treino.
- `n_estimators=100` — 100 árvores de decisão formam a floresta.
- **RMSE** — Raiz do Erro Quadrático Médio. Se RMSE=5, em média o modelo erra $5 no preço.
- `pd.Timestamp.toordinal` — transforma `2024-02-20` em um inteiro (ex: 738937). Necessário porque datas não são números.

---

## 9. Redes Neurais com Keras/TensorFlow

### Conceito

Redes neurais são inspiradas no cérebro: **neurônios artificiais** recebem entradas, aplicam uma função e passam o resultado adiante.  

Componentes:
- **Camada de entrada** — recebe os dados
- **Camadas ocultas** — extraem padrões progressivamente
- **Camada de saída** — gera a previsão
- **Função de ativação** — decide se o neurônio "dispara" (ReLU, sigmoid, softmax)
- **Épocas** — quantas vezes o modelo vê todos os dados durante o treino

**Analogia:** Imagine uma fábrica com várias esteiras de processamento. Cada esteira (camada) refina o produto um pouco mais. No final, a última esteira entrega o produto acabado (a previsão).

---

### API Funcional do Keras (`index8.py`)

```python
import tensorflow as tf
import numpy as np

# Dados sintéticos: 100 amostras, 3 features cada
X = np.random.random((100, 3))
y = np.random.random((100, 1))

# Definindo a arquitetura com a API Funcional
inputs = tf.keras.Input(shape=(3,))           # camada de entrada: 3 features
x = tf.keras.layers.Dense(10, activation='relu')(inputs)  # camada oculta: 10 neurônios
outputs = tf.keras.layers.Dense(1)(x)         # camada de saída: 1 valor (regressão)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=5)
```

**Por que API Funcional?** Permite criar arquiteturas complexas com múltiplas entradas/saídas e bifurcações (ex: redes siamesas).

---

### API Sequential do Keras — Classificação Binária (`index9.py`)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.random.random((100, 8))       # 100 amostras, 8 features
y = np.random.randint(2, size=(100, 1))  # 0 ou 1 (binário)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))   # camada 1: 8->12
model.add(Dense(8, activation='relu'))                  # camada 2: 12->8
model.add(Dense(1, activation='sigmoid'))               # saída: probabilidade 0-1

model.compile(
    loss='binary_crossentropy',   # função de perda para classificação binária
    optimizer='adam',             # otimizador que adapta o learning rate
    metrics=['accuracy']          # métricas acompanhadas durante o treino
)

model.fit(X, y, epochs=150, batch_size=10)
```

**Funções de ativação explicadas:**
| Ativação | Fórmula | Quando usar |
|----------|---------|-------------|
| **ReLU** | max(0, x) | Camadas ocultas — elimina valores negativos |
| **Sigmoid** | 1/(1+e⁻ˣ) | Saída binária (0 ou 1) — dá probabilidade |
| **Softmax** | eˣⁱ/Σeˣʲ | Saída multiclasse — soma das probs = 1 |

**Funções de perda:**
| Loss | Quando usar |
|------|-------------|
| `mean_squared_error` | Regressão |
| `binary_crossentropy` | Classificação binária |
| `categorical_crossentropy` | Classificação multiclasse (one-hot) |

---

## 10. Redes Neurais com PyTorch

### Conceito

PyTorch é mais "manual" que Keras — você define o loop de treino explicitamente.  
Isso dá mais controle e é preferido em pesquisa.

**Analogia:** Keras é o micro-ondas (aperta um botão e está pronto). PyTorch é o fogão (você controla o fogo, a panela, o tempo).

---

### Exemplo Simples — Regressão (`exemplo3/app.py`)

```python
import torch
import torch.optim as optim

# Dados reais conhecidos
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[2.5], [3.5], [5.5], [6.5]])

# Define a arquitetura
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear = torch.nn.Linear(1, 1)   # 1 entrada -> 1 saída

    def forward(self, x):
        return self.linear(x)    # passagem para frente (forward pass)

model = SimpleNN()
loss_function = torch.nn.MSELoss()            # erro quadrático médio
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Gradiente Descendente

# Loop de treino (manual!)
for epoch in range(100):
    y_pred = model(x_data)            # forward: calcula previsão
    loss = loss_function(y_pred, y_data)  # calcula o erro
    
    optimizer.zero_grad()   # zera os gradientes do passo anterior
    loss.backward()          # backward: calcula gradientes pelo erro
    optimizer.step()         # atualiza os pesos

x_test = torch.tensor([[5.0]])
print(f"Previsão para x=5.0: {model(x_test).item()}")
```

**O ciclo de treino em PyTorch (sempre o mesmo):**
1. `forward` — calcula a saída
2. `loss` — mede o erro
3. `zero_grad()` — limpa gradientes antigos
4. `backward()` — calcula novos gradientes (backpropagation)
5. `step()` — atualiza os pesos

---

### Rede Neural Maior — 3 camadas (`index10.py`)

```python
import torch
import torch.nn as nn
import torch.optim as optim

X = torch.randn(100, 3)   # 100 amostras, 3 features
y = torch.randn(100, 1)   # 100 targets

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)   # 3 -> 10 neurônios
        self.fc2 = nn.Linear(10, 1)   # 10 -> 1 saída

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # ativa com ReLU após primeira camada
        x = self.fc2(x)               # saída sem ativação (regressão)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
```

---

## 11. Detecção de Fraude — os 5 conceitos juntos

Este é o exemplo mais rico do repositório. Usa **apenas a biblioteca padrão do Python** para mostrar os algoritmos por dentro.

### Os 5 conceitos

```
FASE 1 → Feature Extraction  : transforma transação em números
FASE 2 → Desvio Padrão       : mede o quanto foge do padrão
FASE 3 → Distribuição Normal  : diz quão rara é essa fuga
FASE 4 → Correlação           : quais pistas andam junto com fraude
FASE 5 → Naive Bayes          : combina tudo numa probabilidade final
```

---

### FASE 1 — Feature Extraction

```python
NOMES_FEATURES = ["valor", "hora", "distancia_km", "min_desde_ultima"]

def extrair_features(transacao: dict) -> list:
    return [
        transacao["valor"],             # quanto foi gasto (R$)
        transacao["hora"],              # hora do dia (0 a 23)
        transacao["distancia_km"],      # distância da última compra
        transacao["min_desde_ultima"],  # tempo desde a última compra
    ]
```

**Por que:** Modelos não entendem texto. Uma transação (`{"valor": 4000, "hora": 3, ...}`) precisa virar uma lista de números `[4000, 3, 800, 20]`.

---

### FASE 2 e 3 — Desvio Padrão + Distribuição Normal

```python
def z_score(valor, media, desvio):
    """Quantos desvios padrão o valor está longe da média."""
    if desvio == 0:
        return 0.0
    return (valor - media) / desvio

def densidade_normal(x, media, desvio):
    """A fórmula do sino (curva normal)."""
    coef = 1 / (desvio * math.sqrt(2 * math.pi))
    expoente = -((x - media) ** 2) / (2 * desvio ** 2)
    return coef * math.exp(expoente)
```

**Analogia:** Se o cliente sempre gasta em média R$150 (desvio = R$80), uma compra de R$4.000 tem z-score = (4000 - 150) / 80 ≈ 48. Algo a **48 desvios padrão** da média é estatisticamente impossível numa transação legítima.

---

### FASE 4 — Correlação de Pearson

```python
def correlacao_pearson(xs, ys):
    media_x, media_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - media_x) * (y - media_y) for x, y in zip(xs, ys))
    var_x = math.sqrt(sum((x - media_x) ** 2 for x in xs))
    var_y = math.sqrt(sum((y - media_y) ** 2 for y in ys))
    return cov / (var_x * var_y)
```

**Resultado:** Um número de **-1 a +1**.
- `+1` = sobem juntos (valor alto → mais fraude)
- `-1` = inversamente relacionados
- `0` = sem relação

**Exemplo de saída:**
```
valor             corr = +0.71  (sobe -> mais fraude)
hora              corr = -0.45  (hora alta -> menos fraude, ou seja, madrugada = fraude)
distancia_km      corr = +0.68  (longe -> mais fraude)
min_desde_ultima  corr = -0.52  (intervalo curto -> mais fraude)
```

---

### FASE 5 — Naive Bayes Gaussiano

```python
class DetectorBayes:
    def treinar(self, legitimas, fraudes):
        total = len(legitimas) + len(fraudes)
        self.prior = {
            "legitima": len(legitimas) / total,   # prob base: ~83% são legítimas
            "fraude": len(fraudes) / total,        # prob base: ~17% são fraudes
        }
        # Para cada feature, aprende média e desvio de cada classe
        self.stats = {
            "legitima": self._stats_por_feature(legitimas),
            "fraude": self._stats_por_feature(fraudes),
        }

    def probabilidade_fraude(self, transacao):
        features = extrair_features(transacao)
        scores = {}
        for classe in ("legitima", "fraude"):
            log_prob = math.log(self.prior[classe])   # começa com a crença base
            for i, x in enumerate(features):
                media, desvio = self.stats[classe][i]
                # Altura da curva normal: quão provável é esse valor nessa classe
                verossimilhanca = densidade_normal(x, media, desvio)
                log_prob += math.log(verossimilhanca + 1e-12)  # evita log(0)
            scores[classe] = log_prob

        # Normaliza para probabilidade 0-1
        maior = max(scores.values())
        p_leg = math.exp(scores["legitima"] - maior)
        p_fra = math.exp(scores["fraude"] - maior)
        return p_fra / (p_leg + p_fra)
```

**Por que log?** Multiplicar muitos números pequenos resulta em underflow (o número fica tão pequeno que vira zero). Usando logaritmo, multiplicação vira soma — matematicamente equivalente e numericamente estável.

**Resultado final:**
```
Compra suspeita (R$4.000, 3h, 800km, 20min): 99.4% de chance de fraude → BLOQUEAR
Compra normal  (R$120,  13h, 5km, 400min):  0.02% de chance de fraude → Aprovar
```

---

## 12. MNIST — Reconhecimento de Dígitos

### Contexto

MNIST é o "Hello World" do Deep Learning: 70.000 imagens 28×28 de dígitos escritos à mão (0-9).

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Carrega: 60k treino, 10k teste
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalização: divide por 255 para colocar pixels no range [0, 1]
X_train = X_train / 255.0
X_test  = X_test / 255.0

# One-hot encoding: y=3 vira [0,0,0,1,0,0,0,0,0,0]
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),   # achata 28x28=784 pixels em um vetor
    Dense(128, activation='relu'),   # camada oculta: 784 -> 128
    Dense(10, activation='softmax') # saída: 10 classes, probabilidades que somam 1
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # para multiclasse com one-hot
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')   # tipicamente ~98%
```

**Por que Flatten?** A imagem é uma matriz 28×28. A camada Dense espera um vetor. `Flatten` transforma `(28, 28)` em `(784,)`.

**Por que normalizar por 255?** Pixels vão de 0 a 255. Dividindo, ficam entre 0 e 1. Redes neurais treinam muito melhor com valores pequenos e na mesma escala.

**Por que one-hot encoding?** O rótulo `3` poderia implicar que `4 > 3` matematicamente, o que é falso para classes. `[0,0,0,1,0,0,0,0,0,0]` torna cada classe independente.

---

## 13. Iris — Classificação Multiclasse

### Contexto

Dataset clássico: 150 flores com 4 medidas (comprimento/largura de sépala e pétala) em 3 espécies.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

iris = load_iris()
X = iris.data    # (150, 4) — 150 amostras, 4 features
y = iris.target  # [0, 1, 2] — 3 espécies

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converte [0,1,2] para one-hot [[1,0,0],[0,1,0],[0,0,1]]
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))  # 4 features -> 10 neurônios
model.add(Dense(8, activation='relu'))                 # 10 -> 8
model.add(Dense(3, activation='softmax'))              # 8 -> 3 classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.1)

_, accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia: {accuracy*100:.2f}%')   # geralmente > 95%

# Mapeia índice para nome da espécie
species = {0: 'Iris setosa', 1: 'Iris versicolor', 2: 'Iris virginica'}
predictions = np.argmax(model.predict(X_test), axis=1)
```

**`np.argmax`:** A rede retorna probabilidades para cada classe, ex: `[0.02, 0.91, 0.07]`. `argmax` retorna o índice do maior valor → `1` (Iris versicolor).

---

## 14. Previsão de Ações com Random Forest

Já coberto na seção 8. Conceitos adicionais:

**Pipeline completo de um projeto de ML:**
```
1. Coleta de dados    → yfinance.download()
2. Pré-processamento  → tratar datas, selecionar features
3. Divisão treino/teste → train_test_split
4. Treinamento        → model.fit(X_train, y_train)
5. Avaliação          → mean_squared_error, rmse
6. Previsão           → model.predict(X_test)
```

**Overfitting vs Underfitting:**
- **Overfitting:** Modelo decora os dados de treino mas vai mal no teste (muito complexo)
- **Underfitting:** Modelo vai mal em ambos (muito simples)
- `random_state=42` ajuda na reprodutibilidade dos experimentos

---

## 15. LLMs com Ollama e LangChain

```python
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Conecta ao modelo llama2 rodando localmente
llm = Ollama(
    model="llama2",
    num_gpu=0,   # usa CPU (0 GPUs)
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    # StreamingStdOutCallbackHandler: imprime tokens conforme são gerados (streaming)
)

def gerar_insights_sobre_filmes(pergunta):
    prompt = f"Responda a seguinte pergunta sobre filmes: {pergunta}\n"
    prompt += "Por favor, forneça um resumo detalhado e quaisquer informações relevantes."
    return llm.invoke(prompt)

def main():
    pergunta = input("Sobre qual filme você deseja saber? ")
    resposta = gerar_insights_sobre_filmes(pergunta)
    print(f"Resposta: {resposta}")
```

**Conceitos:**
- **Ollama:** Roda LLMs (Large Language Models) localmente, sem precisar de API externa.
- **LangChain:** Framework para construir aplicações com LLMs — facilita prompts, chaining, memória.
- **Streaming:** `StreamingStdOutCallbackHandler` exibe o texto sendo gerado token por token, como o ChatGPT.
- **`llm.invoke(prompt)`:** Envia o prompt e recebe a resposta completa.

**Diferença entre ML clássico e LLM:**
| ML Clássico | LLM |
|------------|-----|
| Treinado para uma tarefa específica | Generalista, conversa em linguagem natural |
| Precisa de dados rotulados | Pré-treinado em textos da internet |
| Você treina do zero ou fine-tuna | Você usa direto com prompts |

---

## 16. API REST com FastAPI

```python
from fastapi import FastAPI
from routers import data, calculations

app = FastAPI()

# Registra os roteadores (conjuntos de endpoints)
app.include_router(data.router)
app.include_router(calculations.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the InvestPy API"}
```

```python
# routers/calculations.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class StockRequest(BaseModel):   # define o formato do JSON recebido
    ticker: str
    start_date: str
    end_date: str
    api_key: str

@router.post("/indicators/")
def calculate_indicators(request: StockRequest):
    try:
        df = dados.obter_dados_acao(request.ticker, request.api_key)
        df_filtered = df[(df['data'] >= request.start_date) & (df['data'] <= request.end_date)]
        df_with_indicators = calculos.calcular_retorno_diario(df_filtered)
        return {"retorno_diario": df_with_indicators.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

**Conceitos:**
- **FastAPI:** Framework Python para criar APIs REST rápidas com validação automática.
- **Pydantic BaseModel:** Define o schema do body do request. Se o cliente mandar tipo errado, FastAPI rejeita automaticamente com mensagem de erro clara.
- **Router:** Organiza endpoints em arquivos separados — `data.py` cuida dos dados, `calculations.py` dos cálculos.
- **HTTPException:** Retorna código de erro HTTP (400, 404, 500) com mensagem para o cliente.

---

## 17. Pré-processamento de Dados

> Baseado no arquivo `exemplo-processa-dados.py`.

### O código real

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carregar um dataset exemplo
data = load_iris()
X, y = data.data, data.target

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir o pipeline
pipeline = Pipeline([
   ('scaler', StandardScaler()),        # 1º passo: padroniza as escalas
   ('pca', PCA(n_components=3)),         # 2º passo: reduz de 4 para 3 dimensões
   ('classifier', RandomForestClassifier())  # 3º passo: classifica
])

# Treinar o pipeline (executa os 3 passos em sequência)
pipeline.fit(X_train, y_train)

# Fazer previsões
predictions = pipeline.predict(X_test)
print(predictions)
```

Este exemplo mostra o conceito mais importante de pré-processamento profissional no scikit-learn: o **Pipeline**, que encadeia várias etapas numa única sequência automática.

---

### O que é um Pipeline?

Um **Pipeline** é uma "linha de montagem" que conecta várias transformações + o modelo final. Os dados entram numa ponta, passam por cada etapa em ordem e saem como previsão na outra ponta.

**Analogia:** É como uma linha de produção de fábrica. A matéria-prima (dados brutos) entra, passa pela estação de limpeza (scaler), depois pela estação de moldagem (PCA), e por fim pela montagem final (classifier). Você aperta um botão só (`fit`) e a esteira faz tudo em ordem.

**Por que usar Pipeline?**
| Vantagem | Explicação |
|----------|-----------|
| **Organização** | Todas as etapas num único objeto, em ordem clara |
| **Sem vazamento de dados** | O scaler é ajustado só no treino automaticamente; evita *data leakage* |
| **Reprodutibilidade** | O mesmo processamento é aplicado no treino e na previsão, sempre igual |
| **Menos erros** | Você não esquece de aplicar uma transformação no teste |

Cada item da lista é uma tupla `('nome', objeto)`. O nome (`'scaler'`, `'pca'`, `'classifier'`) é um apelido usado para referenciar a etapa depois (importante na Seção 18).

---

### Etapa 1 — StandardScaler (Padronização)

```python
('scaler', StandardScaler())
```

**O que faz:** Coloca todas as features na mesma escala — média 0 e desvio padrão 1. Fórmula: `z = (x - média) / desvio`.

**Por que importa no Iris:** As 4 medidas (comprimento/largura de sépala e pétala) têm escalas diferentes. Sem padronizar, a feature com números maiores "grita mais alto" e domina indevidamente.

**Analogia:** Imagine comparar altura (1,70 m) com salário (5.000). O salário tem números muito maiores e pesaria mais no modelo só por causa da escala. Padronizar é colocar todo mundo para "falar no mesmo volume".

**Normalização vs Padronização:**
- **Padronização (StandardScaler):** média 0, desvio 1. Boa para SVM, regressão, PCA — usada aqui.
- **Normalização (MinMaxScaler):** comprime tudo entre 0 e 1. Boa para redes neurais e imagens (pixels/255).

---

### Etapa 2 — PCA (Redução de Dimensionalidade)

```python
('pca', PCA(n_components=3))   # de 4 features para 3
```

**O que é PCA (Principal Component Analysis):** técnica que **reduz o número de dimensões** (colunas) mantendo o máximo de informação possível. No Iris, vai de 4 features para 3 "componentes principais".

**Analogia:** É como resumir um livro de 400 páginas em 300, mantendo o essencial da história. Você perde um pouco de detalhe, mas ganha em simplicidade e velocidade. O PCA descarta a "redundância" — features que dizem quase a mesma coisa são combinadas.

**Como funciona (intuição):** O PCA encontra as direções onde os dados mais variam (os "componentes principais") e projeta os dados nessas direções. As direções de pouca variação (pouca informação) são descartadas.

**Por que usar:**
- Menos dimensões = treino mais rápido e menos risco de overfitting
- Ajuda a **visualizar** dados de muitas dimensões (reduzindo para 2 ou 3, dá para plotar)
- Remove ruído e redundância

**⚠️ Importante:** O PCA vem *depois* do scaler no pipeline — ele é sensível à escala, por isso padronizamos antes.

---

### Etapa 3 — RandomForestClassifier

```python
('classifier', RandomForestClassifier())
```

O classificador final (visto na Seção 8): uma "floresta" de árvores de decisão que votam na classe. Recebe os dados já padronizados e reduzidos pelo PCA, e faz a classificação das espécies de Iris.

**O fluxo completo do pipeline:**
```
Dados brutos (4 features)
      ↓  StandardScaler
Dados padronizados (média 0, desvio 1)
      ↓  PCA(n_components=3)
Dados reduzidos (3 componentes)
      ↓  RandomForestClassifier
Previsão da espécie (0, 1 ou 2)
```

Quando você chama `pipeline.fit(X_train, y_train)`, os três passos são treinados em cadeia. Em `pipeline.predict(X_test)`, os dados de teste passam pelas **mesmas** transformações antes de chegar ao classificador — garantindo consistência total.

---

## 18. Otimização de Hiperparâmetros

> Baseado no arquivo `exemplo-otimiza-hiperparametros.py`.

### O código real

Este exemplo é a evolução do anterior: pega o **mesmo Pipeline** (Scaler → PCA → Random Forest) e, em vez de treinar com valores fixos, usa o **GridSearchCV** para descobrir automaticamente a melhor combinação de hiperparâmetros.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris

data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mesmo pipeline da Seção 17
pipeline = Pipeline([
   ('scaler', StandardScaler()),
   ('pca', PCA(n_components=3)),
   ('classifier', RandomForestClassifier())
])

# Definir a grade de parâmetros para GridSearch
param_grid = {
   'classifier__n_estimators': [50, 100, 150],   # nº de árvores a testar
   'classifier__max_depth': [10, 20, 30]          # profundidade máxima a testar
}

# Executar GridSearchCV (cv=5 -> validação cruzada em 5 partes)
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Mostrar os melhores parâmetros encontrados
print(grid_search.best_params_)
```

---

### Parâmetro vs Hiperparâmetro

| Tipo | Quem define | Exemplos |
|------|-------------|----------|
| **Parâmetro** | O modelo aprende sozinho durante o treino | pesos das árvores, coeficientes da regressão |
| **Hiperparâmetro** | **Você** define antes de treinar | `n_estimators`, `max_depth`, `n_neighbors`, `epochs` |

**Analogia:** Ao assar um bolo, os **parâmetros** são como a massa cresce no forno (acontece sozinho). Os **hiperparâmetros** são a temperatura e o tempo que *você* escolhe. Escolher errado queima ou deixa cru — mesmo com bons ingredientes. O GridSearch é como testar várias combinações de temperatura e tempo para achar o ponto perfeito do bolo.

---

### A sintaxe `classifier__n_estimators` (dois underscores)

```python
'classifier__n_estimators': [50, 100, 150]
```

Esse é o detalhe mais importante do exemplo. O `__` (duplo underscore) diz ao GridSearch: **"ajuste o parâmetro `n_estimators` da etapa chamada `classifier` do pipeline"**.

- `classifier` → o apelido que demos à etapa Random Forest no pipeline
- `__` → separador "entre no objeto e acesse este parâmetro"
- `n_estimators` → o hiperparâmetro em si

Assim você pode otimizar hiperparâmetros de **qualquer etapa** do pipeline. Por exemplo, `pca__n_components: [2, 3, 4]` testaria diferentes reduções de dimensão.

---

### Como o GridSearchCV funciona

```python
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
```

**Grid Search** ("busca em grade") testa **todas as combinações** dos valores da grade:

```
n_estimators: [50, 100, 150]   ->  3 valores
max_depth:    [10, 20, 30]     ->  3 valores
                                   ─────────────
Total: 3 × 3 = 9 combinações
```

Cada uma das 9 combinações é avaliada com **validação cruzada de 5 partes (`cv=5`)**, ou seja, treinada e testada 5 vezes → **9 × 5 = 45 treinos** no total. No fim, `best_params_` retorna a combinação campeã.

**Saída típica:**
```python
{'classifier__max_depth': 10, 'classifier__n_estimators': 100}
```

---

### Validação Cruzada (Cross-Validation) — o `cv=5`

**O que é:** Divide os dados de treino em 5 partes (*folds*). Treina em 4 e testa na parte restante, rodando 5 vezes com partes diferentes. A média dos 5 resultados é a nota final daquela combinação.

```
Rodada 1:  [TESTE][treino][treino][treino][treino]
Rodada 2:  [treino][TESTE][treino][treino][treino]
Rodada 3:  [treino][treino][TESTE][treino][treino]
Rodada 4:  [treino][treino][treino][TESTE][treino]
Rodada 5:  [treino][treino][treino][treino][TESTE]
           → nota final = média das 5 rodadas
```

**Analogia:** Em vez de fazer uma única prova (que pode dar sorte ou azar), o "aluno" (modelo) faz 5 provas com questões diferentes e tira a média. É uma avaliação muito mais justa e confiável.

**Por que importa:** Evita a ilusão de um bom resultado por acaso numa única divisão treino/teste. É a diferença entre "achei que funcionou" e "tenho confiança de que funciona".

---

### Bônus — acessando o melhor modelo

Depois do `fit`, além de `best_params_`, você tem:

```python
print(grid_search.best_score_)      # melhor acurácia média na validação cruzada
melhor_modelo = grid_search.best_estimator_   # o pipeline já treinado com a melhor combinação
predictions = melhor_modelo.predict(X_test)   # pronto para prever
```

**Alternativa mais rápida — Random Search:** quando há muitos hiperparâmetros, testar *todas* as combinações fica caro. O `RandomizedSearchCV` testa apenas N combinações aleatórias — muito mais rápido e geralmente quase tão bom.

---

## 19. Hugging Face — Modelos Pré-treinados

### O que é o Hugging Face?

O **Hugging Face** é o "GitHub da Inteligência Artificial": uma plataforma onde a comunidade compartilha **modelos prontos e pré-treinados**, datasets e aplicações. Em vez de treinar uma rede neural do zero (o que custa milhões de dólares e semanas de GPU), você **baixa um modelo que já sabe** fazer a tarefa e usa em segundos.

**Analogia:** É como contratar um funcionário que já tem 10 anos de experiência, em vez de treinar um estagiário do zero. O modelo já "estudou" bilhões de textos ou imagens — você só o coloca para trabalhar.

**Componentes principais:**
| Conceito | O que é |
|----------|---------|
| **Model Hub** | Repositório com 500k+ modelos prontos (texto, imagem, áudio) |
| **`transformers`** | Biblioteca Python para carregar e usar modelos de linguagem |
| **`diffusers`** | Biblioteca para modelos de geração de imagem (Stable Diffusion) |
| **`datasets`** | Acesso a milhares de datasets prontos |
| **Tokenizer** | Converte texto em números que o modelo entende |
| **`push_to_hub`** | Publica seu próprio modelo na plataforma |

---

### O que é um Transformer?

Os modelos modernos de IA (ChatGPT, BERT, etc.) usam a arquitetura **Transformer**, que introduziu o mecanismo de **atenção** (*attention*): o modelo aprende a "prestar atenção" nas palavras mais relevantes de uma frase para entender o contexto.

**Analogia:** Ao ler "o banco estava cheio", você usa o contexto para saber se é banco de dinheiro ou de praça. A atenção faz o modelo olhar as palavras vizinhas para desfazer a ambiguidade — exatamente como nosso cérebro.

**BERT / DistilBERT** (usados nos exemplos) são Transformers especializados em *entender* texto (classificação, sentimento). O **DistilBERT** é uma versão "destilada" (comprimida) do BERT: 40% menor e 60% mais rápido, mantendo ~97% da qualidade.

---

### Exemplo 1 — Geração de Imagem com Stable Diffusion (`exemplo1/app/app.py`)

```python
import torch
from diffusers import StableDiffusionPipeline

# Baixa o modelo Stable Diffusion pré-treinado (~4 GB na primeira vez)
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Usa GPU se disponível (muito mais rápido), senão CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)

def generate_image(prompt):
    generated_image = pipeline(prompt).images[0]   # gera a imagem a partir do texto
    return generated_image

prompt = "a photo of an astronaut riding a horse on mars"
image = generate_image(prompt)
image.save("astronaut_rides_horse.png")
```

**Linha a linha:**
- `StableDiffusionPipeline` — um *pipeline* junta todas as etapas (tokenizar o texto, gerar ruído, refinar em imagem) num só objeto fácil de usar.
- `from_pretrained("CompVis/stable-diffusion-v1-4")` — baixa o modelo do Hub. `"CompVis/..."` é o endereço `organização/nome-do-modelo`.
- `"cuda" if torch.cuda.is_available()` — detecta placa de vídeo NVIDIA. Geração de imagem é pesada; sem GPU pode levar minutos por imagem.
- `pipeline(prompt).images[0]` — o modelo transforma **texto em imagem** (*text-to-image*). Retorna uma lista; pegamos a primeira `[0]`.

**O que é Stable Diffusion?** Um modelo de **difusão**: começa com ruído puro (estática de TV) e, passo a passo, "limpa" o ruído até formar a imagem que corresponde ao texto. É como um escultor que parte de um bloco de mármore (ruído) e vai removendo até surgir a estátua descrita.

---

### Exemplo 2 — Carregando um Dataset do Hugging Face (`exemplo2/index.py`)

```python
import pandas as pd

# Mapeia os arquivos de cada divisão (split) do dataset IMDB
splits = {
    'train': 'plain_text/train-00000-of-00001.parquet',
    'test':  'plain_text/test-00000-of-00001.parquet',
    'unsupervised': 'plain_text/unsupervised-00000-of-00001.parquet'
}

# Lê direto do Hub usando o protocolo "hf://"
df = pd.read_parquet("hf://datasets/stanfordnlp/imdb/" + splits["train"])

print(df.info())
```

**Conceitos:**
- **IMDB dataset:** 50.000 avaliações de filmes rotuladas como positivas ou negativas — o dataset clássico para treinar análise de sentimento.
- **`hf://datasets/...`** — o Pandas lê o arquivo **direto dos servidores do Hugging Face**, sem download manual.
- **Parquet** — formato de arquivo colunar, comprimido e rápido, muito usado para grandes volumes de dados (bem melhor que CSV para milhões de linhas).
- **Splits (train/test/unsupervised):** dados já vêm divididos em treino, teste e não-rotulado — pronto para uso.

---

### Exemplo 3 — Baixar, Salvar e Publicar um Modelo (`exemplo3/`)

Este exemplo mostra o ciclo completo de trabalho com modelos no Hub. Usa 3 arquivos:

**`.env` — variáveis de ambiente (segredos):**
```
MODEL_NAME=distilbert-base-uncased
MODEL_REPO_NAME=tadrianonet/curso-fiap
HUGGINGFACE_TOKEN=      # seu token de acesso (nunca compartilhe!)
```

**`main.py` — carrega o modelo e o publica:**
```python
import os
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_config():
    load_dotenv()   # lê o arquivo .env
    config = {
        "model_name": os.getenv("MODEL_NAME"),
        "model_repo_name": "tadrianonet/curso-fiap",
        "huggingface_token": os.getenv("HUGGINGFACE_TOKEN"),
    }
    return config

def load_model_and_tokenizer(model_name):
    # num_labels=2 -> classificação binária (ex: positivo/negativo)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def main():
    config = load_config()
    model, tokenizer = load_model_and_tokenizer(config["model_name"])

    # Salva localmente numa pasta
    model.save_pretrained("./simple_model")
    tokenizer.save_pretrained("./simple_model")

    # Publica no Hugging Face Hub
    from publish import publish_model
    publish_model("./simple_model", config["model_repo_name"], config["huggingface_token"])

if __name__ == "__main__":
    main()
```

**`publish.py` — envia para o Hub:**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def publish_model(model_dir, model_repo_name, huggingface_token):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # push_to_hub sobe o modelo E o tokenizer para o seu repositório
    model.push_to_hub(model_repo_name, token=huggingface_token)
    tokenizer.push_to_hub(model_repo_name, token=huggingface_token)
```

**Conceitos importantes:**
- **`Auto...` classes:** `AutoModel` e `AutoTokenizer` são "carregadores inteligentes" — detectam automaticamente a arquitetura certa (BERT, GPT, etc.) só pelo nome do modelo. Você não precisa saber os detalhes internos.
- **Tokenizer + Model andam juntos:** o tokenizer converte texto→números do jeito específico que *aquele* modelo espera. Usar o tokenizer errado quebra tudo. Por isso sempre se salva e publica os dois juntos.
- **`.env` e `python-dotenv`:** segredos (tokens, senhas) ficam num arquivo `.env` que **nunca** vai para o Git (fica no `.gitignore`). O código lê com `os.getenv()`. Isso evita vazar sua chave de acesso publicamente.
- **`push_to_hub`:** publica o modelo no seu repositório do Hub, tornando-o acessível para qualquer pessoa (ou só você) baixar com `from_pretrained`.

---

### Exemplo 3b — Testando o Modelo Publicado (`exemplo3/teste.py`)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

nome_repo = "tadrianonet/curso-fiap"

# Baixa o modelo que acabamos de publicar
tokenizer = AutoTokenizer.from_pretrained(nome_repo)
model = AutoModelForSequenceClassification.from_pretrained(nome_repo)

class_labels = {0: "Negative", 1: "Positive"}
text = "Este é um exemplo para demonstrar como publicar um modelo."

# Converte o texto em tensores PyTorch
inputs = tokenizer(text, return_tensors="pt")

# torch.no_grad(): desliga o cálculo de gradientes (não estamos treinando -> mais rápido)
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits                          # scores brutos por classe
predicted_class_id = torch.argmax(logits).item() # índice do maior score
predicted_class_label = class_labels[predicted_class_id]

print(f"Texto: {text}")
print(f"Classe prevista: {predicted_class_label}")
```

**Linha a linha:**
- `tokenizer(text, return_tensors="pt")` — transforma a frase em números; `"pt"` = formato PyTorch (*PyTorch tensors*).
- `model(**inputs)` — o `**` desempacota o dicionário de entradas (input_ids, attention_mask) como argumentos.
- **`logits`** — saídas brutas do modelo, ex: `[-2.3, 1.8]`. Ainda não são probabilidades.
- `torch.argmax(logits)` — pega o índice do maior valor. Aqui, índice 1 (Positive).
- `torch.no_grad()` — na *inferência* (uso, não treino) não precisamos de gradientes; desligar economiza memória e acelera.

---

### Exemplo 4 — Análise de Sentimento com Softmax (`modelo-professor-exemplo/app.py`)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("tadrianonet/distilbert-text-classification")
model = AutoModelForSequenceClassification.from_pretrained("tadrianonet/distilbert-text-classification")

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # softmax transforma logits em probabilidades que somam 1
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    return predicted_label, probabilities

texts_to_test = [
    "Estou extremamente feliz com os resultados que conseguimos!",
    "A comida estava deliciosa e o atendimento foi excelente.",
    # ...
]
classes = ["Negativo/Neutro", "Positivo"]

for text in texts_to_test:
    predicted_label, probabilities = predict(text)
    print(f"Texto: {text}")
    print(f"Rótulo: {predicted_label} ({classes[predicted_label]})")
    print(f"Probabilidades: {probabilities}\n")
```

**Diferença para o exemplo anterior:** aqui aplicamos **`softmax`** aos logits para obter **probabilidades reais** (ex: `[0.02, 0.98]` = 98% positivo). No `teste.py`, usamos `argmax` direto nos logits — o resultado da *classe* é o mesmo, mas o softmax dá a **confiança** da previsão, informação valiosa.

**Por que softmax?** Logits podem ser qualquer número (`-2.3`, `1.8`). Softmax os "espreme" para o intervalo [0, 1] e faz somar 100%, transformando scores em probabilidades interpretáveis.

---

### Exemplo 5 — App Web de Análise de Sentimento em JavaScript (`modelo-analise-sentimento/`)

Este exemplo mostra que Hugging Face **não é só Python** — há a biblioteca `@xenova/transformers` que roda modelos **direto no JavaScript/Node.js**, sem precisar de servidor Python.

**`index.js` — servidor Express com o modelo:**
```javascript
import express from 'express';
import bodyParser from 'body-parser';
import { pipeline } from '@xenova/transformers';

const app = express();
const PORT = process.env.PORT || 3001;

app.use(bodyParser.urlencoded({ extended: true }));
app.set('view engine', 'ejs');   // motor de templates HTML

// Página inicial
app.get('/', (req, res) => {
    res.render('index', { result: null });
});

// Rota que analisa o sentimento do texto enviado
app.post('/analyze', async (req, res) => {
    try {
        const { text } = req.body;
        // Carrega o pipeline de análise de sentimento (modelo DistilBERT)
        let pipe = await pipeline('sentiment-analysis',
                   'Xenova/distilbert-base-uncased-finetuned-sst-2-english');
        let out = await pipe(text);     // executa a previsão
        res.render('index', { result: out });
    } catch (error) {
        res.render('index', { result: `Error: ${error.message}` });
    }
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
```

**Arquitetura da aplicação web:**
```
Usuário digita texto no navegador (index.ejs)
          ↓  POST /analyze
Servidor Node.js (index.js)
          ↓  pipeline('sentiment-analysis', ...)
Modelo DistilBERT (Xenova) roda a previsão
          ↓
Resultado volta e é exibido na tela
```

**Conceitos:**
- **`pipeline('sentiment-analysis', ...)`** — o conceito de *pipeline* do Hugging Face existe também em JS: uma linha carrega o modelo e o deixa pronto para usar.
- **`@xenova/transformers`** — porta os modelos do Hugging Face para JavaScript, rodando até no navegador (via WebAssembly). Democratiza IA para desenvolvedores web.
- **Express + EJS:** Express é o framework de servidor web do Node.js (equivalente ao FastAPI em Python); EJS é o motor de templates que injeta o resultado no HTML.
- **`async/await`:** carregar e rodar o modelo leva tempo; `await` espera a conclusão sem travar o servidor.

**Por que isso é importante?** Mostra que modelos de IA podem rodar em **qualquer stack** — Python para ciência de dados e treino, JavaScript para colocar IA direto em aplicações web e no navegador do usuário.

---

### Resumo — Formas de usar o Hugging Face

| Biblioteca | Linguagem | Para quê |
|-----------|-----------|----------|
| `transformers` | Python | Modelos de texto (NLP): classificação, sentimento, tradução |
| `diffusers` | Python | Geração de imagem (Stable Diffusion) |
| `datasets` | Python | Baixar datasets prontos |
| `@xenova/transformers` | JavaScript | Rodar modelos em Node.js / navegador |

**Fluxo típico de trabalho:**
```
1. Escolher modelo no Hub (huggingface.co/models)
2. from_pretrained("org/modelo")   -> baixa o modelo e o tokenizer
3. tokenizer(texto)                -> texto vira números
4. model(**inputs)                 -> gera logits
5. softmax + argmax                -> probabilidade e classe final
   (opcional) push_to_hub          -> publica seu próprio modelo
```

---

## 20. Glossário Rápido

| Termo | Definição |
|-------|-----------|
| **Feature** | Uma característica/coluna de entrada do modelo (ex: tamanho da casa, peso da fruta) |
| **Label / Target** | A resposta que o modelo deve prever (ex: preço, classe, probabilidade) |
| **Treino / Teste** | Divisão dos dados: treino para aprender, teste para avaliar |
| **Overfitting** | Modelo decorou os dados de treino e não generaliza para dados novos |
| **Underfitting** | Modelo é simples demais e não captura os padrões |
| **Epoch** | Uma passagem completa pelos dados de treino |
| **Batch** | Subconjunto de dados processados de uma vez antes de atualizar pesos |
| **Learning Rate** | Tamanho do passo na atualização dos pesos (muito grande = instável; muito pequeno = lento) |
| **Gradient Descent** | Algoritmo de otimização que minimiza a função de perda |
| **Backpropagation** | Calcula os gradientes percorrendo a rede de trás para frente |
| **One-hot Encoding** | Transforma classe numérica em vetor binário: 3 classes → [1,0,0], [0,1,0], [0,0,1] |
| **Normalização** | Colocar features na mesma escala (ex: dividir pixels por 255) |
| **R²** | Coeficiente de determinação (0 a 1): quanto da variação o modelo explica |
| **RMSE** | Raiz do Erro Quadrático Médio — erro típico nas mesmas unidades do target |
| **Acurácia** | % de classificações corretas |
| **Prior (Bayes)** | Probabilidade antes de ver evidências: ex: 17% das transações são fraude |
| **Softmax** | Converte scores em probabilidades que somam 1 (classificação multiclasse) |
| **ReLU** | Função de ativação: max(0, x) — elimina negativos, resolve vanishing gradient |
| **Adam** | Otimizador adaptativo — ajusta o learning rate automaticamente |
| **Hiperparâmetro** | Configuração escolhida por você (ex: n_clusters, n_neighbors, epochs) — não aprendida |
| **Parâmetro** | Aprendido pelo modelo (pesos, vieses) |
| **Grid Search** | Testa todas as combinações de hiperparâmetros de uma grade |
| **Cross-Validation** | Divide os dados em K partes e testa K vezes para avaliação confiável |
| **Pipeline (sklearn)** | Linha de montagem que encadeia transformações + modelo numa sequência |
| **PCA** | Reduz o número de dimensões mantendo o máximo de informação |
| **Redução de dimensionalidade** | Diminuir o número de features (colunas) dos dados |
| **Data Leakage** | Vazamento: o modelo "espia" dados de teste que não deveria ver |
| **Imputação** | Preencher valores ausentes (ex: com a média) |
| **Transformer** | Arquitetura de rede neural baseada em atenção (base do BERT, GPT) |
| **Atenção (Attention)** | Mecanismo que faz o modelo focar nas palavras relevantes do contexto |
| **Tokenizer** | Converte texto em números que o modelo entende (e vice-versa) |
| **Logits** | Saídas brutas do modelo, antes de virarem probabilidades |
| **Pipeline (HF)** | Objeto que junta tokenizer + modelo + pós-processamento numa linha |
| **from_pretrained** | Baixa um modelo/tokenizer pré-treinado do Hugging Face Hub |
| **push_to_hub** | Publica seu modelo no Hugging Face Hub |
| **Fine-tuning** | Ajustar um modelo pré-treinado para uma tarefa específica |
| **Inferência** | Usar o modelo já treinado para fazer previsões (não é treino) |
| **DistilBERT** | Versão comprimida do BERT: 40% menor, 60% mais rápido |
| **Stable Diffusion** | Modelo que gera imagens a partir de texto removendo ruído gradualmente |

---

# PARTE II — Curso de IA & Machine Learning (Aulas Práticas)

> Esta parte acompanha, aula a aula, os notebooks do curso (`machine-learning/CURSO_IA_ML`). Cada seção explica **o dataset, o pré-processamento, os algoritmos, as funções usadas e o porquê de cada decisão** — no mesmo estilo linha a linha da Parte I.

---

## 21. Aula 1 — Conceitos Básicos e Aplicações de ML

> Aula teórica (`Aula 1/`). Fundamenta todos os projetos práticos das aulas seguintes.

### O que é Machine Learning?

Machine Learning (Aprendizado de Máquina) é, basicamente, **ensinar o computador a aprender com os dados** — em vez de seguir regras escritas à mão. Duas definições clássicas fundamentam o campo:

> **"Campo de estudo que dá aos computadores a habilidade de aprender sem ser explicitamente programado."** — Arthur Samuel (1959)

> **"Diz-se que um programa aprende pela experiência E em relação a uma tarefa T e uma medida de desempenho P, se o seu desempenho em T, medido por P, melhora com a experiência E."** — Tom Mitchell (1997)

A definição de Mitchell dá o vocabulário prático: **Tarefa (T)**, **Experiência (E, os dados)** e **Desempenho (P, a métrica)**.

**Programação tradicional vs Machine Learning:**

| | Entra | Sai |
|---|-------|-----|
| **Programação tradicional** | Dados + Regras (código) | Respostas |
| **Machine Learning** | Dados + Respostas (exemplos) | Regras (o modelo) |

**Analogia:** Ensinar uma criança a reconhecer gatos. Você não descreve "orelha triangular, bigodes, 4 patas". Você mostra centenas de fotos dizendo "isto é um gato". Depois de exemplos suficientes, ela generaliza para gatos que nunca viu. ML faz o mesmo com dados.

### Por que usar ML? O exemplo do filtro de spam

| Abordagem | Como funciona | Problema |
|-----------|---------------|----------|
| **Método tradicional** | Longa lista de regras/códigos ("se contém 'grátis', é spam") | Difícil de manter; cada novo truque de spam exige nova regra |
| **Machine Learning** | Aprende sozinho **quais palavras indicam spam** a partir de e-mails rotulados | Programa menor, fácil de manter e mais preciso; adapta-se a spam novo |

É a motivação central do ML: quando as regras são complexas demais ou mudam o tempo todo, **deixar o algoritmo aprender os padrões** vence a programação manual.

### IA ⊃ ML ⊃ Deep Learning

```
INTELIGÊNCIA ARTIFICIAL  (qualquer técnica que simule inteligência)
   └── MACHINE LEARNING   (aprende com dados, sem regras explícitas)
          └── DEEP LEARNING  (ML com redes neurais profundas)
```

- **IA** é o guarda-chuva: qualquer sistema que imita a inteligência humana (inclusive regras fixas).
- **ML** é o subconjunto que **aprende com dados**.
- **Deep Learning** é o subconjunto de ML que usa **redes neurais com muitas camadas** (visto nas Seções 9–13).

### Os três tipos de aprendizado

| Tipo | Tem rótulo (y)? | O que faz | Exemplos no guia |
|------|-----------------|-----------|------------------|
| **Supervisionado** | Sim | Aprende a partir de exemplos com resposta certa (um "professor") | Regressão Linear, KNN, SVM, Random Forest |
| **Não supervisionado** | Não | Descobre estrutura/grupos sozinho, sem professor | K-Means, PCA |
| **Por reforço** | Recompensa | Aprende por tentativa e erro, com **punição × recompensa**, atualizando uma "política" de regras | Jogos, robótica |

**Algoritmos por tipo (como listados na aula):**

| Supervisionado | Não Supervisionado |
|----------------|--------------------|
| KNN, Regressão Linear, Regressão Logística, SVM, Árvores de Decisão, Árvores Aleatórias (Random Forest), Redes Neurais | K-Means, DBSCAN, Cluster Hierárquico, T-SNE, PCA, Apriori |

> **E se a base não tiver rótulos?** Nem sempre os dados vêm "rotulados bonitinhos". Às vezes você cria o rótulo (variável alvo) com **regra de negócio**; quando nem isso é possível, parte-se para técnicas **não supervisionadas** (clustering) que encontram padrões sozinhas — por exemplo, agrupar visitantes de um site por comportamento sem dizer ao algoritmo quem é parecido com quem.

Dentro do **supervisionado** há duas famílias de problema:
- **Regressão** → prever um **valor contínuo** (preço, temperatura, custo). É o foco das Aulas 2 e do Desafio.
- **Classificação** → prever uma **categoria** (fraude/não fraude, espécie, churn/não churn).

### Aplicações reais de ML

| Área | Aplicação |
|------|-----------|
| Finanças | Detecção de fraude, score de crédito, previsão de ações |
| Saúde | Diagnóstico por imagem, previsão de risco de doenças |
| Varejo | Recomendação de produtos, segmentação de clientes |
| Indústria | Manutenção preditiva, controle de qualidade |
| Linguagem | Tradução, chatbots, análise de sentimento (LLMs) |

### O fluxo de um projeto de ML (usado em todas as aulas seguintes)

```
1. Coleta de dados          → CSV, Excel, API, banco de dados
2. Análise exploratória     → entender colunas, escalas, correlações, nulos
3. Pré-processamento        → tratar nulos, codificar categorias, escalonar
4. Divisão treino/teste     → train_test_split
5. Treinamento              → model.fit(X_train, y_train)
6. Avaliação                → R², MAE, RMSE, acurácia...
7. Ajuste / novo modelo     → trocar algoritmo, otimizar hiperparâmetros
```

Guarde esse ciclo: as Aulas 2, 3, 4 e o Desafio são aplicações concretas dele. Veja o [Glossário](#20-glossário-rápido) para relembrar qualquer termo.

---

## 22. Aula 2 — Regressão Linear Simples (Sorvete)

> Notebook `Aula 2/Regressão Linear Simples.ipynb` + `Sorvete.xlsx` (100 registros).

**Objetivo:** prever as **vendas de sorvete** a partir da **temperatura** do dia. Uma única feature de entrada (temperatura) → uma saída (vendas). É o caso mais simples de regressão: `y = β·x + α`.

### 1. Importando bibliotecas e dados

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dados = pd.read_excel("Sorvete.xlsx")
dados.head()
```

- `pd.read_excel` — lê planilhas `.xlsx` (equivalente ao `read_csv`, mas para Excel). O dataset tem 2 colunas: `Temperatura` e `Vendas_Sorvetes`.
- Já importamos tudo que o fluxo exige: o divisor treino/teste, o modelo e as 3 métricas de regressão.

### 2. Análise exploratória — o gráfico de dispersão

```python
plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvetes (milhares)')
plt.title('Relação entre Temperatura e Vendas de Sorvetes')
plt.show()
```

O *scatter plot* mostra visualmente que os pontos sobem quase em linha reta: quanto mais quente, mais sorvete. Esse padrão linear é exatamente o que a regressão linear sabe modelar.

### 3. Medindo a correlação

```python
dados.corr()
# Temperatura x Vendas_Sorvetes = 0.985589
```

**`.corr()`** calcula a **correlação de Pearson** entre as colunas (o mesmo conceito da Seção 11, Fase 4). O valor **0.98** confirma numericamente o que o gráfico sugeriu: correlação altíssima e positiva.

**Por que checar a correlação antes de modelar?** A correlação mede a **força e a direção** da relação linear. Uma correlação alta indica que uma **regressão linear é apropriada** — a feature realmente ajuda a prever o alvo. Se fosse ~0, o modelo linear seria inútil.

### 4. A equação da regressão

`Yi = β·Xi + α + εi`

| Termo | Nome | Significado |
|-------|------|-------------|
| `α` | intercepto | valor previsto quando X = 0 |
| `β` | coeficiente / declive | quanto Y muda a cada +1 em X |
| `X` | variável preditora (independente) | temperatura |
| `Y` | variável resposta (dependente / alvo) | vendas |
| `ε` | resíduo | erro: diferença entre o real e o previsto |

**Como o modelo "aprende" α e β?** Pelo método dos **Mínimos Quadrados Ordinários (OLS)**: ele encontra a reta que **minimiza a soma dos erros ao quadrado** (RSS — *residual sum of squares*). Elevar ao quadrado penaliza erros grandes e garante que positivos e negativos não se anulem.

### 5. Separando treino e teste

```python
X = dados[['Temperatura']]   # DataFrame (2D) — note os [[ ]]
y = dados['Vendas_Sorvetes'] # Series (1D)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# X_train.shape = (80, 1)   |   X_test.shape = (20, 1)
```

- `X = dados[['Temperatura']]` — duplo colchete devolve um **DataFrame** (formato 2D `linhas × colunas`) que o scikit-learn exige como entrada. Colchete simples devolveria uma Series 1D.
- `test_size=0.2` — reserva **20% (20 registros)** para teste; 80% (80) para treino. O modelo **nunca vê** o teste durante o treino.
- `random_state=42` — fixa o sorteio da divisão para o resultado ser **reprodutível**.

### 6. Treinando e prevendo

```python
modelo = LinearRegression()
modelo.fit(X_train, y_train)          # aprende α e β nos dados de treino
previsoes = modelo.predict(X_test)    # prevê nas temperaturas de teste
```

### 7. Avaliando o modelo

```python
erro_medio_quadratico = mean_squared_error(y_test, previsoes)  # 101.65
erro_absoluto_medio   = mean_absolute_error(y_test, previsoes) # 7.68
r_quadrado            = r2_score(y_test, previsoes)            # 0.9594
```

| Métrica | Valor | Como ler |
|---------|-------|----------|
| **MSE** (erro quadrático médio) | 101.65 | média dos erros ao quadrado — penaliza erros grandes |
| **MAE** (erro absoluto médio) | 7.68 | em média erra ~7,68 mil sorvetes (mesma unidade do alvo) |
| **R²** (coef. de determinação) | 0.959 | explica **95,9%** da variação das vendas |

**RMSE vs MSE:** o RMSE é a raiz do MSE (`np.sqrt(mse)`), voltando à unidade original. **R² próximo de 1** é ótimo: o modelo captou quase toda a relação. Como só há 1 feature muito correlacionada, esse resultado forte era esperado.

**Resumo da Aula 2 (simples):** dataset com 1 feature → confirma correlação → `train_test_split` → `LinearRegression` → avalia com MAE/MSE/R². É o fluxo da Seção 21 no seu formato mínimo.

---

## 23. Aula 2 — Regressão Linear Múltipla (Casas da Califórnia, projeto end-to-end)

> Notebook `Aula 2/Case end to end Regressão.ipynb` + `housing.csv` (20.640 distritos, 10 colunas). É o **projeto mais completo do curso**: pega dados sujos e reais e percorre todo o pipeline até comparar modelos.

**Objetivo:** prever o **valor médio das casas** (`median_house_value`) de cada distrito da Califórnia a partir de várias features (localização, idade dos imóveis, cômodos, renda, proximidade do oceano...). "Múltipla" porque agora há **várias variáveis preditoras**.

### 1. Conhecendo os dados

```python
import pandas as pd
dataset = pd.read_csv("housing.csv")
dataset.shape          # (20640, 10)
dataset.info()
dataset.describe()
```

- `.shape` → 20.640 linhas × 10 colunas.
- `.info()` → tipos e contagem de não-nulos. Descobre-se que **`total_bedrooms` tem só 20.433 preenchidos** → **207 distritos com valor faltante** (nulos a tratar).
- `.describe()` → estatísticas (média, desvio, min, max, quartis) das colunas numéricas. Revela escalas muito diferentes (ex.: `median_income` ~0–15 vs `total_rooms` até 39.320).

```python
set(dataset["ocean_proximity"])
# {'<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'}
dataset["ocean_proximity"].value_counts()
```

`ocean_proximity` é a **única coluna de texto (categórica)** — tem 5 categorias. Modelos não entendem texto, então ela precisará ser codificada mais adiante.

```python
dataset.hist(bins=50, figsize=(20,15))
```

Histogramas de todas as colunas revelam problemas: `median_house_value` está **truncado em ~500 mil** (limite artificial da base) e várias distribuições são **assimétricas**. Diagnóstico antes de modelar evita surpresas.

### 2. Amostragem estratificada por renda

Um especialista avisou que a **renda média** é o atributo mais importante. Ao dividir treino/teste, precisamos garantir que **as duas bases tenham a mesma proporção de faixas de renda** — senão o teste pode ficar enviesado.

```python
import numpy as np
# Cria faixas discretas de renda (income_cat) a partir da renda contínua
dataset["income_cat"] = pd.cut(dataset["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

- `pd.cut` — recorta uma variável **contínua em faixas (bins)** discretas. Aqui a renda vira 5 categorias (1 a 5).

```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set  = dataset.loc[test_index]
```

- **`StratifiedShuffleSplit`** — divide treino/teste **preservando a proporção** de cada faixa de renda em ambos os lados (amostragem *estratificada*). Comparar `value_counts()/len(...)` mostra que treino, teste e base completa têm proporções quase idênticas (faixa 3 ≈ 35% nos três). É mais justo que o `train_test_split` aleatório puro.

```python
# income_cat era só auxiliar — remove das duas bases
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

### 3. Análise geográfica e correlações

```python
housing = strat_train_set.copy()  # trabalha só no treino (não "espia" o teste)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
```

Plotando longitude/latitude coloridas pelo preço, aparece o mapa da Califórnia: preços altos **perto do oceano e em áreas densas**. `alpha=0.4` revela concentração de pontos; `s=` dá o tamanho pela população; `c=`/`cmap` dão a cor pelo preço.

```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# median_income     0.687151   <- mais correlacionada
# total_rooms       0.135140
# ...
# latitude         -0.142673
```

A correlação confirma: **`median_income` (0.69)** é de longe a melhor preditora do preço. As demais têm relação fraca.

### 4. Tratando valores nulos (imputação)

```python
housing = strat_train_set.drop("median_house_value", axis=1)  # X (features)
housing_labels = strat_train_set["median_house_value"].copy() # y (alvo)
```

Sempre separe o alvo (`y`) das features (`X`) antes de transformar.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1)  # mediana só funciona em números
imputer.fit(housing_num)          # calcula a mediana de cada coluna -> imputer.statistics_
X = imputer.transform(housing_num) # substitui os NaN pela mediana da coluna
```

- **`SimpleImputer(strategy="median")`** — preenche valores faltantes com a **mediana** da coluna (mais robusta a outliers que a média). `fit` aprende as medianas no treino; `transform` aplica. Assim os 207 `total_bedrooms` nulos são preenchidos sem descartar linhas.

**Por que mediana e não apagar as linhas?** Descartar jogaria fora 207 distritos inteiros de informação. Imputar preserva os dados.

### 5. Codificando a variável categórica

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

housing_cat = housing[['ocean_proximity']]

# OrdinalEncoder: cada categoria vira um inteiro (0,1,2,3,4)
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# OneHotEncoder: cada categoria vira uma coluna binária (0/1)
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

| Encoder | Saída | Quando usar |
|---------|-------|-------------|
| **OrdinalEncoder** | 1 coluna com inteiros (0–4) | categorias **com ordem** (baixo/médio/alto) |
| **OneHotEncoder** | 5 colunas binárias | categorias **sem ordem** (nominais) — o caso aqui |

**Por que One-Hot para `ocean_proximity`?** Com OrdinalEncoder o modelo pensaria que `NEAR OCEAN (4) > INLAND (1)` matematicamente, o que é falso — não há ordem entre localizações. O One-Hot cria uma coluna independente por categoria, evitando essa ordem inexistente (mesma lógica do one-hot da Seção 12).

### 6. Montando o Pipeline completo

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Pipeline para as colunas NUMÉRICAS: imputar nulos + padronizar escala
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
```

```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)      # todas as colunas numéricas
cat_attribs = ["ocean_proximity"]    # a coluna categórica

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),     # aplica o num_pipeline nas numéricas
    ("cat", OneHotEncoder(), cat_attribs),  # aplica One-Hot na categórica
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape   # (16512, 13)
```

- **`ColumnTransformer`** — aplica **transformações diferentes a colunas diferentes** e junta tudo. As numéricas passam por imputação + `StandardScaler`; a categórica por One-Hot. É a peça-chave que organiza todo o pré-processamento em um só objeto (conceito do Pipeline da Seção 17, agora em um caso real).
- Resultado: **13 colunas** (8 numéricas padronizadas + 5 binárias do One-Hot), sem nulos, tudo na mesma escala. Dados "limpinhos" prontos para o modelo.

### 7. Modelo 1 — Regressão Linear

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))  # 69050
lin_mae  = mean_absolute_error(housing_labels, housing_predictions)          # 49905
r2       = r2_score(housing_labels, housing_predictions)                     # 0.6438
```

O **MAPE** (erro percentual médio) é calculado por uma função manual:

```python
def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    return np.mean(relative_errors) * 100    # 28.65%
```

Um erro médio de **~US$ 69 mil (RMSE)** e **MAPE de 28,65%** é alto demais para preços que giram em torno de US$ 120–265 mil. A regressão linear é **simples demais** (*underfitting*) para capturar as relações não lineares dos imóveis.

### 8. Modelo 2 — Árvore de Decisão

```python
from sklearn.tree import DecisionTreeRegressor

model_dtr = DecisionTreeRegressor(max_depth=10)
model_dtr.fit(housing_prepared, housing_labels)

housing_predictions = model_dtr.predict(housing_prepared)
# RMSE = 47873  |  MAE = 32067  |  R² = 0.8288  |  MAPE = 17.94%
```

- **`DecisionTreeRegressor`** — modela relações **não lineares** dividindo os dados em regras sucessivas ("se renda > X e perto do oceano, então preço ~Y"). `max_depth=10` limita a profundidade para não decorar demais (controle de overfitting).

**Comparando os dois modelos:**

| Modelo | RMSE | MAE | R² | MAPE |
|--------|------|-----|-----|------|
| Regressão Linear | 69.050 | 49.905 | 0.644 | 28,65% |
| Árvore de Decisão | 47.873 | 32.067 | 0.829 | 17,94% |

A árvore vence com folga → melhor ajuste aos padrões não lineares.

> **⚠️ Nota didática:** neste notebook as métricas foram medidas sobre os **próprios dados de treino** (`housing_prepared`). O correto para avaliar generalização é aplicar o `full_pipeline.transform` no `strat_test_set` e medir no **teste**. Erro baixo no treino pode mascarar overfitting — por isso, no mundo real, sempre reporte a métrica no conjunto de teste (ou use validação cruzada, Seção 18).

**Resumo da Aula 2 (múltipla):** EDA → estratificação por renda → imputação de nulos → codificação categórica (One-Hot) → `ColumnTransformer`/Pipeline → compara Regressão Linear vs Árvore com 4 métricas. É o retrato completo de um projeto de ML end-to-end.

---

## 24. Aula 3 — PCA (Análise de Componentes Principais)

> Notebook `Aula 3/Case_PCA...ipynb`. Exemplo didático com **Iris** + exercício com **`players_22.csv`** (FIFA 22: 19.239 jogadores, 110 colunas).

**PCA (Principal Component Analysis)** é uma técnica **não supervisionada** de **redução de dimensionalidade**: transforma muitas colunas correlacionadas em poucas colunas novas ("componentes principais") que **preservam o máximo de variância** (informação) possível. Criada por Karl Pearson (1909). Conceito já introduzido na Seção 17 — aqui é aprofundado com métrica de qualidade.

### Parte A — PCA no Iris (4 → 2 dimensões)

```python
from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Target'] = iris.get('target')

features = ['sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)']
X = df[features].values
y = df['Target'].values
```

**Passo 1 — Padronizar (obrigatório antes do PCA):**

```python
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)   # média 0, desvio 1 em cada feature
```

**Por que padronizar antes do PCA?** O PCA é **sensível à escala**: ele busca as direções de maior variância. Uma feature com números grandes teria variância artificialmente maior e **dominaria** os componentes só por causa da unidade. O `StandardScaler` (normalização-z) coloca todas com média 0 e desvio 1, dando peso justo a cada uma.

**Passo 2 — Aplicar o PCA:**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)                 # quero reduzir de 4 para 2 componentes
principalComponents = pca.fit_transform(X)

df_pca = pd.DataFrame(principalComponents, columns=['PC1', 'PC2'])
```

- `n_components=2` — quantas dimensões manter. `fit_transform` calcula os componentes e projeta os dados neles. As 4 medidas viram apenas **PC1 e PC2**.

**Passo 3 — Quanta informação foi preservada?**

```python
print('Variance of each component:', pca.explained_variance_ratio_)
# [0.72962445 0.22850762]
print('Total:', round(sum(pca.explained_variance_ratio_)*100, 2))
# 95.81
```

- **`explained_variance_ratio_`** — a **Razão de Variância Explicada**: quanto da variância total cada componente captura. Aqui **PC1 = 73%** e **PC2 = 23%**, somando **95,81%**. Ou seja: reduzimos de 4 para 2 colunas **perdendo só ~4% da informação**. Excelente troca.

**Intuição matemática:** o PCA encontra **autovetores** (as direções de maior variação) e seus **autovalores** (quanta informação cada direção carrega). Mantemos os autovetores de maior autovalor e descartamos os de pouca variância (redundância/ruído).

### Parte B — Exercício FIFA 22 (110 → 10 dimensões)

```python
df_fifa = pd.read_csv("players_22.csv")
df_fifa.shape          # (19239, 110)
```

Com 110 colunas, treinar um modelo (ex.: clustering) fica caro e ruidoso. Vamos usar PCA para simplificar.

**Passo 1 — Só numéricas + tratar nulos:**

```python
import numpy as np
from sklearn.impute import SimpleImputer

df_fifa_numerico = df_fifa.select_dtypes([np.number])   # descarta texto/URLs
imputer = SimpleImputer(strategy='mean')                # preenche NaN com a média
df_fifa_numerico = pd.DataFrame(
    imputer.fit_transform(df_fifa_numerico),
    columns=df_fifa_numerico.columns)
```

- `select_dtypes([np.number])` — mantém apenas colunas numéricas (PCA não opera em texto).
- O PCA **não aceita nulos**, por isso a imputação pela média é obrigatória antes.

**Passo 2 — Padronizar e escolher o nº de componentes pela variância acumulada:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_fifa_padronizado = scaler.fit_transform(df_fifa_numerico)

pca = PCA()                      # sem limitar -> calcula todos os componentes
pca.fit(df_fifa_padronizado)
variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)  # soma acumulada

limiar_de_variancia = 0.80
num_de_pca = np.argmax(variancia_cumulativa >= limiar_de_variancia) + 1
print(num_de_pca)   # 10
```

- `np.cumsum(...)` — soma **acumulada** da variância: PC1, PC1+PC2, PC1+PC2+PC3... Plotada, gera a "curva do cotovelo".
- `np.argmax(variancia_cumulativa >= 0.80) + 1` — acha o **menor número de componentes que atinge 80%** da variância. Resposta: **10 componentes** bastam para reter 80% da informação de 110 colunas → redução drástica.

**Passo 3 — Reduzir de fato:**

```python
pca = PCA(n_components=num_de_pca)                       # 10 componentes
principal_components = pca.fit_transform(df_fifa_padronizado)
```

**Quantos componentes escolher?** Regra prática: mantenha o suficiente para capturar **80%, 95% ou 99%** da variância, conforme o projeto. Mais componentes = mais informação, porém menos simplificação.

### Bônus — Teste de normalidade

```python
from scipy.stats import shapiro
for column in pca_df.columns:
    stat, p_value = shapiro(pca_df[column])
    # p > 0.05 -> parece normal ; p <= 0.05 -> não normal
```

O teste de **Shapiro-Wilk** verifica se cada componente segue distribuição normal. Isso importa porque algoritmos como o **K-Means** (Seção 7) assumem dados aproximadamente esféricos/normais — a saída do PCA costuma alimentar um clustering depois.

**Resumo da Aula 3:** padronizar → `PCA` → ler `explained_variance_ratio_` → escolher nº de componentes pela variância acumulada. Reduz dimensionalidade mantendo a informação essencial, acelerando e limpando os dados para o próximo modelo.

---

## 25. Aula 4 — Feature Scaling (Normalização e Padronização)

> Notebook `Aula 4/Normalização e padronização...ipynb` + `Churn_Modelling.csv` (9.865 clientes de banco, 14 colunas, separador `;`).

**Objetivo:** prever **churn** — se o cliente vai **deixar o banco** (`Exited` = 1) ou não (0). É uma **classificação binária**. O foco da aula é mostrar, na prática, **como o escalonamento das features muda o resultado** de um modelo.

### Por que escala importa?

"Escala" é a **amplitude** dos valores de uma coluna. No dataset, as features têm intervalos muito diferentes:

| Feature | Mínimo | Máximo |
|---------|--------|--------|
| CreditScore | 350 | 850 |
| Age | 18 | 92 |
| Tenure | 0 | 10 |
| Balance | 0 | 250.898 |
| NumOfProducts | 1 | 4 |
| EstimatedSalary | 11,58 | 199.992 |

Algoritmos baseados em **distância** (como o KNN, Seção 5) calculam a proximidade entre pontos. Sem escalonar, `Balance` (até 250 mil) **domina** o cálculo e `Age` (até 92) vira irrelevante — não porque seja menos importante, mas por causa da magnitude. Escalonar coloca todas para "falar no mesmo volume".

### 1. Codificando as categorias com LabelEncoder

```python
import pandas as pd
df = pd.read_csv("Churn_Modelling.csv", sep=";")   # atenção ao separador ;

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Surname']   = label_encoder.fit_transform(df['Surname'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender']    = label_encoder.fit_transform(df['Gender'])
```

- **`LabelEncoder`** — transforma rótulos de texto em inteiros (ex.: `France→0, Germany→1, Spain→2`). Necessário porque o computador só opera com números.

### 2. Separando X, y e treino/teste

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Exited'])   # features
y = df['Exited']                  # alvo (churn: 0 ou 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

### 3. Normalização — MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)                              # aprende min/max SÓ do treino
x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled  = scaler.transform(X_test)
```

- **MinMaxScaler (normalização)** — comprime todos os valores para o intervalo **[0, 1]**. Fórmula: `x' = (x - min) / (max - min)`.

**Por que `fit` só no treino e não no teste?** Se ajustássemos o scaler no teste, ele "veria" as estatísticas (min/max) dos dados de teste → **data leakage (vazamento)**. O teste deve simular dados nunca vistos. Então: aprende (`fit`) no treino, aplica (`transform`) em ambos com os **mesmos** parâmetros. (É exatamente o problema que o Pipeline da Seção 17 resolve automaticamente.)

### 4. Padronização — StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
x_train_standard_scaled = scaler.transform(X_train)
x_test_standard_scaled  = scaler.transform(X_test)
```

- **StandardScaler (padronização)** — deixa cada feature com **média 0 e desvio 1**. Fórmula: `z = (x - média) / desvio`.

### Normalização vs Padronização

| | MinMaxScaler (Normalização) | StandardScaler (Padronização) |
|---|-----------------------------|-------------------------------|
| Resultado | valores entre 0 e 1 | média 0, desvio 1 |
| Fórmula | `(x-min)/(max-min)` | `(x-média)/desvio` |
| Sensível a outliers? | **Sim** (min/max são extremos) | **Menos** (usa média/desvio) |
| Bom para | redes neurais, imagens (pixels/255) | SVM, regressão, PCA, KNN |

### 5. Comparando o efeito no KNN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Sem escalonar
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
accuracy_score(y_test, model.predict(X_test))          # 0.76

# Com normalização (MinMax)
model_min_max = KNeighborsClassifier(n_neighbors=3)
model_min_max.fit(x_train_min_max_scaled, y_train)     # ~0.81

# Com padronização (Standard)
model_standard = KNeighborsClassifier(n_neighbors=3)
model_standard.fit(x_train_standard_scaled, y_train)   # ~0.81
```

| Cenário | Acurácia |
|---------|----------|
| **Sem** escalonamento | 0.76 |
| Com **normalização** | 0.81 |
| Com **padronização** | 0.81 |

**Conclusão:** só de escalonar as features, a acurácia do KNN subiu de **76% para 81%** — **sem trocar de modelo nem ajustar nada**. O escalonamento é um passo barato de pré-processamento com impacto real em algoritmos baseados em distância.

**Resumo da Aula 4:** codificar categorias (`LabelEncoder`) → dividir treino/teste → escalonar (`fit` só no treino, para evitar leakage) → comparar MinMax vs Standard → medir o ganho de acurácia. Nem todo modelo precisa de escala (árvores não), mas KNN, SVM e redes neurais sim.

---

## 26. Desafio — Insurance (Regressão de custos de seguro)

> Pasta `Desafio/` + `insurance.csv` (1.338 registros). **Desafio prático** para você aplicar sozinho tudo das Aulas 2 a 4.

**Objetivo:** prever o **custo do seguro de saúde** (`charges`, valor contínuo) de uma pessoa a partir de suas características. É um problema de **regressão** — o mesmo tipo da Aula 2, mas agora você conduz o projeto inteiro.

### O dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `age` | numérica | idade |
| `sex` | categórica | `female` / `male` |
| `bmi` | numérica | índice de massa corporal |
| `children` | numérica | nº de dependentes |
| `smoker` | categórica | `yes` / `no` |
| `region` | categórica | região (4 valores) |
| `charges` | numérica | **custo do seguro — o alvo (y)** |

### Roteiro sugerido (aplicando o fluxo da Seção 21)

**1. Explorar os dados**
```python
import pandas as pd
df = pd.read_csv("insurance.csv")
df.head(); df.info(); df.describe()
df.corr(numeric_only=True)['charges']   # quais numéricas mais influenciam o custo?
```
Investigue especialmente `smoker` — costuma ser a variável mais determinante no custo.

**2. Codificar as categorias**
```python
# sex e smoker são binárias -> LabelEncoder (Aula 4)
# region é nominal com 4 categorias -> One-Hot (Aula 2 múltipla)
df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
```
`pd.get_dummies` é o atalho do Pandas para One-Hot Encoding.

**3. Separar X, y e treino/teste**
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

**4. (Opcional) Escalonar** as features numéricas com `StandardScaler` — útil se testar KNN; indiferente para árvores.

**5. Treinar e comparar modelos**
```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```
Comece com `LinearRegression` (baseline) e compare com `DecisionTreeRegressor` e `RandomForestRegressor` — como na Aula 2 múltipla.

**6. Avaliar no conjunto de TESTE** (não no treino!)
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = modelo.predict(X_test)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²  :", r2_score(y_test, y_pred))
```

### Checklist de aprendizado

- [ ] Fiz EDA e olhei correlações com `charges`?
- [ ] Codifiquei **todas** as colunas de texto (`sex`, `smoker`, `region`)?
- [ ] Dividi treino/teste **antes** de escalonar (evitando *data leakage*)?
- [ ] Comparei pelo menos 2 modelos com as mesmas métricas?
- [ ] Reportei as métricas no **teste**, não no treino?

**Dica:** a variável `smoker` costuma dominar a previsão — fumantes pagam muito mais. Um bom modelo (ex.: Random Forest) tende a alcançar **R² ~0.85**. Este desafio consolida todo o pipeline: EDA → codificação → split → (escala) → modelo → avaliação.

---

# PARTE III — Machine Learning Avançado (Pós Tech DTAT — Classificação)

> Esta parte acompanha os notebooks do módulo **`machine-learning-avancado/Pos_Tech_DTAT/Fase 2/03 - Machine Learning Avançado`**. Enquanto a Parte II teve foco em **regressão**, aqui o tema central é **classificação** (prever categorias) e **agrupamento** (clustering). Percorremos, aula a aula, KNN → SVM → clustering → árvores → validação cruzada → métricas de avaliação (matriz de confusão, precision/recall/F1, ROC/AUC), sempre no mesmo estilo linha a linha.

**Regressão × Classificação (a diferença que guia toda a Parte III):**

| | Regressão (Parte II) | Classificação (Parte III) |
|---|----------------------|---------------------------|
| Alvo (y) | número **contínuo** (preço, custo) | **categoria** (fraude/não, contratado/não, tipo de bebida) |
| Exemplos de modelo | LinearRegression, DecisionTreeRegressor | KNN, SVM, DecisionTreeClassifier, RandomForestClassifier |
| Métricas | R², MAE, RMSE, MAPE | Acurácia, Precisão, Recall, F1, ROC/AUC |

---

## 27. Aula 1 — Introdução à Classificação + Pré-processamento de Categóricas

> Notebooks `Aula 1/Introdução_a_modelos_de_classificação.ipynb` (+ `gaf_esp.xlsx`) e `Aula 1/Pré_processamento_de_variáveis_categóricas.ipynb`.

### 27.1 — Case: gafanhoto × esperança (classificação com KNN)

**Problema:** um cientista mediu o **comprimento do abdômen** e o **comprimento das antenas** de insetos e quer identificar **automaticamente** se cada um é *gafanhoto* ou *esperança*. Como há uma resposta certa conhecida (a espécie), é **aprendizado supervisionado de classificação** — e o algoritmo usado é o **KNN** (revisto na Seção 5).

```python
import pandas as pd
dados = pd.read_excel('gaf_esp.xlsx')
dados.head()
dados.groupby('Espécie').describe()   # estatísticas separadas por espécie
```

- `groupby('Espécie').describe()` — agrupa as linhas por espécie e mostra média, desvio, min/max de cada medida **por grupo**. Serve para ver se as duas espécies realmente diferem nas medidas (se diferem, dá para classificar).

```python
dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas')
```

O *scatter plot* revela **dois aglomerados** de pontos — cada espécie ocupa uma região do espaço. É exatamente isso que o KNN explora: pontos próximos tendem a ser da mesma classe.

**Separação treino/teste com estratificação:**

```python
from sklearn.model_selection import train_test_split

x = dados[['Comprimento do Abdômen', 'Comprimento das Antenas']]  # features (2D)
y = dados['Espécie']                                              # alvo (categoria)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42)
```

- **`stratify=y`** — é a novidade-chave desta parte. Garante que a **proporção de cada classe** (gafanhoto/esperança) seja a **mesma** no treino e no teste. Sem isso, o sorteio aleatório poderia colocar quase todos os gafanhotos no treino e deixar o teste desbalanceado. Em classificação, `stratify=y` deve ser quase um reflexo automático. (É o mesmo espírito da amostragem estratificada por renda da Seção 23, agora aplicado à variável-alvo.)

```python
list(y_train).count('Gafanhoto')   # confere o balanceamento das classes no treino
list(y_train).count('Esperança')
```

**Treinando o KNN:**

```python
from sklearn.neighbors import KNeighborsClassifier

modelo_classificador = KNeighborsClassifier(n_neighbors=3)  # hiperparâmetro: 3 vizinhos
modelo_classificador.fit(x_train, y_train)                  # "treino" = memoriza os pontos

modelo_classificador.predict([[8, 6]])   # abdômen=8, antenas=6 -> qual espécie?
```

- `n_neighbors=3` — para classificar um inseto novo, o modelo olha os **3 pontos mais próximos** e faz uma votação (Seção 5).

**Avaliando com acurácia:**

```python
from sklearn.metrics import accuracy_score

y_predito = modelo_classificador.predict(x_test)
accuracy_score(y_true=y_test, y_pred=y_predito)   # % de acertos no teste
```

- **`accuracy_score`** — a métrica mais básica de classificação: **fração de previsões corretas** (acertos / total). Simples e intuitiva, mas cuidado: em bases **desbalanceadas** ela engana (voltamos a isso na Aula 6, Seção 32).

### 27.2 — Pré-processamento de variáveis categóricas

Modelos só operam com **números**. Quando uma coluna é texto (uma *categoria*), precisamos codificá-la. Há duas técnicas centrais — as mesmas citadas na Seção 23, aqui isoladas e comparadas.

**LabelEncoder — categoria vira um inteiro:**

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('Frutas.xlsx')
labelencoder = LabelEncoder()
df['CategoriasFrutas'] = labelencoder.fit_transform(df['Fruta'])
# maçã->0, banana->1, laranja->2 ...
```

- **`LabelEncoder`** — mapeia cada rótulo de texto para um número inteiro (`0, 1, 2, ...`). `fit_transform` aprende o mapeamento e aplica de uma vez.

**One-Hot Encoding (get_dummies) — cada categoria vira uma coluna binária:**

```python
dum_df = pd.get_dummies(df, columns=["Fruta"])
# cria colunas Fruta_maçã, Fruta_banana, Fruta_laranja com valores 0/1
```

- **`pd.get_dummies`** — atalho do Pandas para One-Hot: gera **uma coluna 0/1 por categoria**.

**Qual usar? A regra de ouro:**

| Técnica | Saída | Quando usar |
|---------|-------|-------------|
| **LabelEncoder** | 1 coluna com inteiros | alvo (y), OU categorias **binárias** (2 valores), OU categorias **com ordem** natural (baixo<médio<alto) |
| **One-Hot / get_dummies** | N colunas binárias | features **nominais sem ordem** (fruta, região, cor) |

**Por que não usar LabelEncoder em tudo?** Se codificamos `maçã=0, banana=1, laranja=2`, o modelo poderia inferir que `laranja (2) > maçã (0)` ou que `banana` é a "média" entre elas — uma ordem **inexistente** que distorce o aprendizado. O One-Hot evita isso dando a cada categoria uma dimensão independente (mesma lógica do one-hot da Seção 12).

**Resumo da Aula 1:** classificação supervisionada = prever categoria. Fluxo mínimo: codificar categorias (`LabelEncoder`/`get_dummies`) → `train_test_split` com **`stratify=y`** → `KNeighborsClassifier` → `accuracy_score`. É o esqueleto que todas as aulas seguintes expandem.

---

## 28. Aula 2 — Classificação na Prática (Fraude no Cartão + Recrutamento)

> Notebooks `Aula 2/Análise de fraude em cartão de crédito.ipynb` (+ `card_transdata.csv`, ~1 milhão de transações) e `Aula 2/Modelos de Classificação em Machine Learning.ipynb` (+ `Recrutamento.xlsx`).

### 28.1 — Detecção de fraude com KNN (dataset desbalanceado)

Diferente da detecção de fraude "por dentro" da Seção 11 (feito na mão, com Naive Bayes), aqui usamos **scikit-learn** num dataset real e enorme. As 8 features descrevem cada transação (distância de casa, razão do valor sobre o preço mediano, se foi online, se usou chip/PIN etc.) e o alvo é `fraud` (0/1).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dados = pd.read_csv('card_transdata.csv', sep=',')
dados.shape          # (~1.000.000, 8)
dados.isnull().sum() # checa nulos
dados = dados.dropna()  # remove linhas com nulos
```

**O ponto mais importante da aula — a base é desbalanceada:**

```python
Total = len(dados)
TotalFraudes = dados[dados["fraud"] == 1].fraud.count()
print("Percentual de fraudes na base:", round(TotalFraudes/Total, 2)*100, "%")   # ~9%
```

Só **~9% das transações são fraude**. Isso é típico de fraude/doença/churn e tem uma consequência séria: um modelo "burro" que **chuta sempre 'não-fraude'** já acerta ~91%. Por isso a acurácia sozinha mente — precisaremos de precision/recall (Aula 6).

```python
categorias = ["Non-Fraud", "Fraud"]
plt.pie(dados["fraud"].value_counts(), labels=categorias, autopct="%.0f%%",
        explode=(0, 0.1), colors=("g", "r"))
plt.show()
```

O gráfico de pizza deixa o desbalanceamento visível de imediato.

**EDA — transformação logarítmica e correlação:**

```python
dados_fraudes = dados[dados["fraud"] == 1]

# As 3 primeiras colunas são muito assimétricas -> log10 "encolhe" a cauda longa
for column in [0, 1, 2]:
    dados_fraudes.iloc[:, column] = np.log10(dados_fraudes.iloc[:, column])
```

- **Transformação logarítmica** — quando uma variável tem uma **cauda muito longa** (poucos valores gigantes), aplicar `log10` comprime a escala e aproxima a distribuição de uma normal, facilitando a visualização e alguns modelos.

```python
correlation_matrix = dados.corr().round(2)
sns.heatmap(correlation_matrix, annot=True, linewidths=.5)
```

O **mapa de calor de correlação** ajuda a escolher features. Aqui o notebook seleciona só 3 preditoras fortes:

```python
x = dados[['distance_from_home', 'ratio_to_median_purchase_price', 'online_order']]
y = dados['fraud']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=7)
```

**Feature scaling (obrigatório para KNN):**

```python
scaler = MinMaxScaler()          # (ou StandardScaler)
scaler.fit(x_train)              # aprende min/max SÓ do treino (evita data leakage)
x_train_escalonado = scaler.transform(x_train)
x_test_escalonado  = scaler.transform(x_test)
```

Como o KNN mede distâncias (Seção 25), escalonar é indispensável — senão `distance_from_home` (valores grandes) dominaria. **`fit` só no treino** para não vazar informação do teste.

**Escolhendo o K pelo "gráfico do erro":**

```python
error = []
for i in range(1, 10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_escalonado, y_train)
    pred_i = knn.predict(x_test_escalonado)
    error.append(np.mean(pred_i != y_test))   # taxa de erro para cada K

plt.plot(range(1, 10), error, marker='o')     # olhe onde o erro estabiliza
```

- Rodamos o KNN para **K de 1 a 9** e plotamos a taxa de erro. Escolhemos o K onde o erro é baixo e **estável** — aqui, `n_neighbors=5`. É a versão manual do que o GridSearch (Seção 18/31) automatiza.

```python
modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(x_train_escalonado, y_train)
y_predito = modelo_classificador.predict(x_test_escalonado)
print(accuracy_score(y_test, y_predito))
```

### 28.2 — Recrutamento preditivo (EDA completa + KNN vs SVM)

**Case HighTech:** prever se um candidato será **contratado** (`status`) a partir de notas acadêmicas (`ssc_p`, `hsc_p`, `degree_p`, `mba_p`, `etest_p`), experiência (`workex`), gênero, especialização e salário. É *People Analytics* — classificação binária.

**Tratando nulos com significado de negócio:**

```python
import missingno as msno
msno.matrix(dados)          # visualiza onde estão os nulos
dados.isnull().sum()

dados['salary'].fillna(value=0, inplace=True)   # nulo em salary = não contratado -> 0
```

- **`missingno`** — biblioteca que **desenha** o mapa de valores ausentes. Ótima para enxergar padrões de nulos.
- Aqui o nulo em `salary` **não é erro**: quem não foi contratado não tem salário. Preencher com **0** é a decisão correta de negócio (não imputar a média!). Lição: **entender o porquê do nulo** antes de tratá-lo.

**EDA rica (boxplot, histograma, swarm, violin, pairplot):**

```python
import seaborn as sb
sb.boxplot(x='status', y='salary', data=dados)          # salário por status
sb.swarmplot(data=dados, x="mba_p", y="status", hue="workex")  # nota MBA × experiência
sb.pairplot(dados, vars=['ssc_p','hsc_p','degree_p','mba_p','etest_p'], hue="status")
```

- **`boxplot`** — mostra mediana, quartis e **outliers** de uma variável. Ideal para comparar distribuições entre classes.
- **`swarmplot` / `violinplot`** — revelam a densidade dos pontos; a análise conclui que **MBA alto + experiência** puxam a contratação, e que os **maiores salários foram para homens** (viés de gênero detectado nos dados).
- **`pairplot`** — cruza todas as variáveis 2 a 2, colorindo por classe; notas altas de ensino médio/graduação aparecem associadas à contratação.

**Codificando (LabelEncoder para binárias + One-Hot para nominais):**

```python
from sklearn.preprocessing import LabelEncoder

# binárias/ordinais -> LabelEncoder
for col in ['gender', 'workex', 'specialisation', 'status']:
    dados[col] = LabelEncoder().fit_transform(dados[col])

# nominais com 3+ categorias -> One-Hot
dummy_hsc   = pd.get_dummies(dados['hsc_s'],    prefix='dummy')
dummy_degree= pd.get_dummies(dados['degree_t'], prefix='dummy')
dados_coded = pd.concat([dados, dummy_hsc, dummy_degree], axis=1)
dados_coded.drop(['hsc_s','degree_t','salary'], axis=1, inplace=True)
```

Combina as duas técnicas da Aula 1 no mesmo dataset — exatamente a regra de ouro em ação. Só **depois de tudo numérico** a matriz de correlação fica completa e revela: `ssc_p`, `hsc_p`, `degree_p` e `workex` são as mais correlacionadas com `status`.

> **Correlação não é causalidade!** — a aula reforça esse alerta clássico. Uma feature correlacionada com o alvo é uma **candidata** a boa preditora, não uma prova de causa.

**Comparando dois modelos — KNN vs SVM:**

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(); scaler.fit(x_train)
x_train_esc = scaler.transform(x_train); x_test_esc = scaler.transform(x_test)

# Modelo 1: KNN (K escolhido pelo gráfico de erro)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_esc, y_train)
print(accuracy_score(y_test, knn.predict(x_test_esc)))

# Modelo 2: SVM linear dentro de um Pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
svm = Pipeline([("linear_svc", LinearSVC(C=1))])
svm.fit(x_train_esc, y_train)
print(accuracy_score(y_test, svm.predict(x_test_esc)))
```

- **SVM** (Seção 6) busca o hiperplano de maior margem entre as classes. O **`C`** é o hiperparâmetro de regularização: `C` alto = margem mais "dura" (menos erros no treino, risco de overfitting); `C` baixo = margem mais "mole" (mais tolerante, generaliza melhor).
- Envolver o modelo num **`Pipeline`** (Seção 17) padroniza o fluxo. **Comparar modelos com a mesma métrica** é o hábito profissional que a aula ensina.

**Resumo da Aula 2:** EDA aprofundada → tratar nulos com regra de negócio → codificar (Label + One-Hot) → escalonar (fit só no treino) → escolher K pelo gráfico de erro → **comparar KNN vs SVM**. E a lição central: em bases **desbalanceadas** (fraude ~9%), acurácia não basta.

---

## 29. Aula 3 — Aprendizado Não Supervisionado (Clusterização + Segmentação de Imagens)

> Notebooks `Aula 3/...Clusterização.ipynb` (+ `mall.csv`, clientes de shopping) e `Aula 3/...Segmentação de Imagens de Câncer de Mama.ipynb` (dataset mini-MIAS, imagens `.pgm`).

Aqui **não há rótulos (y)**. O objetivo é **descobrir grupos** nos dados sozinho — aprendizado **não supervisionado** (Seção 7). Dois algoritmos: **K-Means** e **DBSCAN**.

### 29.1 — Segmentação de clientes com K-Means

**Case:** agrupar clientes de um shopping por **renda anual** e **pontuação de gastos** (`Spending Score`), para o marketing tratar cada perfil de forma diferente.

```python
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

dados = pd.read_csv("mall.csv", sep=',')
dados.isnull().sum()
sns.pairplot(dados, hue="Gender")   # descobre que Income × Spending forma grupos visíveis
```

**Feature scaling antes de clusterizar:**

```python
scaler = StandardScaler()
scaler.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
dados_Escalonados = scaler.transform(dados[['Annual Income (k$)','Spending Score (1-100)']])
```

Como o K-Means é baseado em **distância**, escalonar é importante para que renda (0–137k) e score (1–100) tenham peso justo.

**Rodando o K-Means:**

```python
kmeans = KMeans(n_clusters=6, random_state=0)
kmeans.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])

centroides    = kmeans.cluster_centers_   # posição do centro de cada grupo
kmeans_labels = kmeans.predict(dados[['Annual Income (k$)','Spending Score (1-100)']])
```

- `cluster_centers_` — as coordenadas dos centros; `predict` devolve a qual grupo cada cliente pertence.

**Quantos grupos usar? O Método do Cotovelo (Elbow):**

```python
sse = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
    sse.append(km.inertia_)   # inertia_ = soma dos erros quadráticos (SSE) intra-cluster

plt.plot(range(1, 10), sse, '-o')
plt.xlabel('Número de clusters'); plt.ylabel('Inércia')
```

- **`inertia_` (SSE)** — soma das distâncias ao quadrado de cada ponto ao seu centróide. Sempre **diminui** ao aumentar K. Plotando SSE × K, procuramos o **"cotovelo"**: o ponto onde a curva "dobra" e o ganho passa a ser pequeno. Aqui, **K=5** é o cotovelo.

**Analogia:** é como decidir quantas gavetas usar para organizar objetos. Uma gaveta só é ruim; a cada gaveta nova a organização melhora — mas a partir de certo ponto (o cotovelo) uma gaveta a mais quase não ajuda. Pare aí.

### 29.2 — DBSCAN (agrupamento por densidade)

```python
dbscan = DBSCAN(eps=10, min_samples=8)
dbscan.fit(dados[['Annual Income (k$)','Spending Score (1-100)']])
dbscan_labels = dbscan.labels_   # -1 = outlier (ruído)
```

- **DBSCAN** agrupa por **densidade** (regiões com muitos pontos próximos), em vez de distância a um centro. Dois hiperparâmetros: **`eps`** (raio de vizinhança) e **`min_samples`** (mínimo de pontos para formar um núcleo).
- **Vantagem-chave:** ele **descobre outliers** automaticamente, marcando-os com o rótulo **`-1`** — não força todo ponto para dentro de um grupo, como o K-Means faz.

| | K-Means | DBSCAN |
|---|---------|--------|
| Base do agrupamento | distância ao centróide | densidade de pontos |
| Precisa definir nº de grupos? | **Sim** (K) | Não (emerge dos dados) |
| Detecta outliers? | Não | **Sim** (rótulo -1) |
| Formato dos grupos | esféricos/globulares | qualquer formato |
| Fraqueza | sensível a outliers e a K | difícil achar `eps`; sofre em alta dimensão |

### 29.3 — Como validar um clustering (sem rótulos)?

Sem `y` verdadeiro, como saber se o agrupamento é bom? Duas métricas:

```python
# Interna: mede a qualidade do formato dos clusters (-1 ruim ... +1 ótimo)
silhouette_score(dados[['Annual Income (k$)','Spending Score (1-100)']], kmeans_labels)

# Externa: compara dois agrupamentos entre si (0 = diferentes, 1 = idênticos)
adjusted_rand_score(kmeans_labels, dbscan_labels)
```

- **Silhouette Score (interna)** — para cada ponto, compara o quão perto ele está do próprio grupo versus do grupo vizinho. **Perto de +1** = clusters bem separados e coesos; **perto de 0** = grupos sobrepostos; **negativo** = pontos provavelmente no grupo errado.
- **Adjusted Rand Index (externa)** — mede o quanto **dois agrupamentos concordam**. Útil para comparar K-Means × DBSCAN, ou comparar com rótulos conhecidos (quando existem).

### 29.4 — K-Means como filtro de segmentação de imagem (mini-MIAS)

Uma aplicação criativa: usar K-Means para **segmentar mamografias** — agrupar pixels de intensidade parecida, realçando o contraste entre tecidos.

```python
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg

img = mpimg.imread('mdb001.pgm')   # imagem em tons de cinza (matriz de pixels)

def filtro_kmeans(img, clusters):
    vectorized = img.reshape((-1, 1))                 # achata a imagem num vetor de pixels
    kmeans = KMeans(n_clusters=clusters, random_state=0, n_init=5)
    kmeans.fit(vectorized)
    centers = np.uint8(kmeans.cluster_centers_)       # a "cor" representante de cada grupo
    segmented = centers[kmeans.labels_.flatten()]     # troca cada pixel pela cor do seu grupo
    return segmented.reshape((img.shape))             # remonta a imagem

img_segmentada = filtro_kmeans(img, clusters=3)
```

**Linha a linha:**
- `img.reshape((-1, 1))` — transforma a matriz 2D de pixels numa **lista de valores** (cada pixel é uma "amostra" com 1 feature: sua intensidade).
- `KMeans(n_clusters=3)` — agrupa os pixels em **3 níveis de intensidade**. `n_init=5` roda o K-Means 5 vezes com sementes diferentes e fica com o melhor resultado.
- `centers[kmeans.labels_.flatten()]` — substitui cada pixel pela **cor média do seu grupo** → a imagem fica "posterizada" em 3 tons, realçando regiões.
- `reshape((img.shape))` — devolve o vetor ao formato original de imagem.

**A grande ideia:** uma imagem é só uma matriz de números, então **um algoritmo de dados tabulares (K-Means) vira um filtro de imagem**. É o mesmo princípio do MNIST (Seção 12): pixels são dados.

**Resumo da Aula 3:** sem rótulos → **K-Means** (defina K pelo **cotovelo/`inertia_`**) ou **DBSCAN** (densidade, acha outliers) → valide com **silhouette** (interna) e **adjusted rand** (externa). E clustering serve até para **segmentar imagens**.

---

## 30. Aula 4 — Modelos Baseados em Árvores (Decision Tree, Random Forest, SMOTE)

> Notebooks `Aula 4/Classificando bebidas.ipynb` (+ `caffeine.csv`) e `Aula 4/Modelos baseados em árvores.ipynb` (+ `card_transdata.csv`).

Modelos de árvore aprendem **regras de decisão** ("se cafeína > X e volume < Y, então é energético"). Grande vantagem sobre KNN/SVM: **não precisam de feature scaling** e são **fáceis de interpretar** (dá para desenhar a árvore).

### 30.1 — Árvore de Decisão para classificar bebidas

**Case:** classificar o **tipo** de bebida (Café, Energético, Refrigerante, Chá, Água...) a partir de volume, calorias e cafeína.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

x = dados.drop(columns=['type', 'drink'])   # features numéricas
y = dados['type']                           # alvo (6 categorias)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=7)

dt = DecisionTreeClassifier(random_state=7, criterion='gini', max_depth=3)
dt.fit(x_train, y_train)
y_predito = dt.predict(x_test)
```

- **`criterion='gini'`** — o critério que decide **qual pergunta fazer em cada nó**. O *índice de Gini* mede a "impureza" de um nó (quão misturadas estão as classes); a árvore escolhe a divisão que mais **reduz a impureza**. A alternativa é `'entropy'` (ganho de informação) — resultados semelhantes.
- **`max_depth=3`** — limita a **profundidade** da árvore. É o principal controle de **overfitting**: árvores muito profundas decoram o treino. Rasas demais fazem underfitting.

**Visualizando a árvore (a superpotência da interpretabilidade):**

```python
class_names = ['Coffee','Energy Drinks','Energy Shots','Soft Drinks','Tea','Water']
label_names = ['Volume (ml)','Calories','Caffeine (mg)']

tree.plot_tree(dt, feature_names=label_names, class_names=class_names, filled=True)
```

- `plot_tree` **desenha** a árvore inteira: cada nó mostra a pergunta, o Gini, quantas amostras passaram e a classe majoritária. Você **lê** exatamente por que o modelo decidiu — algo impossível numa rede neural.

### 30.2 — O problema do desbalanceamento e o SMOTE

```python
dados['type'].value_counts()   # algumas bebidas têm muitos exemplos, outras pouquíssimos
```

As classes são **desbalanceadas** (ex.: muitos refrigerantes, poucos "energy shots"). Uma árvore simples tende a **ignorar a classe minoritária**. Duas soluções:

**Solução 1 — `class_weight='balanced'`:**

```python
rf = RandomForestClassifier(criterion='entropy', n_estimators=80, max_depth=7,
                            class_weight='balanced', random_state=7)
```

- **`class_weight='balanced'`** — o modelo dá **mais peso aos erros na classe minoritária**, ajustando automaticamente pela frequência. Punir mais o erro na classe rara faz o modelo prestar atenção nela.

**Solução 2 — SMOTE (oversampling sintético):**

```python
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
x_train_os, y_train_os = oversample.fit_resample(x_train, y_train)
# x_train_os agora tem as classes equilibradas (mais linhas que o original)
```

- **SMOTE** (*Synthetic Minority Over-sampling Technique*) — em vez de só copiar exemplos da classe rara, ele **cria exemplos sintéticos novos** interpolando entre vizinhos existentes dela. Resultado: classes equilibradas sem duplicação boba.
- **⚠️ Regra de ouro:** aplique SMOTE **só no treino** (`x_train`), nunca no teste — o teste deve refletir a realidade desbalanceada. Fazer oversample antes do split é *data leakage*.

**Analogia (SMOTE):** se você tem 100 fotos de gatos e só 10 de cachorros, o SMOTE "desenha" cachorros plausíveis parecidos com os 10 que existem, até chegar a 100 — em vez de xerocar os mesmos 10 dez vezes.

### 30.3 — Random Forest = floresta de árvores

```python
rf = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=7)
rf.fit(x_train, y_train)
y_predito_rf = rf.predict(x_test)

rf.estimators_        # a lista de árvores individuais que compõem a floresta
rf.score(x_train, y_train); rf.score(x_test, y_test)  # compara treino vs teste
```

- **Random Forest** (Seção 8) treina **várias árvores** em amostras aleatórias e combina os votos → mais robusto e menos propenso a overfitting que uma árvore só. `n_estimators` = quantas árvores.
- **`rf.estimators_`** — permite até **desenhar cada árvore** da floresta individualmente (`tree.plot_tree(rf.estimators_[0], ...)`).
- Comparar `score` no **treino vs teste** é o teste caseiro de overfitting: se o treino é ~1.0 e o teste bem menor, a floresta decorou.

**Decision Tree × Random Forest:**

| | Decision Tree | Random Forest |
|---|---------------|---------------|
| Estrutura | 1 árvore | N árvores votando |
| Interpretabilidade | **altíssima** (desenha 1 árvore) | menor (é um comitê) |
| Overfitting | propenso | **mais resistente** |
| Robusto a desbalanceamento | pouco | mais (esp. com `class_weight`) |

**Resumo da Aula 4:** árvores aprendem regras legíveis, **dispensam scaling** e se controlam com `max_depth`/`criterion`. Para classes desbalanceadas: **`class_weight='balanced'`** ou **SMOTE (só no treino)**. Random Forest = várias árvores → mais estabilidade.

---

## 31. Aula 5 — Validação Cruzada e Busca de Hiperparâmetros

> Notebook `Aula 5/Validação_Cruzada.ipynb`. Dataset **Vertebral Column** (openml id=1523): 310 pacientes, 6 atributos biomecânicos, 3 diagnósticos (Hérnia de Disco, Normal, Espondilolistese).

Esta aula responde: *"como ter confiança de que a métrica do modelo é real, e não sorte de uma única divisão treino/teste?"* — com **validação cruzada** (introduzida na Seção 18, aqui como protagonista).

```python
from sklearn.datasets import fetch_openml
dados = fetch_openml(data_id=1523)
tabela = pd.DataFrame(data=dados['data'])
tabela['diagnostic'] = [ {'1':'Disk Hernia','2':'Normal','3':'Spondylolisthesis'}[t]
                         for t in dados.target ]
```

- **`fetch_openml`** — baixa datasets prontos direto do repositório **OpenML** (como o `load_iris`, mas de um catálogo online gigante).

**Removendo outlier detectado na EDA:**

```python
tabela.loc[tabela['V6'] > 400]                 # inspeciona o ponto suspeito
tabela.drop(tabela.loc[tabela['V6'] > 400].index, inplace=True)  # remove
```

Um único registro com `V6 > 400` destoa de todos — provável erro de medição. Removê-lo evita que ele distorça o modelo.

**Baseline com KNN (split simples):**

```python
scaler = MinMaxScaler(); scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train); x_test_scaled = scaler.transform(x_test)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(x_train_scaled, y_train)
y_predito = modelo.predict(x_test_scaled)
```

### 31.1 — Validação Cruzada K-Fold

```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True)        # 5 divisões, embaralhando os dados
result = cross_val_score(modelo, x, y, cv=kfold)
print("Scores:", result)
print("Média:", result.mean())
```

- **`KFold(n_splits=5, shuffle=True)`** — parte os dados em **5 blocos (folds)**. `shuffle=True` embaralha antes (importante se os dados vierem ordenados por classe).
- **`cross_val_score`** — treina e avalia o modelo **5 vezes**, cada vez testando num fold diferente e treinando nos outros 4. A **média** dos 5 scores é uma estimativa muito mais **honesta** do desempenho do que uma única divisão (diagrama na Seção 18).

**Por que importa:** uma acurácia de "0.85" numa única divisão pode ter sido sorte. Se os 5 folds dão `[0.84, 0.86, 0.83, 0.85, 0.87]`, você **confia** no ~0.85. Se dão `[0.70, 0.95, 0.60, 0.90, 0.80]`, o modelo é instável — sinal de alerta que o split único esconderia.

### 31.2 — GridSearchCV: achar os melhores hiperparâmetros

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

param_grid = {
    'n_neighbors': [8, 14],                          # nº de vizinhos
    'weights':     ['uniform', 'distance'],          # vizinhos votam igual ou por proximidade
    'metric':      ['cosine', 'euclidean', 'manhattan']  # como medir distância
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,
                    scoring=make_scorer(accuracy_score), cv=5, n_jobs=4, verbose=3)
grid.fit(x_train_scaled, y_train)
print('Melhores parâmetros:', grid.best_params_)
grid.cv_results_   # detalhe de TODAS as combinações testadas
```

- **GridSearchCV** (Seção 18) testa **todas as combinações** da grade, cada uma avaliada por validação cruzada (`cv=5`), e retorna a campeã em `best_params_`. Aqui são 2×2×3 = **12 combinações × 5 folds = 60 treinos**.
- Novos hiperparâmetros do KNN que aparecem aqui:
  - **`weights='distance'`** — vizinhos mais próximos **pesam mais** no voto (mais justo que `'uniform'`, onde todos os K votam igual).
  - **`metric`** — a fórmula de distância: `euclidean` (reta), `manhattan` (em grade/quarteirões), `cosine` (ângulo entre vetores).
- **`n_jobs=4`** — usa 4 núcleos da CPU em paralelo para acelerar a busca.

### 31.3 — Comparando vários algoritmos de uma vez

```python
def AplicaValidacaoCruzada(x, y):
    kfold = KFold(n_splits=10, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=8, metric='euclidean', weights='distance')
    svm = SVC()
    rf  = RandomForestClassifier(random_state=7)

    knn_r = cross_val_score(knn, x, y, cv=kfold).mean()
    svm_r = cross_val_score(svm, x, y, cv=kfold).mean()
    rf_r  = cross_val_score(rf,  x, y, cv=kfold).mean()

    modelos = {"KNN": knn_r, "SVM": svm_r, "RF": rf_r}
    melhor = max(modelos, key=modelos.get)
    print(f"Melhor modelo: {melhor} ({modelos[melhor]:.3f})")
```

Uma função que avalia **KNN, SVM e Random Forest** com a **mesma validação cruzada de 10 folds** e elege o vencedor pela média. É o método correto de **seleção de modelo**: comparar candidatos sob a mesma régua estatística, não por um único palpite.

**Resumo da Aula 5:** não confie em um split único. Use **`cross_val_score` + `KFold`** para medir com confiança, **`GridSearchCV`** para otimizar hiperparâmetros (`n_neighbors`, `weights`, `metric`) e uma **comparação por validação cruzada** para escolher o melhor algoritmo.

---

## 32. Aula 6 — Avaliação: Matriz de Confusão e Classification Report

> Notebook `Aula 6/Validando o case Análise de fraude...ipynb` (+ `card_transdata.csv`). É a **continuação direta da Aula 2**: mesmo modelo de fraude, agora avaliado **do jeito certo**.

A grande lição: em base desbalanceada (fraude ~9%), **acurácia mente**. Precisamos abrir os **tipos de acerto e erro**.

### 32.1 — A Matriz de Confusão

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

matriz = confusion_matrix(y_test, y_predito)
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predição"); plt.ylabel("Dados Reais")
```

A **matriz de confusão** cruza o real com o previsto. Para fraude (classe positiva = 1):

|  | Previsto: Não-Fraude | Previsto: Fraude |
|---|---|---|
| **Real: Não-Fraude** | ✅ Verdadeiro Negativo (VN) | ❌ Falso Positivo (FP) |
| **Real: Fraude** | ❌ Falso Negativo (FN) | ✅ Verdadeiro Positivo (VP) |

- **Falso Positivo (FP):** transação legítima classificada como fraude → cliente bloqueado à toa (irritante).
- **Falso Negativo (FN):** fraude que passou como legítima → **prejuízo real** (o erro mais caro aqui).

**A matriz mostra o que a acurácia esconde:** um modelo pode ter 91% de acurácia e mesmo assim **deixar passar quase toda fraude** (muitos FN). A matriz revela isso na hora.

### 32.2 — Classification Report: Precision, Recall, F1

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_predito))
```

Isso imprime, **por classe**, três métricas derivadas da matriz:

| Métrica | Fórmula | Pergunta que responde | Analogia |
|---------|---------|------------------------|----------|
| **Precision** (Precisão) | VP / (VP + FP) | "Dos que **previ** como fraude, quantos **eram** fraude?" | evita alarme falso |
| **Recall** (Revocação) | VP / (VP + FN) | "Das fraudes **reais**, quantas eu **peguei**?" | evita deixar passar |
| **F1-score** | média harmônica de precision e recall | equilíbrio entre os dois | nota única de balanço |

**O trade-off que define o projeto:**
- Em **fraude/doença**, priorize **recall** — é melhor um alarme falso (FP) do que deixar passar uma fraude/um câncer (FN).
- Em **filtro de spam**, priorize **precision** — é melhor deixar passar um spam (FN) do que mandar um e-mail importante para a lixeira (FP).
- **F1** é o resumo quando você quer equilíbrio.

**Por que a média harmônica no F1?** Ela só é alta quando **ambos** (precision e recall) são altos. Se um deles é baixo, o F1 despenca — impede o modelo de "trapacear" maximizando só um.

**support** (também no report) = quantos exemplos reais de cada classe existem no teste — lembra o desbalanceamento a cada avaliação.

**Resumo da Aula 6:** troque acurácia por **matriz de confusão** + **classification report**. Entenda **FP vs FN** no contexto do problema e escolha otimizar **precision** (evitar alarme falso) ou **recall** (não deixar passar) — o **F1** equilibra os dois.

---

## 33. Aula 7 — Curva ROC e AUC

> Notebook `Aula 7/Curva ROC AUC.ipynb` (+ `Recrutamento.xlsx`). Retoma o case de recrutamento da Aula 2 e adiciona a métrica mais robusta para classificação binária.

Todo classificador binário na verdade produz uma **probabilidade** (ex.: "72% de chance de ser contratado") e aplica um **limiar** (threshold, padrão 0.5) para decidir a classe. A **curva ROC** pergunta: *e se variássemos esse limiar?*

```python
# 1. Pegar a PROBABILIDADE da classe positiva, não a classe pronta
y_prob = modelo_classificador.predict_proba(x_test_escalonado)[:, 1]
```

- **`predict_proba(...)[:, 1]`** — em vez de `predict` (que já entrega 0/1), pegamos a **probabilidade** da classe 1. É isso que a ROC precisa.

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='red', label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')      # linha do "chute aleatório"
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
```

**O que os eixos significam:**
- **TPR (True Positive Rate)** = Recall = das positivas reais, quantas peguei. Queremos **alto**.
- **FPR (False Positive Rate)** = das negativas reais, quantas marquei errado como positivas. Queremos **baixo**.
- A **curva ROC** traça TPR × FPR para **todos os limiares possíveis**. Cada ponto da curva é um threshold diferente.

**AUC (Area Under the Curve) — a nota final:**

| AUC | Interpretação |
|-----|---------------|
| **1.0** | classificador perfeito |
| **0.9–1.0** | excelente |
| **0.7–0.9** | bom |
| **0.5** | **inútil** — igual a jogar uma moeda (a linha tracejada diagonal) |
| **< 0.5** | pior que o acaso (está invertendo as classes) |

- **`auc(fpr, tpr)`** — calcula a **área sob a curva ROC**. Intuição: é a probabilidade de o modelo dar uma **nota (probabilidade) maior a um positivo aleatório do que a um negativo aleatório**. Quanto mais a curva "abraça" o canto superior esquerdo, maior a AUC.

**Por que AUC é tão usada?** Diferente da acurácia, ela **independe do limiar escolhido** e é **robusta ao desbalanceamento** — resume em um só número a capacidade do modelo de **separar** as classes. Ótima para **comparar modelos** (o de maior AUC ordena melhor os casos).

**Analogia:** a ROC é como avaliar um segurança de boate por **todos os níveis de rigor** ao mesmo tempo, do "deixa todo mundo entrar" ao "não deixa ninguém". A AUC resume: independentemente do rigor, quão bem ele **separa** quem deveria entrar de quem não deveria?

**Resumo da Aula 7:** para classificação binária, use `predict_proba` → **`roc_curve`** → **`auc`**. A curva ROC mostra o trade-off TPR×FPR em todos os limiares; a **AUC** dá uma nota única, robusta a desbalanceamento e ideal para comparar modelos.

---

## 34. Desafio — HR Analytics (previsão de abandono/churn)

> Pasta `Desafio/` + `HR_Abandono.csv` (separador `;`, decimal `,`). **Desafio prático** para aplicar sozinho toda a Parte III.

**Objetivo:** prever se um funcionário vai **deixar a empresa** (`left` = 1) ou não (0) — uma **classificação binária** de *churn* (rotatividade). É o mesmo tipo de problema da fraude e do recrutamento, agora conduzido inteiramente por você.

### O dataset

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `satisfaction_level` | numérica | nível de satisfação (0–1) |
| `last_evaluation` | numérica | nota da última avaliação (0–1) |
| `average_montly_hours` | numérica | média de horas mensais |
| `time_spend_company` | numérica | anos de empresa |
| `Work_accident` | binária | sofreu acidente de trabalho (0/1) |
| `promotion_last_5years` | binária | foi promovido nos últimos 5 anos (0/1) |
| `num_project` | numérica | nº de projetos |
| `salary` | **categórica** | `low` / `medium` / `high` |
| `depto` | **categórica** | departamento (sales, technical, ...) |
| `left` | binária | **abandonou a empresa — o alvo (y)** |

### Roteiro sugerido (juntando Aulas 1 a 7)

**1. Ler o CSV com o separador e decimal certos** (atenção — não é o padrão!):
```python
import pandas as pd
df = pd.read_csv("HR_Abandono.csv", sep=";", decimal=",")
df = df.drop(columns=["id"])   # id é só identificador, não é feature
df.info(); df.isnull().sum()
```

**2. EDA — entender quem sai** (Aula 2):
```python
df['left'].value_counts(normalize=True)   # qual o % de churn? a base é desbalanceada?
df.groupby('left').mean(numeric_only=True) # satisfação/horas diferem entre quem fica e quem sai?
import seaborn as sns
sns.heatmap(df.corr(numeric_only=True), annot=True)
```
Investigue `satisfaction_level` — costuma ser a variável mais associada ao abandono.

**3. Codificar as categorias** (Aula 1):
```python
# salary tem ordem (low < medium < high) -> mapear/LabelEncoder
df['salary'] = df['salary'].map({'low': 0, 'medium': 1, 'high': 2})
# depto é nominal -> One-Hot
df = pd.get_dummies(df, columns=['depto'], drop_first=True)
```

**4. Separar X, y e treino/teste com estratificação** (Aula 1):
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=['left']); y = df['left']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

**5. (Se usar KNN/SVM) escalonar** — `fit` só no treino (Aulas 2 e 4). Para árvores, pule.

**6. Treinar e comparar modelos** (Aulas 4 e 5):
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(random_state=7, class_weight='balanced')
print(cross_val_score(rf, X_train, y_train, cv=5).mean())   # valida antes
```
Compare pelo menos **Random Forest × KNN** com validação cruzada.

**7. Avaliar DO JEITO CERTO no teste** (Aulas 6 e 7):
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))
```

### Checklist de aprendizado

- [ ] Li o CSV com `sep=";"` **e** `decimal=","`?
- [ ] Removi o `id` e codifiquei `salary` (ordinal) e `depto` (One-Hot)?
- [ ] Usei **`stratify=y`** no split (a base de churn é desbalanceada)?
- [ ] Validei com **cross-validation** antes de confiar num número?
- [ ] Avaliei com **matriz de confusão + classification report + AUC**, não só acurácia?
- [ ] Pensei no trade-off: é pior um FP (reter quem ia ficar) ou um FN (perder quem ia sair)?

**Dica:** Random Forest costuma ir muito bem aqui (AUC alta). Mas o valor do projeto está em **explicar** *por que* as pessoas saem (baixa satisfação, muitas horas, sem promoção) — porque isso vira ação de RH. Este desafio consolida toda a Parte III: EDA → codificação → split estratificado → modelo de árvore → **validação cruzada** → avaliação com matriz de confusão, F1 e ROC/AUC.

---

## Glossário da Parte III (Classificação)

| Termo | Definição |
|-------|-----------|
| **Classificação** | Prever uma **categoria** (vs. regressão, que prevê número) |
| **stratify=y** | No split, mantém a mesma proporção de classes no treino e teste |
| **LabelEncoder** | Codifica categoria em inteiro (bom para alvo, binárias ou ordinais) |
| **One-Hot / get_dummies** | Uma coluna binária por categoria (para nominais sem ordem) |
| **Desbalanceamento** | Uma classe é muito mais frequente que a outra (fraude ~9%) |
| **SMOTE** | Cria exemplos sintéticos da classe minoritária (só no treino!) |
| **class_weight='balanced'** | Pesa mais os erros na classe minoritária |
| **Decision Tree** | Modelo de regras "se-então"; interpretável; não precisa de scaling |
| **Gini / Entropy** | Critérios que medem impureza de um nó para escolher as divisões |
| **max_depth** | Profundidade máxima da árvore; principal controle de overfitting |
| **K-Means** | Clustering por distância a centróides; defina K pelo cotovelo |
| **inertia_ (SSE)** | Soma dos erros quadráticos intra-cluster; base do método do cotovelo |
| **Método do Cotovelo (Elbow)** | Escolhe K no ponto onde o ganho de reduzir SSE "dobra" |
| **DBSCAN** | Clustering por densidade; acha outliers (rótulo -1); não precisa de K |
| **eps / min_samples** | Hiperparâmetros do DBSCAN: raio e mínimo de pontos por núcleo |
| **Silhouette Score** | Validação interna de clustering (-1 ruim … +1 ótimo) |
| **Adjusted Rand Index** | Validação externa: concordância entre dois agrupamentos (0 a 1) |
| **Cross-Validation (K-Fold)** | Treina/testa K vezes em folds diferentes; média confiável |
| **GridSearchCV** | Testa todas as combinações de hiperparâmetros com validação cruzada |
| **weights / metric (KNN)** | Peso do voto por distância; fórmula de distância (euclidean, etc.) |
| **Matriz de Confusão** | Cruza real × previsto: VP, VN, FP, FN |
| **Falso Positivo (FP)** | Previu positivo, era negativo (alarme falso) |
| **Falso Negativo (FN)** | Previu negativo, era positivo (deixou passar) |
| **Precision** | VP/(VP+FP): dos previstos positivos, quantos acertei |
| **Recall (Revocação)** | VP/(VP+FN): dos positivos reais, quantos peguei |
| **F1-score** | Média harmônica de precision e recall (equilíbrio) |
| **Curva ROC** | Traça TPR × FPR para todos os limiares de decisão |
| **AUC** | Área sob a ROC; nota única robusta a desbalanceamento (0.5 = acaso, 1 = perfeito) |
| **predict_proba** | Retorna a probabilidade da classe (necessário para a ROC) |
| **cohen_kappa** | Concordância corrigida pelo acaso entre previsto e real |

---

# PARTE IV — Visão Computacional (Computer Vision)

> Esta parte acompanha os notebooks do módulo **`vision-computer/IADEVS_COMPUTERVISION`**. Enquanto as Partes I–III trabalharam com dados tabulares e texto, aqui o dado é a **imagem** — uma matriz de pixels. Percorremos, aula a aula, o processamento clássico de imagens (OpenCV) → OCR → detecção de faces → CNNs → detecção de objetos (YOLO) → geração de imagens (GAN) → aplicações em tempo real com webcam (MediaPipe).

**A ideia central da Parte IV — imagem é número:** já vimos isso no MNIST (Seção 12) e na segmentação de mamografias (Seção 29). Uma imagem em tons de cinza é uma **matriz** onde cada célula é a intensidade de um pixel (0=preto … 255=branco). Uma imagem colorida são **3 matrizes** (canais). Todo algoritmo de visão computacional é, no fundo, matemática sobre esses números.

**Panorama das abordagens desta parte:**

| Abordagem | Aula | Precisa treinar? | Para quê |
|-----------|------|------------------|----------|
| Processamento clássico (OpenCV) | 1 | Não | filtros, bordas, redimensionar, desenhar |
| OCR (Tesseract/Paddle) | 2 | Não (modelo pronto) | extrair **texto** de imagens |
| Haar Cascades | 3 | Não (classificador pronto) | detectar **faces/olhos** |
| CNN | 4 | **Sim** | **classificar** imagens |
| YOLO | 5 | Sim (fine-tuning) | **detectar objetos** (caixas) |
| GAN | 6 | Sim | **gerar** imagens novas |
| MediaPipe | Projetos | Não (modelo pronto) | **rastrear** mãos/corpo em tempo real |
| Trackers (KCF, CSRT) | 42 | Não | **seguir** um objeto ao longo do vídeo |
| Cascade Classifier (aprofundado) | 43 | Opcional (treinável) | **detectar** objetos com features Haar/LBP |

---

## 35. Aula 1 — Fundamentos de Visão Computacional com OpenCV

> Notebook `Aula_01_Introdução_à_Visão_Computacional.ipynb`. Biblioteca principal: **OpenCV** (`opencv-python`), instalada com `!pip install opencv-python`.

**O que é OpenCV:** a biblioteca-padrão de visão computacional — um "canivete suíço" para ler, transformar, filtrar, desenhar e salvar imagens/vídeos. Importada como `cv2`.

### 35.1 — Carregar e exibir uma imagem (e a pegadinha do BGR)

```python
import cv2

imagem = cv2.imread('/content/FOTO.jpg')   # lê a imagem do disco -> matriz NumPy

cv2.imshow('Imagem', imagem)   # abre uma janela (só funciona em IDE local, NÃO no Colab)
cv2.waitKey(0)                 # espera uma tecla ser pressionada (0 = espera infinita)
cv2.destroyAllWindows()        # fecha as janelas abertas
```

- `cv2.imread` — lê a imagem para uma **matriz** de pixels. Se o caminho estiver errado, retorna `None` silenciosamente (armadilha clássica).
- `cv2.imshow` + `cv2.waitKey` + `cv2.destroyAllWindows` — o trio para exibir numa **janela nativa** (IDE/desktop). **No Google Colab** isso não funciona (não há janela); usa-se `matplotlib` ou `from google.colab.patches import cv2_imshow`.

**⚠️ A pegadinha mais importante do OpenCV — ordem BGR:** o OpenCV carrega os canais de cor na ordem **B-G-R (Azul-Verde-Vermelho)**, e não RGB. Se você exibir direto no matplotlib (que espera RGB), as cores ficam **trocadas** (céu azul vira laranja). Por isso convertemos antes:

```python
import matplotlib.pyplot as plt

imagem = cv2.imread('/content/FOTO.jpg')
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)   # reordena BGR -> RGB
plt.imshow(imagem_rgb)
plt.show()
```

- **`cv2.cvtColor`** — converte entre espaços de cor. `COLOR_BGR2RGB` só reordena os canais para o matplotlib exibir cores corretas.

### 35.2 — Conversão para escala de cinza (grayscale)

```python
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
plt.imshow(imagem_gray, cmap='gray')
```

- **`COLOR_BGR2GRAY`** — transforma as 3 matrizes de cor em **uma só** (intensidade). **Por que fazer isso?** Muitos algoritmos (detecção de bordas, faces, OCR) só precisam de forma/contraste, não de cor. Grayscale reduz os dados a **1/3**, acelera o processamento e simplifica o problema. É quase sempre o primeiro passo de um pipeline de visão.

### 35.3 — Redimensionar (resize)

```python
dimensoes = (800, 600)   # (largura, altura)
imagem_redim = cv2.resize(imagem, dimensoes, interpolation=cv2.INTER_AREA)
```

- **`cv2.resize`** — muda a resolução. **Por que importa:** modelos de ML exigem que todas as imagens de entrada tenham o **mesmo tamanho** (ex.: a CNN da Aula 4 espera 28×28; o YOLO da Aula 5, 416×416). Padronizar o tamanho é pré-processamento obrigatório.
- **`interpolation`** — a fórmula para calcular os pixels novos. `INTER_AREA` é a melhor para **reduzir**; `INTER_LINEAR`/`INTER_CUBIC` para **ampliar**.

### 35.4 — Suavização com filtro Gaussiano (blur)

```python
imagem_suavizada = cv2.GaussianBlur(imagem, (15, 15), 0)
```

- **`cv2.GaussianBlur`** — "borra" a imagem calculando, para cada pixel, uma **média ponderada dos vizinhos** (peso maior no centro, forma de sino — a curva normal da Seção 11). `(15, 15)` é o tamanho do *kernel* (janela): quanto maior, mais borrado. O `0` deixa o OpenCV calcular o desvio-padrão automaticamente.
- **Por que borrar de propósito?** Para **reduzir ruído**. Ruído são variações bruscas entre pixels vizinhos que atrapalham a detecção de bordas e objetos. Suavizar antes = menos falsas bordas depois.

### 35.5 — Detecção de bordas com Canny

```python
imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
bordas = cv2.Canny(imagem_gray, 100, 200)   # dois limiares: fraco e forte
```

- **`cv2.Canny`** — o detector de bordas mais famoso. Uma **borda** é uma mudança abrupta de intensidade (a fronteira entre objeto e fundo). O resultado é uma imagem preta com as **bordas em branco**.
- Os dois números (`100`, `200`) são os limiares de **histerese**: gradientes acima de 200 são borda certa; abaixo de 100 são descartados; entre os dois, só viram borda se estiverem **conectados** a uma borda forte. Trabalha sempre sobre a imagem **em cinza**.

**Analogia:** o Canny é como um artista fazendo o **contorno** de um desenho — joga fora o preenchimento e mantém só as linhas que definem as formas. É base para detecção de objetos, contagem, medição.

### 35.6 — Desenhar formas e salvar

```python
inicio = (300, 5)      # canto superior esquerdo (x, y)
fim    = (550, 350)    # canto inferior direito (x, y)
cor    = (255, 0, 0)   # AZUL (lembre: BGR, não RGB!)
espessura = 2

imagem_ret = cv2.rectangle(imagem.copy(), inicio, fim, cor, espessura)

cv2.imwrite('/content/imagem_processada.jpg', imagem_ret)   # salva no disco
```

- **`cv2.rectangle`** — desenha um retângulo (útil para **marcar regiões de interesse**, como faces detectadas na Aula 3). Note `(255, 0, 0)` = azul, porque é **BGR**. Use `imagem.copy()` para não riscar o original.
- **`cv2.imwrite`** — grava a imagem processada em arquivo. Detecta o formato pela extensão (`.jpg`, `.png`).

**Resumo da Aula 1:** OpenCV = manipulação de imagens. Fluxo típico: `imread` → converter cor (cuidado com **BGR**) → cinza → suavizar (blur) → detectar bordas (Canny) → desenhar/marcar → `imwrite`. É o pré-processamento que alimenta tudo que vem depois.

---

## 36. Aula 2 — OCR (Reconhecimento de Texto) com Tesseract e PaddleOCR

> Notebook `Aula_02_ocr_com_tesseract_e_paddle.ipynb`. **OCR** (*Optical Character Recognition*) = extrair **texto** que está dentro de uma **imagem** (foto de documento, placa, nota fiscal).

### 36.1 — Tesseract OCR

```python
!pip install pytesseract
!apt-get install -y tesseract-ocr tesseract-ocr-por   # o motor + o idioma português

import pytesseract, cv2

img = cv2.imread('/content/ESPORTE.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

texto = pytesseract.image_to_string(img_gray, lang='por')   # OCR em português
print(texto)
```

- **Tesseract** — motor de OCR open-source (mantido pelo Google). `pytesseract` é o "invólucro" Python. É preciso instalar **o motor** (`tesseract-ocr`) **e o pacote do idioma** (`tesseract-ocr-por` para português) separadamente.
- **`image_to_string(img, lang='por')`** — recebe a imagem e devolve o texto. `lang='por'` melhora muito o reconhecimento de acentos e palavras em português.

**A grande lição da aula — o pré-processamento decide a qualidade do OCR.** Uma imagem "suja" (fundo irregular, ruído) confunde o Tesseract. A aula testa **4 técnicas** para "limpar" a imagem antes:

**(1) Thresholding fixo (binarização):**
```python
_, img_thresh = cv2.threshold(img_gray, 142, 255, cv2.THRESH_BINARY)
```
- **`cv2.threshold`** — transforma a imagem em **preto e branco puro**: todo pixel acima de 142 vira branco (255), abaixo vira preto (0). Separa o texto do fundo. Problema: o limiar `142` é **chutado na mão** e pode não servir para todas as imagens.

**(2) Otsu (limiar automático):**
```python
_, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```
- **Otsu** — calcula **sozinho** o melhor limiar, analisando o histograma da imagem para separar da melhor forma os dois grupos de pixels (texto × fundo). Tira o "chute" do passo anterior.

**(3) Limiarização adaptativa:**
```python
img_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 33, 9)
```
- **`adaptiveThreshold`** — usa um limiar **diferente para cada região** da imagem (janela de 33 px, ajuste de 9). Essencial quando a **iluminação é desigual** (uma parte do documento mais clara que a outra) — algo que o limiar único não resolve.

**(4) Remoção de ruído (median blur):**
```python
img_median = cv2.medianBlur(img_gray, 3)
```
- **`medianBlur`** — troca cada pixel pela **mediana** dos vizinhos. Excelente para remover ruído "sal e pimenta" (pontinhos pretos e brancos) sem borrar as letras tanto quanto o Gaussiano.

**Medindo objetivamente qual técnica foi melhor:**
```python
from difflib import SequenceMatcher

def compare_texts(original, ocr_text):
    return SequenceMatcher(None, original, ocr_text).ratio()   # 0 a 1 (similaridade)
```
- **`SequenceMatcher.ratio()`** — compara o texto extraído com o texto **verdadeiro (gabarito)** e devolve a **fração de similaridade** (0 a 1). Assim dá para dizer, com número, qual pré-processamento produziu o OCR mais fiel. É a mesma mentalidade de "métrica de avaliação" das partes anteriores, agora aplicada a texto.

### 36.2 — PaddleOCR + extração estruturada com regex

```python
!pip install paddleocr==2.7.3 paddlepaddle==2.6.1
from paddleocr import PaddleOCR

ocr = PaddleOCR(use_angle_cls=True, lang='en')
resultado = ocr.ocr(pdf_path)   # devolve texto + a posição (caixa) de cada trecho
```

- **PaddleOCR** — motor de OCR mais moderno (baseado em deep learning), da Baidu. Costuma ser **mais robusto** que o Tesseract em textos difíceis e já entrega a **localização** de cada palavra na imagem. `use_angle_cls=True` corrige texto **rotacionado/inclinado**.

**Do texto bruto para dados estruturados (caso real: conta de luz):**
```python
import re

padrao_valor_total = r"R\$(\d+,\d{2})"   # captura "R$123,45"
padrao_cpf_cnpj    = r"CPF/ CNPJ:\s([\d.-]+)"

def extrair(texto, padrao):
    match = re.search(padrao, texto)
    return match.group(1) if match else None

valor_total = extrair(ocr_text, padrao_valor_total)
```

- **Expressões regulares (regex)** — depois do OCR ler tudo, o `re` (Seção do módulo Python) **pesca campos específicos** (valor, vencimento, CPF, referência) por padrões de texto. É assim que se automatiza a leitura de **notas fiscais, boletos, contas** — o pipeline **OCR → regex → dados estruturados** é um dos usos comerciais mais comuns de visão computacional.

**Resumo da Aula 2:** OCR extrai texto de imagens. O segredo da **qualidade** é o **pré-processamento** (threshold, Otsu, adaptativo, median blur) — meça o ganho com `SequenceMatcher`. Tesseract é o clássico; PaddleOCR é mais robusto e localiza o texto. Junte com **regex** para transformar o texto lido em campos estruturados.

---

## 37. Aula 3 — Detecção de Faces com Haar Cascades

> Notebook `Aula_03_detecçãod_e_faces.ipynb`. Técnica **clássica** (pré-deep-learning) de detecção, rápida e leve, que roda até em hardware fraco.

```python
import cv2

# Classificadores pré-treinados que já vêm com o OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

imagem = cv2.imread('/content/FOTO.jpg')
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)   # detecção roda no cinza

# Detecta faces (retorna uma lista de caixas [x, y, largura, altura])
faces = face_cascade.detectMultiScale(imagem_cinza, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(55, 55))

# Desenha um retângulo em cada face encontrada
for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

**O que é um Haar Cascade:** um classificador que aprende a reconhecer objetos (faces, olhos) procurando **padrões de contraste** típicos — por exemplo, "a região dos olhos é mais escura que a das bochechas". Ele passa uma "janela" por toda a imagem, em **vários tamanhos**, testando esses padrões em cascata (daí *cascade*): filtros rápidos descartam logo as regiões óbvias sem rosto, e só as promissoras seguem para testes mais caros.

**Os parâmetros do `detectMultiScale` (o que ajustar):**

| Parâmetro | O que faz | Efeito |
|-----------|-----------|--------|
| **`scaleFactor=1.1`** | quanto a janela cresce a cada passada (10%) | menor = mais preciso e mais lento |
| **`minNeighbors=5`** | quantas detecções vizinhas confirmam um rosto | maior = menos falsos positivos, pode perder rostos |
| **`minSize=(55,55)`** | tamanho mínimo do objeto em pixels | ignora rostos menores que isso |

- **`cv2.data.haarcascades + '...xml'`** — o OpenCV **já vem** com classificadores prontos (rosto frontal, olhos, sorriso, etc.). Você não treina nada: apenas carrega o `.xml` e usa.
- **`detectMultiScale`** — roda o detector em **várias escalas** (por isso *MultiScale*), permitindo achar rostos grandes e pequenos na mesma foto. Devolve as coordenadas `(x, y, w, h)` de cada caixa.

> **Nota:** o mesmo mecanismo serve para **olhos** (`eye_cascade`), sorrisos, placas etc. — basta trocar o arquivo `.xml`. Haar Cascade é ótimo por ser **rápido e sem GPU**, mas é sensível a rostos de lado ou muito inclinados; para casos difíceis, hoje se usam CNNs/YOLO (Aulas 4 e 5).

**Resumo da Aula 3:** detecção de faces sem treinar nada, com **classificadores Haar prontos** do OpenCV. Converta para cinza → `detectMultiScale` (ajuste `scaleFactor`/`minNeighbors`) → desenhe as caixas. Leve, rápido, clássico.

---

## 38. Aula 4 — Redes Neurais Convolucionais (CNN)

> Notebook `Aula_04_CNN.ipynb`. A CNN é o modelo de deep learning **especializado em imagens** — o motor por trás do reconhecimento visual moderno.

**O que é uma CNN:** uma rede neural (Seções 9–13) com camadas especiais — as **convolucionais** — inspiradas no córtex visual. Em vez de olhar pixel por pixel isolado, ela aprende **filtros** que detectam padrões locais (bordas → texturas → formas → objetos), do simples ao complexo, camada após camada.

### 38.1 — A operação de convolução (na mão)

```python
import numpy as np

input_matrix = np.array([[1,2,3,0,1],
                         [4,5,6,1,2],
                         [7,8,9,0,1],
                         [0,1,2,3,4],
                         [1,2,3,4,5]])      # "imagem" 5x5 (intensidade dos pixels)

filter_matrix = np.array([[1,0,-1],
                          [1,0,-1],
                          [1,0,-1]])        # filtro/kernel 3x3 (detector de borda vertical)

output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        region = input_matrix[i:i+3, j:j+3]         # recorta um pedaço 3x3 da imagem
        output[i, j] = np.sum(region * filter_matrix)  # produto elemento a elemento e soma
```

**Como funciona a convolução (linha a linha):**
- O **filtro (kernel)** é uma pequena matriz que "desliza" sobre a imagem. Em cada posição, multiplica-se elemento a elemento o pedaço da imagem pelo filtro e **soma tudo** num único número.
- Esse número vai para o **mapa de características** (a saída). O filtro deste exemplo (`[1,0,-1]` nas linhas) responde forte onde há uma **mudança vertical** de intensidade — ou seja, **detecta bordas verticais**.
- **A mágica da CNN:** você **não** define os filtros na mão. A rede **aprende sozinha** os melhores filtros durante o treino (via backpropagation, Seção 10) — começando por detectores de borda e evoluindo para detectores de olhos, rodas, letras...

**Analogia:** o filtro é um "carimbo" que procura um padrão específico. Passe o carimbo por toda a foto; onde o padrão aparece, o resultado "acende". Uma CNN tem dezenas desses carimbos por camada, e aprende quais carimbos valem a pena.

### 38.2 — Max Pooling (reduzir dimensão)

```python
input_matrix = np.array([[1,3,2,4],
                         [5,6,1,2],
                         [3,0,2,1],
                         [1,2,3,0]])       # 4x4

output = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        region = input_matrix[i*2:i*2+2, j*2:j*2+2]  # janela 2x2 sem sobreposição
        output[i, j] = np.max(region)                # pega o MAIOR valor da janela
```

- **Max Pooling** — percorre a imagem em janelas (aqui 2×2) e mantém só o **valor máximo** de cada uma, reduzindo o tamanho pela metade. Serve para: (1) **diminuir a quantidade de dados** e cálculos; (2) manter só a característica **mais forte** de cada região; (3) dar **robustez** — se o objeto se deslocar alguns pixels, o máximo tende a ser o mesmo (invariância à translação).

### 38.3 — Montando e treinando a CNN no MNIST

```python
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normaliza: (N, 28, 28, 1), pixels de 0-255 para 0-1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images  = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

**A arquitetura camada a camada:**
| Camada | Papel |
|--------|-------|
| `Conv2D(32, (3,3), relu)` | 32 filtros 3×3 aprendem padrões simples (bordas). `input_shape=(28,28,1)` = imagem 28×28 em cinza (1 canal) |
| `MaxPooling2D((2,2))` | reduz pela metade, mantém o mais forte |
| `Conv2D(64, (3,3))` ×2 | mais filtros = padrões mais complexos (curvas, laços do dígito) |
| `Flatten()` | "achata" os mapas 2D num vetor 1D para as camadas densas |
| `Dense(64, relu)` | camada totalmente conectada que combina as características |
| `Dense(10, softmax)` | saída: 10 dígitos, probabilidades que somam 1 (Seção 9) |

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))
test_loss, test_acc = model.evaluate(test_images, test_labels)   # ~99%
```

- **`sparse_categorical_crossentropy`** — detalhe importante: é a perda para classificação multiclasse **quando os rótulos são inteiros** (`3`, `7`...). É a irmã da `categorical_crossentropy` (Seção 9), que exige rótulos em **one-hot**. Aqui, como `train_labels` são inteiros, usa-se a versão *sparse* — **evita** ter de fazer `to_categorical`. (No MNIST da Seção 12 usou-se one-hot + `categorical_crossentropy`; os dois caminhos levam ao mesmo lugar.)
- `validation_data` — mede a acurácia no teste a cada época, revelando overfitting; o `history` guarda as curvas para plotar treino × validação.

**CNN × rede densa comum (por que CNN vence em imagem):** a rede densa da Seção 12 achata a imagem logo de cara e trata cada pixel de forma isolada, ignorando a **vizinhança**. A CNN preserva a estrutura 2D e detecta padrões **locais** independentemente de **onde** aparecem na imagem — por isso generaliza muito melhor em visão.

**Resumo da Aula 4:** CNN = convolução (filtros que aprendem padrões locais) + pooling (reduz e destaca) + camadas densas (classificam). Normalize os pixels (/255), empilhe `Conv2D`/`MaxPooling2D`, finalize com `Flatten`+`Dense`+`softmax`. Use `sparse_categorical_crossentropy` com rótulos inteiros. É a base de toda visão computacional moderna.

---

## 39. Aula 5 — Detecção de Objetos com YOLOv5

> Notebook `Aula_05_Yolo.ipynb` + subprojeto `yolov5_face_mask_detection/`. Case: detectar se pessoas estão **com ou sem máscara** (`MASK` / `NO-MASK`).

**Classificação × Detecção — a diferença-chave:** a CNN da Aula 4 responde *"que dígito é esta imagem?"* (uma classe para a imagem inteira). A **detecção de objetos** responde *"quais objetos existem e ONDE?"* — desenha uma **caixa delimitadora (bounding box)** ao redor de cada objeto e o classifica. Uma foto pode ter várias pessoas, cada uma com sua caixa e rótulo.

**O que é YOLO:** *You Only Look Once*. Ao contrário de métodos antigos que varriam a imagem em milhares de pedaços, o YOLO olha a imagem **uma única vez** e prevê todas as caixas e classes de uma vez só — por isso é **rápido o bastante para vídeo em tempo real**. O `YOLOv5` (da Ultralytics) é uma implementação popular em PyTorch.

```python
# 1. Baixar o repositório e instalar dependências
!git clone https://github.com/ultralytics/yolov5
!pip install -U -r yolov5/requirements.txt

# 2. Baixar o dataset (imagens já anotadas com caixas, via Roboflow) e o arquivo de config
!wget .../Mask_Wearing...yolov5pytorch.zip
!unzip -o Mask_Wearing...zip

# 3. Treinar (fine-tuning) no dataset de máscaras
%cd /content/yolov5/
!python train.py --img 416 --batch 16 --epochs 300 \
    --data '../data.yaml' --cfg ../mask_yolov5s.yaml --weights '' \
    --name mask_yolov5s_results
```

**Os argumentos do `train.py` (linha a linha):**
| Argumento | Significado |
|-----------|-------------|
| `--img 416` | redimensiona todas as imagens para 416×416 (padroniza a entrada) |
| `--batch 16` | 16 imagens por lote antes de atualizar os pesos |
| `--epochs 300` | 300 passagens completas pelo dataset |
| `--data data.yaml` | aponta onde estão as imagens e **quais são as classes** (mask/no-mask) |
| `--cfg mask_yolov5s.yaml` | a **arquitetura** do modelo (a versão `s` = *small*, leve e rápida) |
| `--weights ''` | treina **do zero**; se apontasse `yolov5s.pt`, faria *transfer learning* |

**Anotações e o formato YOLO:** para o YOLO aprender, cada imagem de treino vem com um arquivo de **anotação** listando as caixas: `classe x_centro y_centro largura altura` (coordenadas normalizadas de 0 a 1). Ferramentas como o **Roboflow** (usado aqui) ajudam a rotular as imagens e exportar nesse formato.

**Usando o modelo treinado para detectar em uma imagem nova:**
```python
!python detect.py --weights runs/train/mask_yolov5s_results/weights/last.pt \
    --img 416 --conf 0.3 --source /content/mask.png
```
- **`detect.py`** — roda a **inferência**: aplica o modelo treinado (`--weights ...last.pt`) numa imagem/vídeo (`--source`).
- **`--conf 0.3`** — o **limiar de confiança**: só mostra detecções com ≥30% de certeza. Subir esse valor reduz falsos positivos; baixar pega mais objetos (é o mesmo trade-off do threshold da ROC, Seção 33).

**Métricas de detecção:** o YOLO reporta **mAP** (*mean Average Precision*), **precision** e **recall** (Seção 32) — mas calculadas considerando também **o quão bem a caixa prevista cobre a caixa real** (métrica **IoU**, *Intersection over Union*: a sobreposição entre as duas caixas).

**Resumo da Aula 5:** detecção de objetos = classe **+ localização (caixa)**. YOLO ("olha uma vez") é rápido para tempo real. Fluxo: dataset anotado (Roboflow) → `train.py` (ajuste `--img`, `--epochs`, `--cfg`) → `detect.py` (ajuste `--conf`). Avaliação por mAP/IoU.

---

## 40. Aula 6 — GANs (Redes Generativas Adversárias)

> Notebook `Aula_6_GAN.ipynb`. Enquanto as aulas anteriores **interpretam** imagens, a GAN as **cria**.

**O que é uma GAN:** *Generative Adversarial Network*, introduzida por **Ian Goodfellow (2014)**. São **duas redes que competem** entre si:
- **Gerador (Generator):** parte de **ruído aleatório** e tenta produzir imagens **falsas** que pareçam reais.
- **Discriminador (Discriminator):** um "detetive" que recebe imagens reais e falsas e tenta **dizer qual é qual**.

**A dinâmica adversarial (a ideia genial):** os dois treinam juntos, em oposição. O gerador melhora para **enganar** o discriminador; o discriminador melhora para **não ser enganado**. Essa "queda de braço" empurra o gerador a produzir imagens cada vez mais realistas — até ficarem quase indistinguíveis das verdadeiras.

**Analogia:** um **falsário** (gerador) tentando imitar quadros e um **perito** (discriminador) tentando flagrar as falsificações. A cada rodada, o falsário fica mais habilidoso e o perito mais criterioso. No fim, o falsário pinta obras que enganam qualquer um.

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Carrega MNIST e normaliza para [-1, 1] (combina com a saída 'tanh' do gerador)
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()   # rótulos ignorados!
train_images = (train_images.astype('float32') - 127.5) / 127.5
train_images = np.expand_dims(train_images, axis=-1)              # (N, 28, 28, 1)

def make_generator_model():
    model = models.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))  # vetor latente de 100
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))                                 # vira "imagem" pequena
    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    # ... mais uma Conv2DTranspose até chegar a 28x28x1, com ativação final 'tanh'
```

**Conceitos-chave do gerador (linha a linha):**
- **Vetor latente `(100,)`** — a "semente" de ruído aleatório. Cada vetor diferente gera uma imagem diferente. É o "DNA" da imagem falsa.
- **`Conv2DTranspose` (deconvolução)** — o **oposto** da `Conv2D`: em vez de encolher a imagem, ela **aumenta** (upsampling). O gerador vai de um vetorzinho até uma imagem 28×28 empilhando essas camadas. `strides=(2,2)` dobra o tamanho a cada passo.
- **`BatchNormalization`** — normaliza as saídas entre camadas, **estabilizando** um treino que é notoriamente instável nas GANs.
- **`LeakyReLU`** — variante da ReLU que deixa passar um pouquinho dos valores negativos (em vez de zerá-los). Ajuda o gradiente a fluir — recomendada em GANs.
- **`tanh` na saída** — espreme os pixels no intervalo **[-1, 1]**; por isso as imagens reais também foram normalizadas para [-1, 1] (com `-127.5`/`127.5`). Gerador e dados precisam falar a mesma "escala".
- Note que os **rótulos do MNIST são ignorados** (`(train_images, _)`): a GAN não classifica, ela **aprende a distribuição** das imagens para gerar novas.

**Onde se usa GAN:** gerar rostos realistas de pessoas que não existem, aumentar resolução de imagens (*super-resolution*), criar dados sintéticos para treinar outros modelos, arte, e — com evolução — a base conceitual de muita geração de imagem moderna.

### Tabela comparativa — as redes neurais do curso

| Rede | O que faz | Entrada → Saída | Aula |
|------|-----------|-----------------|------|
| **MLP / Densa** | classifica/regride dados tabulares | vetor → classe/valor | Parte I (Seções 9–10) |
| **CNN** | **interpreta** imagens | imagem → classe | 38 |
| **YOLO (CNN de detecção)** | localiza objetos | imagem → caixas + classes | 39 |
| **GAN** | **gera** imagens | ruído → imagem | 40 |

**Resumo da Aula 6:** a GAN cria imagens com duas redes em disputa — **gerador** (faz falsas a partir de ruído) e **discriminador** (tenta pegá-las). O gerador usa `Conv2DTranspose` para "crescer" a imagem, `BatchNorm`+`LeakyReLU` para estabilizar e `tanh` na saída (dados normalizados em [-1,1]). É o modelo **generativo** do curso, oposto conceitual da CNN.

---

## 41. Projetos com MediaPipe — Hand Tracking (Libras) e Análise de Agachamento

> Subprojetos `hand-tracking-libras/` e `simple-squat-analysis/`. Mostram visão computacional **em tempo real** pela webcam, sem treinar modelo nenhum, usando o **MediaPipe**.

**O que é MediaPipe:** framework do Google com modelos **prontos** que detectam, em tempo real, **pontos de referência (landmarks)** do corpo — mãos, pose, rosto. Você não treina nada: recebe as coordenadas dos pontos e constrói sua lógica em cima delas. Instala com `!pip install mediapipe`.

### 41.1 — Hand Tracking: reconhecendo vogais em Libras

O MediaPipe Hands detecta **21 pontos** por mão (base, juntas e ponta de cada dedo). A ideia do projeto: a partir de quais dedos estão abertos/fechados, **identificar a vogal** do alfabeto manual (A, E, I, O, U).

```python
import cv2, mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, max_num_hands=2,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=mode, max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # MediaPipe espera RGB
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)   # converte coord. normalizada -> pixel
                landmark_list.append([id, cx, cy])
        return landmark_list
```

**Linha a linha (o essencial):**
- `mp.solutions.hands` — o modelo pronto de detecção de mãos. `min_detection_confidence` é a certeza mínima para aceitar uma detecção (ideia igual à do Haar, Seção 37).
- `find_hands` — converte para **RGB** (o MediaPipe, ao contrário do OpenCV, espera RGB) e processa. `draw_landmarks` desenha os pontos e as conexões (aquele "esqueleto" da mão).
- `find_position` — o passo-chave: o MediaPipe devolve coordenadas **normalizadas** (0 a 1); multiplicamos por largura/altura (`lm.x * w`) para obter o **pixel** de cada ponto. Retorna uma lista `[id, x, y]` dos 21 landmarks.

```python
def identify_vowel(landmarks):
    thumb_tip = landmarks[4][1], landmarks[4][2]     # ponta do polegar
    index_tip = landmarks[8][1], landmarks[8][2]     # ponta do indicador
    # ... (12=médio, 16=anelar, 20=mínimo)

    # Um dedo está "aberto" se a ponta está acima (y menor) da junta de referência
    index_is_open = index_tip[1] < landmarks[6][2]

    def euclidean_distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) ** 0.5

    # Combinações de dedos abertos/fechados + distâncias -> cada vogal
    if thumb_is_open and index_is_open and middle_is_open \
       and not ring_is_open and not pinky_is_open:
        return 'U'
    # ... demais regras para A, E, I, O
```

**A lógica (sem machine learning!):**
- **Índices dos landmarks:** as **pontas** dos dedos são os pontos 4 (polegar), 8, 12, 16 e 20. As **juntas** intermediárias servem de referência.
- **Dedo aberto/fechado:** compara o `y` da ponta com o `y` de uma junta. Como em imagem o `y` **cresce para baixo**, uma ponta "acima" da junta (y menor) significa dedo **esticado/aberto**.
- **Distância euclidiana** (Seção 5) — mede o quão perto as pontas estão entre si (ex.: na vogal **O** os dedos se juntam formando um círculo).
- Cada vogal vira uma **combinação de regras** (`if`) sobre dedos abertos e distâncias. É visão computacional resolvendo um problema **sem treinar modelo** — só geometria sobre os landmarks.

### 41.2 — Análise de agachamento com MediaPipe Pose

Mesma ideia, agora com o **corpo inteiro**: o `mp.solutions.pose` detecta articulações (quadril, joelho, tornozelo...) e medimos o **ângulo do joelho** para avaliar a profundidade do agachamento.

```python
import cv2, mediapipe as mp, math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def calculate_angle(hip, knee, ankle):
    # vetores coxa (quadril->joelho) e canela (joelho->tornozelo)
    v1 = (knee[0]-hip[0],  knee[1]-hip[1])
    v2 = (ankle[0]-knee[0], ankle[1]-knee[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]                       # produto escalar
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    return math.degrees(math.acos(dot / (mag1 * mag2)))   # ângulo em graus

def classify_squat_depth(hip, knee, ankle):
    angle = calculate_angle(hip, knee, ankle)
    if angle > 90:   return 'Above 90 degrees'
    elif angle == 90: return 'At 90 degrees'
    else:            return 'Below 90 degrees'
```

**Como funciona:**
- `mp_pose.Pose()` — modelo pronto que devolve as articulações do corpo (`PoseLandmark.LEFT_HIP`, `LEFT_KNEE`, `LEFT_ANKLE`...), cada uma com coordenadas normalizadas convertidas para pixel.
- **`calculate_angle`** — usa **trigonometria vetorial**: monta dois vetores (coxa e canela), calcula o ângulo entre eles pelo **produto escalar** e o `arccos`. É matemática pura de ensino médio aplicada aos pontos do corpo.
- **`classify_squat_depth`** — regra de negócio: joelho abaixo de 90° = agachamento profundo. Serve para dar **feedback de exercício** em tempo real (personal trainer virtual).

**O padrão comum dos dois projetos:** MediaPipe entrega os **landmarks** → você converte para pixels → aplica **geometria** (distâncias, ângulos) → toma uma **decisão** por regras. É visão computacional prática, em tempo real e **sem treinar modelo** — porque o modelo (detecção de mãos/pose) já vem pronto.

**Resumo dos Projetos:** MediaPipe = landmarks prontos de mão/corpo em tempo real. Sobre eles, geometria simples (dedo aberto por comparação de `y`, distância euclidiana, ângulo por produto escalar) resolve problemas reais — de tradução de Libras a análise de exercício — **sem machine learning nenhum**, só OpenCV + matemática.

---

## 42. Rastreamento de Objetos (Object Tracking) — KCF, CSRT e a família do OpenCV

> Complemento prático à Parte IV. Até aqui **detectamos** objetos quadro a quadro (Haar na Seção 37, YOLO na 39). Agora vamos **rastreá-los**: seguir *o mesmo* objeto ao longo de um vídeo, mantendo sua identidade de um frame para o outro.

### 42.1 — Detecção × Rastreamento: qual a diferença?

| | Detecção (frame a frame) | Rastreamento (tracking) |
|---|--------------------------|--------------------------|
| Pergunta | "onde estão os objetos **neste** frame?" | "para onde foi **aquele** objeto que eu já marquei?" |
| Identidade | não sabe se o carro do frame 2 é o mesmo do frame 1 | **mantém o ID** do objeto entre frames |
| Custo | alto (roda o detector inteiro toda vez) | **baixo** (só procura perto de onde o objeto estava) |
| Precisa de caixa inicial? | não | **sim** — você marca o objeto uma vez |

**Por que rastrear em vez de só detectar?** Três motivos:
1. **Velocidade** — um detector como o YOLO é pesado. O tracker só examina uma pequena vizinhança ao redor da última posição, então roda **muito mais rápido** (essencial para tempo real em hardware fraco).
2. **Identidade** — a detecção pura não sabe que "o carro na posição A no frame 1" é "o carro na posição B no frame 2". O tracker preserva esse **ID**, permitindo contar, medir velocidade e traçar trajetória.
3. **Continuidade** — se o detector falhar em um frame (objeto meio escondido), o tracker **interpola** e não perde o objeto.

**Analogia:** a **detecção** é tirar uma foto e apontar "aqui tem um cachorro". O **rastreamento** é apontar o dedo para o cachorro e **acompanhá-lo com o dedo** enquanto ele corre — você não redescobre que é um cachorro a cada instante, só segue o mesmo alvo.

> **Na prática, os dois se combinam:** um sistema real usa **detecção** de vez em quando (achar objetos novos) + **rastreamento** no meio (seguir os já achados de forma barata). Esse padrão se chama *tracking-by-detection* e é a base de sistemas como o **DeepSORT**.

### 42.2 — Como funciona um tracker de correlação (a intuição)

KCF e CSRT são **correlation filters** (filtros de correlação). A ideia:
1. Você marca o objeto num frame (a caixa inicial, o *template*).
2. O tracker aprende um "filtro" que responde **forte** no centro do objeto e fraco ao redor.
3. No frame seguinte, ele **desliza** esse filtro pela vizinhança (como a convolução da Seção 38) e encontra o **pico de resposta** — ali está o objeto agora.
4. Atualiza o filtro com a nova aparência e repete.

É basicamente perguntar, a cada frame: *"em qual posição próxima o pedaço da imagem mais se parece com o objeto que eu estava seguindo?"*

### 42.3 — O fluxo de código (igual para qualquer tracker)

```python
import cv2

# 1. Criar o tracker (troque o tipo para comparar KCF vs CSRT)
tracker = cv2.TrackerCSRT_create()      # ou cv2.TrackerKCF_create()

# 2. Abrir o vídeo/webcam e ler o primeiro frame
cap = cv2.VideoCapture('video.mp4')     # ou 0 para a webcam
ok, frame = cap.read()

# 3. Marcar o objeto a seguir (uma caixa: x, y, largura, altura)
bbox = cv2.selectROI('Selecione o objeto', frame, False)  # arraste o mouse e ENTER
tracker.init(frame, bbox)               # inicializa o tracker com essa caixa

# 4. Loop: para cada frame, atualizar a posição
while True:
    ok, frame = cap.read()
    if not ok:
        break

    sucesso, bbox = tracker.update(frame)   # <- o coração do tracking

    if sucesso:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # desenha a caixa
    else:
        cv2.putText(frame, 'Objeto perdido!', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Linha a linha (o essencial):**
- **`cv2.TrackerCSRT_create()` / `cv2.TrackerKCF_create()`** — instancia o tracker. Trocar o tipo é a única mudança para comparar algoritmos.
- **`cv2.selectROI`** — abre uma janela onde você **arrasta o mouse** para marcar o objeto (a *Region Of Interest*). Retorna a caixa `(x, y, w, h)`. Em produção, essa caixa viria de um **detector** (YOLO/Haar) em vez do mouse.
- **`tracker.init(frame, bbox)`** — "ensina" ao tracker qual é o objeto (a aparência dentro da caixa).
- **`tracker.update(frame)`** — a cada novo frame, devolve `(sucesso, nova_caixa)`. Se `sucesso=False`, o tracker **perdeu** o objeto (saiu de cena, oclusão total).

> **⚠️ Detalhe de versão do OpenCV:** a partir do OpenCV 4.5.1, vários trackers migraram para o submódulo **`legacy`**. Se `cv2.TrackerKCF_create()` der erro, use `cv2.legacy.TrackerKCF_create()`. Além disso, alguns trackers (CSRT, KCF, MOSSE) exigem o pacote **`opencv-contrib-python`** (instale com `pip install opencv-contrib-python`), não o `opencv-python` básico.

### 42.4 — KCF em detalhe (Kernelized Correlation Filters)

- **O que é:** rastreador baseado em **filtros de correlação kernelizados**. Usa uma propriedade matemática (matrizes circulantes + **FFT**, a Transformada Rápida de Fourier) para treinar e aplicar o filtro **muito rápido**, extraindo características **HOG** (*Histogram of Oriented Gradients* — histogramas de direção de borda) do objeto.
- **Ponto forte:** **velocidade**. Roda folgado em tempo real, até em CPU modesta. Boa precisão para objetos que se movem de forma previsível sem mudar muito.
- **Fraquezas:**
  - **Oclusão total** — se o objeto some atrás de algo, o KCF geralmente o perde e **não recupera**.
  - **Mudança de escala** — o KCF trabalha com caixa de **tamanho fixo**; se o objeto se aproxima/afasta (fica maior/menor), ele não acompanha bem o tamanho.
- **Quando usar:** quando **velocidade** importa mais que precisão milimétrica e o objeto não é muito ocluído — ex.: seguir um rosto próximo da câmera, um objeto em esteira.

### 42.5 — CSRT em detalhe (Channel and Spatial Reliability Tracker)

- **O que é:** *Discriminative Correlation Filter with Channel and Spatial Reliability*. Uma evolução mais sofisticada: usa um **mapa de confiabilidade espacial** que aprende **quais partes** da caixa realmente pertencem ao objeto (ignorando o fundo dentro do retângulo) e combina vários **canais** de características (HOG + cores).
- **Ponto forte:** **precisão**. Lida bem com **mudança de escala**, objetos **não-retangulares** e deformações. Segue o alvo com caixas mais justas e é mais robusto que o KCF.
- **Fraqueza:** **mais lento** (tipicamente ~25 FPS vs. dezenas/centenas do KCF). Consome mais CPU.
- **Quando usar:** quando **precisão** importa mais que a taxa de quadros — ex.: medir com exatidão a trajetória de um objeto, rastrear algo que muda de tamanho, análise esportiva detalhada.

**KCF × CSRT — o trade-off central:**

| | KCF | CSRT |
|---|-----|------|
| Prioridade | **Velocidade** | **Precisão** |
| Mudança de escala | ruim | **boa** |
| Oclusão | fraca | razoável |
| Velocidade (FPS) | muito alta | moderada (~25) |
| Escolha típica | tempo real em hardware limitado | quando o erro custa caro |

### 42.6 — A família de trackers do OpenCV (visão geral)

O OpenCV traz vários trackers prontos, cada um com um perfil velocidade × precisão diferente:

| Tracker | Perfil | Observação |
|---------|--------|-----------|
| **MOSSE** | ultrarrápido, precisão baixa | o mais veloz; ótimo para hardware fraco |
| **KCF** | rápido, precisão média | bom equilíbrio; falha em oclusão/escala |
| **CSRT** | lento, precisão alta | melhor precisão dos clássicos |
| **MedianFlow** | reporta bem quando falha | ótimo para movimento suave e previsível; ruim em saltos |
| **MIL** | robusto a oclusão parcial | mais lento que o KCF, pode "derivar" |
| **Boosting** | antigo (baseado em AdaBoost) | legado; superado pelos demais |
| **TLD** | recupera após oclusão | tende a muitos falsos positivos |
| **GOTURN** | baseado em **deep learning** | precisa de arquivos de modelo; usa CNN |

**Regra prática de escolha:** precisa do **mais rápido**? MOSSE ou KCF. Precisa do **mais preciso**? CSRT. Objeto **some e volta**? Considere TLD ou (melhor) *tracking-by-detection* com re-detecção.

### 42.7 — Rastrear vários objetos ao mesmo tempo

Para seguir **N objetos**, cria-se um tracker independente por objeto (ou usa-se o `MultiTracker`, quando disponível na sua versão):

```python
multi_tracker = cv2.legacy.MultiTracker_create()
for caixa in caixas_iniciais:                       # uma caixa por objeto
    multi_tracker.add(cv2.legacy.TrackerCSRT_create(), frame, caixa)

# no loop:
sucesso, caixas = multi_tracker.update(frame)       # atualiza todas de uma vez
for cx in caixas:
    x, y, w, h = [int(v) for v in cx]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

Para cenários sérios de **múltiplos objetos** (MOT, *Multiple Object Tracking*) com entra-e-sai de cena, o padrão moderno é **detector (YOLO) + associação de identidades** via algoritmos como **SORT** e **DeepSORT** — que juntam a detecção da Seção 39 com a lógica de identidade desta seção.

### 42.8 — Aplicações na vida real

| Área | Como o tracking é usado |
|------|--------------------------|
| **Segurança/vigilância** | seguir uma pessoa entre câmeras, detectar invasão, "linha virtual" cruzada |
| **Trânsito** | contar veículos, medir **velocidade** (pixels/frame → km/h), detectar conversão proibida |
| **Esportes** | rastrear bola e jogadores, gerar mapas de calor, estatísticas de movimentação |
| **Varejo** | contar clientes, medir tempo de permanência, analisar fluxo em corredores |
| **Drones** | modo *follow-me* (o drone segue o piloto/objeto marcado) |
| **Realidade Aumentada** | fixar um objeto virtual sobre um objeto real em movimento |
| **Carros autônomos** | acompanhar pedestres e outros veículos para prever trajetórias |
| **Robótica industrial** | seguir peças em esteira para pegar/inspecionar |

**Resumo da Seção 42:** rastreamento segue **o mesmo** objeto entre frames, mantendo identidade e economizando processamento vs. detectar tudo toda hora. **KCF** = rápido (mas sofre com escala/oclusão); **CSRT** = preciso (mas mais lento). Fluxo idêntico: `Tracker*_create` → `selectROI` → `init` → `update` no loop. Em produção, combine com **detecção** (tracking-by-detection / DeepSORT).

---

## 43. Aprofundando: Classificador em Cascata e Haar Cascade

> Aprofundamento teórico da Seção 37. Aqui abrimos a "caixa-preta" do `CascadeClassifier`: **por que** ele funciona, **como** é construído e **como treinar o seu próprio**.

Este é o algoritmo **Viola-Jones** (Paul Viola & Michael Jones, **2001**) — o primeiro detector de faces rápido o suficiente para rodar em tempo real, e que dominou a área por mais de uma década (é o que tem nas câmeras que desenham o quadradinho no rosto). Ele se apoia em **quatro ideias** encaixadas.

### 43.1 — Ideia 1: Features Haar (o que o classificador "olha")

Uma **feature Haar** (ou *Haar-like feature*) é um padrão retangular dividido em regiões claras e escuras. O valor da feature é simplesmente:

```
valor = (soma dos pixels na região CLARA) − (soma dos pixels na região ESCURA)
```

Existem alguns tipos (bordas, linhas, quatro-retângulos):

```
[ ][ ][ ]      ██████         ██  ░░
[ ][ ][ ]  →   ░░░░░░   →     ██  ░░   ...
 (borda)      (linha)      (diagonal)
```

**A intuição:** rostos têm regularidades de **contraste**. Por exemplo, a **região dos olhos é mais escura** que a das bochechas logo abaixo; o **cavalete do nariz é mais claro** que os lados. Uma feature Haar posicionada ali dá um valor alto — é um "detector de contraste" simples. Uma face é reconhecida pela **combinação** de muitas dessas features.

**O problema:** numa janela de detecção de 24×24 pixels há **mais de 160.000** features Haar possíveis (todos os tamanhos e posições). Calcular tudo, para toda janela, em toda escala, seria lento demais. As próximas duas ideias resolvem isso.

### 43.2 — Ideia 2: Imagem Integral (calcular rápido)

Como cada feature exige **somar pixels de um retângulo** milhões de vezes, precisa ser instantâneo. A **imagem integral** (*integral image* / *summed-area table*) é uma imagem auxiliar onde cada ponto guarda a **soma de todos os pixels acima e à esquerda** dele.

Com ela, a soma de **qualquer** retângulo, de qualquer tamanho, é calculada com apenas **4 acessos à memória** (as quatro quinas), em vez de somar pixel a pixel:

```
soma_do_retângulo = D − B − C + A     (A, B, C, D = as 4 quinas na imagem integral)
```

**Por que é genial:** o custo de avaliar uma feature passa a ser **constante**, independentemente do tamanho do retângulo. É o truque que torna o Viola-Jones viável em tempo real.

### 43.3 — Ideia 3: AdaBoost (escolher as melhores features)

Das 160.000+ features, a **imensa maioria é inútil**. O **AdaBoost** (*Adaptive Boosting*) é o algoritmo que, no treino, **seleciona as poucas centenas de features que realmente importam** e as combina.

Como funciona a intuição:
- Cada feature isolada é um **classificador fraco** (*weak learner*) — mal acerta um pouco mais que o cara-ou-coroa.
- O AdaBoost escolhe a melhor feature, dá **mais peso aos exemplos que ela errou**, escolhe a próxima feature focada nesses erros, e assim por diante.
- A soma ponderada de muitos classificadores fracos vira um **classificador forte** (*strong classifier*) e preciso.

**Conexão com o guia:** é o mesmo princípio de "muitos modelos fracos viram um forte" da Random Forest (Seção 8) — mas aqui os fracos são **features Haar** e a combinação é **sequencial e ponderada** (boosting) em vez de votação paralela (bagging).

### 43.4 — Ideia 4: A Cascata de Classificadores (o "Cascade")

Mesmo com features boas, testar todas em **cada** janela da imagem é caro — e **99% das janelas não têm rosto** (são parede, céu, chão). A **cascata** resolve isso com uma fila de **estágios**:

```
Janela da imagem
   │
[Estágio 1]  poucas features, ultrarrápido  ──►  não parece rosto?  ──► DESCARTA (fim)
   │ passou
[Estágio 2]  mais features                  ──►  não parece rosto?  ──► DESCARTA
   │ passou
[Estágio 3]  ...                            ──►  ...
   │ passou por TODOS os estágios
   ▼
   ROSTO detectado ✔
```

- Os **primeiros estágios** são baratos (poucas features) e **rejeitam rapidamente** as regiões óbvias sem rosto.
- Só as regiões promissoras avançam para estágios mais caros e criteriosos.
- Resultado: quase todo o trabalho é gasto nas poucas janelas que **realmente** podem conter um rosto. Por isso se chama **cascata atencional** — o esforço se concentra onde importa.

**Analogia:** é como uma **triagem em vários portões**. O primeiro segurança faz uma pergunta rápida e barra 90% das pessoas na hora; quem passa enfrenta perguntas cada vez mais detalhadas. No fim, só quem passou por **todos** os portões é aprovado. Isso é muito mais rápido do que fazer a entrevista completa com cada pessoa.

### 43.5 — Usando cascatas prontas (revisão + além do rosto)

O OpenCV já vem com vários `.xml` treinados. Basta trocar o arquivo:

```python
import cv2

# Cada arquivo é uma cascata treinada para um objeto diferente
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
body_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```

**Recapitulando os parâmetros do `detectMultiScale` (agora com o "porquê"):**

| Parâmetro | O que faz | Efeito de aumentar |
|-----------|-----------|--------------------|
| `scaleFactor` | de quanto a janela cresce a cada escala (ex.: 1.1 = +10%) | mais rápido, porém **pode pular** rostos |
| `minNeighbors` | quantas detecções sobrepostas confirmam o objeto | **menos falsos positivos**, mas pode perder rostos reais |
| `minSize` | menor objeto detectável (px) | ignora objetos pequenos → mais rápido |
| `maxSize` | maior objeto detectável (px) | ignora objetos grandes |

**Composição de cascatas (detectar dentro do detectado):** um padrão comum é achar o rosto e, **dentro** de cada rosto, procurar olhos/sorriso — reduzindo a área de busca e os falsos positivos:

```python
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]                 # recorta só a região do rosto
    olhos = eye_cascade.detectMultiScale(roi_gray)  # procura olhos DENTRO do rosto
```

### 43.6 — LBP Cascade: a alternativa mais rápida

Além do Haar, o OpenCV suporta cascatas baseadas em **LBP** (*Local Binary Patterns*). Em vez de somas de retângulos, o LBP compara cada pixel com seus **vizinhos** gerando um código binário de textura.

| | Haar Cascade | LBP Cascade |
|---|--------------|-------------|
| Precisão | geralmente **maior** | um pouco menor |
| Velocidade (treino e uso) | mais lento | **mais rápido** (trabalha com inteiros) |
| Quando preferir | precisão em rosto frontal | dispositivos limitados / treino rápido |

### 43.7 — Treinando a SUA própria cascata

Dá para treinar uma cascata para **qualquer objeto rígido** (logotipo, placa, produto). O fluxo clássico do OpenCV usa duas ferramentas de linha de comando:

1. **Reúna as amostras:**
   - **Positivas** — muitas imagens contendo o objeto, com a caixa anotada (centenas a milhares).
   - **Negativas** (*background*) — imagens **sem** o objeto (o dobro ou mais das positivas).
2. **Gere o vetor de positivas** com `opencv_createsamples` → produz um arquivo `.vec`.
3. **Treine a cascata** com `opencv_traincascade`:

```bash
opencv_traincascade -data cascade/ -vec positivas.vec -bg negativas.txt \
    -numPos 900 -numNeg 1800 -numStages 12 \
    -w 24 -h 24 -featureType HAAR
```

- **`-numPos` / `-numNeg`** — quantas amostras positivas/negativas por estágio.
- **`-numStages`** — quantos estágios da cascata (mais estágios = mais preciso, treino mais longo, risco de overfitting).
- **`-w` / `-h`** — tamanho da janela de treino (o objeto é normalizado para esse tamanho; 24×24 é comum).
- **`-featureType`** — `HAAR` ou `LBP`.
- O resultado é um **`cascade.xml`** que você carrega no `CascadeClassifier` exatamente como os prontos.

> **Realidade de 2020+:** treinar Haar/LBP é trabalhoso e o resultado perde para **detectores de deep learning** (YOLO, SSD, Seção 39) em precisão e em objetos variados. Hoje o Haar Cascade brilha em **nichos**: rápido, leve, sem GPU, para **objetos rígidos e frontais** com boa iluminação. Para o resto, prefira uma CNN de detecção.

### 43.8 — Limitações (e quando NÃO usar)

- **Só funciona bem de frente** — o `haarcascade_frontalface` falha com rostos de perfil ou muito inclinados.
- **Sensível à iluminação** — sombras fortes atrapalham (por isso o pré-processamento da Seção 35 ajuda).
- **Falsos positivos** — pode "ver rosto" em texturas aleatórias; ajuste `minNeighbors` para reduzir.
- **Objetos rígidos** — vai bem em rosto/placa/olho; mal em objetos deformáveis ou muito variados.

### 43.9 — Aplicações na vida real

| Área | Uso do Cascade Classifier |
|------|---------------------------|
| **Câmeras/celulares** | o quadradinho de **autofoco no rosto**; disparo ao detectar sorriso |
| **Controle de acesso** | detectar o rosto antes de passar ao reconhecimento facial |
| **Contagem de pessoas** | `haarcascade_fullbody` para contar gente em ambientes |
| **Trânsito** | `haarcascade_russian_plate_number` para localizar **placas** antes do OCR (Seção 36) |
| **Varejo/marketing** | medir atenção (rosto voltado para a vitrine), estimar fluxo |
| **Interação/jogos** | rastrear olhos/rosto para controlar interfaces sem as mãos |
| **Pré-filtro de pipelines** | achar rapidamente a região de interesse e só então rodar um modelo pesado nela |

**Resumo da Seção 43:** o Cascade Classifier (Viola-Jones) combina **4 ideias** — features **Haar** (contraste), **imagem integral** (soma instantânea), **AdaBoost** (seleciona as features úteis) e a **cascata** de estágios (rejeita cedo o que não é objeto, ganhando velocidade). Use `.xml` prontos para rosto/olho/placa, componha detecções (olho dentro do rosto), troque por **LBP** se precisar de mais velocidade, e treine o seu com `opencv_traincascade`. É leve e sem GPU — para objetos rígidos e frontais; para o resto, YOLO/CNN vencem.

---

## 44. Glossário da Parte IV (Visão Computacional)

| Termo | Definição |
|-------|-----------|
| **OpenCV (`cv2`)** | Biblioteca-padrão de processamento de imagens e vídeo |
| **BGR** | Ordem de canais do OpenCV (Azul-Verde-Vermelho); converta p/ RGB antes de exibir no matplotlib |
| **Grayscale (escala de cinza)** | Imagem de 1 canal (intensidade); reduz dados e simplifica o processamento |
| **cvtColor** | Converte entre espaços de cor (BGR↔RGB, BGR→GRAY) |
| **resize / interpolation** | Muda a resolução; `INTER_AREA` para reduzir |
| **GaussianBlur** | Suavização por média ponderada (sino); reduz ruído |
| **medianBlur** | Suavização pela mediana; remove ruído "sal e pimenta" |
| **Canny** | Detector de bordas (dois limiares de histerese) |
| **Threshold / Otsu / Adaptativo** | Binarização (P&B); Otsu acha o limiar sozinho; adaptativo varia por região |
| **OCR** | Extrair texto de dentro de imagens (Tesseract, PaddleOCR) |
| **Haar Cascade** | Classificador clássico e rápido p/ detectar faces/olhos; `detectMultiScale` |
| **scaleFactor / minNeighbors** | Parâmetros do Haar: escala da janela e nº de confirmações |
| **CNN** | Rede neural para imagens; aprende filtros que detectam padrões locais |
| **Convolução (kernel/filtro)** | Filtro que desliza pela imagem e extrai características (bordas, texturas) |
| **Mapa de características** | Saída de uma camada convolucional |
| **Max Pooling** | Reduz dimensão pegando o máximo de cada janela; dá robustez à translação |
| **Flatten** | Achata mapas 2D em vetor 1D para as camadas densas |
| **sparse_categorical_crossentropy** | Perda multiclasse quando os rótulos são inteiros (sem one-hot) |
| **Detecção de objetos** | Classe **+ localização** (bounding box) de cada objeto na imagem |
| **YOLO** | *You Only Look Once*; detector de objetos rápido (tempo real) |
| **Bounding box** | Caixa retangular que envolve um objeto detectado |
| **IoU / mAP** | Métricas de detecção: sobreposição das caixas / precisão média |
| **conf (confiança)** | Limiar mínimo de certeza para aceitar uma detecção |
| **GAN** | Rede generativa: gerador × discriminador competindo |
| **Gerador / Discriminador** | Cria imagens falsas / tenta distingui-las das reais |
| **Vetor latente** | Ruído de entrada do gerador; "semente" da imagem |
| **Conv2DTranspose (deconvolução)** | Aumenta a imagem (upsampling); oposto da Conv2D |
| **LeakyReLU** | ReLU que deixa passar um pouco dos negativos; comum em GANs |
| **BatchNormalization** | Normaliza saídas entre camadas; estabiliza o treino |
| **MediaPipe** | Modelos prontos de landmarks (mão, pose, rosto) em tempo real |
| **Landmark** | Ponto de referência detectado (ex.: ponta do dedo, joelho) |
| **Tempo real (webcam)** | Processar frames ao vivo (`cv2.VideoCapture`) |
| **Rastreamento (tracking)** | Seguir o **mesmo** objeto entre frames, mantendo o ID |
| **Detecção × Rastreamento** | Achar objetos em cada frame × seguir um já marcado (mais barato) |
| **selectROI** | Marcar com o mouse a caixa inicial do objeto a rastrear |
| **tracker.init / update** | Inicializa com a caixa / devolve a nova posição a cada frame |
| **KCF** | Tracker de correlação **rápido** (HOG+FFT); sofre com escala/oclusão |
| **CSRT** | Tracker **preciso** (confiabilidade espacial); lida com escala; mais lento |
| **MOSSE** | Tracker ultrarrápido, precisão baixa |
| **Tracking-by-detection** | Detectar de vez em quando + rastrear no meio (ex.: SORT/DeepSORT) |
| **opencv-contrib-python** | Pacote extra necessário p/ vários trackers (CSRT, KCF, MOSSE) |
| **Viola-Jones** | Algoritmo (2001) por trás do Haar Cascade |
| **Feature Haar** | Padrão de contraste (claro − escuro) usado como detector fraco |
| **Imagem integral** | Tabela auxiliar que soma qualquer retângulo em 4 acessos |
| **AdaBoost** | Boosting que seleciona/combina as features Haar úteis |
| **Cascata de classificadores** | Estágios que rejeitam cedo o que não é objeto (rápido) |
| **Classificador fraco/forte** | Feature isolada / combinação ponderada de muitas |
| **LBP Cascade** | Alternativa ao Haar por padrões de textura; treino mais rápido |
| **opencv_traincascade** | Ferramenta p/ treinar a própria cascata (positivas/negativas) |
| **HOG** | Histograma de orientação de bordas; característica usada por KCF |

---

## Mapa Visual dos Algoritmos

```
                    DADOS
                      │
          ┌───────────┴───────────┐
          │                       │
    SUPERVISIONADO          NÃO SUPERVISIONADO
    (tem rótulos y)          (sem rótulos)
          │                       │
    ┌─────┴─────┐            K-Means
    │           │            (agrupamento)
 REGRESSÃO  CLASSIFICAÇÃO
 (valor     (categoria)
 contínuo)      │
    │       ┌───┴────────┐
 Linear     │            │
 Regression KNN         SVM
 RandomForest           NaiveBayes
                         │
                    DEEP LEARNING
                    (Redes Neurais)
                    ┌──────┴──────┐
                  Keras        PyTorch
               (mais fácil)  (mais controle)
```

---

*Material gerado em Agosto/2026 com base nos arquivos do repositório IA for Devs.*
