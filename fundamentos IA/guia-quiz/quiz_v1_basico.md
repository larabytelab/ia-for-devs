# Quiz de Machine Learning — Versão 1: Conceitos Básicos

> Responda todas as questões antes de conferir o gabarito no final.
> Nível: iniciante — foco em definições e analogias.

---

## Questões

**1.** O que é aprendizado supervisionado?

- a) O modelo aprende sem nenhuma orientação humana
- b) O modelo aprende a partir de dados que já têm as respostas certas (rótulos)
- c) O modelo só funciona com imagens
- d) O modelo precisa de uma conexão com a internet para aprender

---

**2.** O algoritmo K-NN classifica um novo ponto baseando-se em:

- a) Uma fórmula matemática complexa treinada previamente
- b) Os K vizinhos mais próximos daquele ponto nos dados
- c) A média de todos os dados de treino
- d) Uma árvore de perguntas sim/não

---

**3.** Para que serve a Regressão Linear?

- a) Agrupar dados semelhantes em clusters
- b) Classificar dados em categorias (ex.: spam ou não spam)
- c) Prever um número contínuo (ex.: preço de uma casa)
- d) Detectar anomalias em dados

---

**4.** O que é uma "folha" (leaf) em uma Árvore de Decisão?

- a) A raiz da árvore, onde começa a análise
- b) Uma pergunta intermediária que divide os dados
- c) O nó final onde está a resposta (classe ou valor)
- d) Um ramo que vai para a esquerda

---

**5.** O algoritmo K-Means precisa que você informe:

- a) O formato dos grupos que serão criados
- b) O número de grupos (K) antes de rodar
- c) Os rótulos de cada ponto do dataset
- d) A distância máxima entre dois pontos

---

**6.** O DBSCAN é diferente do K-Means porque:

- a) É mais rápido e usa menos memória
- b) Só funciona com dados de texto
- c) Descobre o número de grupos sozinho e detecta outliers
- d) Exige que os dados estejam normalizados

---

**7.** O que significa desvio padrão alto em um conjunto de dados?

- a) Os dados estão muito concentrados perto da média
- b) Os dados estão muito espalhados (longe da média)
- c) A média do conjunto é muito grande
- d) Os dados seguem uma distribuição uniforme

---

**8.** Na distribuição normal (curva em sino), o que acontece com ~95% dos dados?

- a) Ficam a 1 desvio padrão da média
- b) Ficam a 2 desvios padrão da média
- c) Ficam a 3 desvios padrão da média
- d) Ficam exatamente na média

---

**9.** O Teorema de Bayes é melhor descrito como:

- a) Um método para calcular a média de uma amostra
- b) Uma forma de atualizar uma crença com base em novas evidências
- c) Um algoritmo que agrupa dados por similaridade
- d) Uma técnica de compressão de dados

---

**10.** O que é "correlação" entre duas variáveis?

- a) Uma prova de que uma variável causa a outra
- b) A média das duas variáveis somadas
- c) Uma medida de quanto as duas variáveis variam juntas (−1 a +1)
- d) A diferença entre os valores máximos das duas variáveis

---

**11.** O que significa "extrair features" (extração de recursos)?

- a) Deletar colunas desnecessárias do banco de dados
- b) Transformar dados brutos em representações numéricas úteis para o modelo
- c) Criar backups dos dados originais
- d) Ordenar os dados por ordem de importância

---

**12.** No DBSCAN, um ponto com rótulo **-1** significa que:

- a) Ele é o centro de um cluster
- b) Ele está na borda de um cluster
- c) Ele é um outlier (ruído), sem grupo definido
- d) Ele tem o menor valor no dataset

---

**13.** Qual das opções é um exemplo de **dado estruturado**?

- a) Uma foto tirada com o celular
- b) Um arquivo de áudio de uma reunião
- c) Uma tabela de planilha com colunas fixas (nome, idade, salário)
- d) Um e-mail com texto livre

---

**14.** Tweets, vídeos do YouTube e fotos do Instagram são exemplos de qual tipo de dado?

- a) Dados estruturados
- b) Dados semiestruturados
- c) Dados não estruturados
- d) Dados temporais

---

**15.** Dados **semiestruturados** são aqueles que:

- a) Têm formato de tabela com linhas e colunas fixas
- b) Não têm absolutamente nenhuma organização
- c) Têm alguma organização (como tags ou chaves), mas não seguem um esquema rígido de tabela — ex.: JSON, XML
- d) São dados coletados em tempo real por sensores

---

**16.** Dados de temperatura registrados a cada hora por um sensor ao longo de um mês são exemplos de:

- a) Dados semiestruturados
- b) Dados não estruturados
- c) Dados temporais (séries temporais)
- d) Dados estruturados estáticos

---

**17.** O aprendizado **semissupervisionado** combina:

- a) Algoritmos supervisionados e não supervisionados rodando separadamente
- b) Uma pequena quantidade de dados rotulados com uma grande quantidade de dados sem rótulo
- c) Apenas dados não rotulados com regras criadas manualmente
- d) Dados de treino e teste sem divisão entre eles

---

## Gabarito

| Questão | Resposta | Explicação rápida |
|---------|----------|-------------------|
| 1 | b | No supervisionado, os dados de treino têm rótulos (respostas corretas). |
| 2 | b | K-NN olha os K vizinhos mais próximos e segue a maioria. |
| 3 | c | Regressão linear prevê valores contínuos como preços e temperaturas. |
| 4 | c | A folha é o nó final da árvore — onde a resposta é entregue. |
| 5 | b | O K do K-Means é o número de grupos definido pelo usuário. |
| 6 | c | DBSCAN acha o número de grupos sozinho e marca pontos isolados como ruído. |
| 7 | b | Desvio padrão alto = dados espalhados; baixo = dados concentrados. |
| 8 | b | Regra 68-95-99: 95% dos dados ficam a 2 desvios padrão da média. |
| 9 | b | Bayes parte de uma estimativa inicial e a atualiza com cada nova evidência. |
| 10 | c | Correlação mede relação entre variáveis, mas não implica causa. |
| 11 | b | Extrair features = transformar dados brutos em números úteis para o ML. |
| 12 | c | Rótulo -1 no DBSCAN = outlier, ponto que não pertence a nenhum grupo. |
| 13 | c | Dados estruturados cabem em tabelas com colunas e tipos fixos. |
| 14 | c | Tweets, vídeos e fotos não têm estrutura tabular — são não estruturados. |
| 15 | c | JSON e XML têm organização via chaves/tags, mas sem esquema rígido de tabela. |
| 16 | c | Dados com registro ao longo do tempo formam séries temporais. |
| 17 | b | Semissupervisionado usa poucos rótulos + muitos dados sem rótulo — equilibra custo e precisão. |

---

### Como se avaliar
- **14–17 acertos:** Ótima base! Passe para a Versão 2.
- **10–13 acertos:** Bom progresso. Revise os conceitos errados e tente de novo.
- **Menos de 10:** Releia os guias de estudo antes de continuar.
