Deixa eu recapitular cada um com sua analogia, na ordem em que os vimos:
K-NN — "você é a companhia que mantém." Para classificar algo novo, olhe os K vizinhos mais próximos e siga a maioria. Como descobrir se seu bairro é residencial olhando as casas mais próximas da sua. Supervisionado, "preguiçoso" (só memoriza), e sensível à escala dos dados.
Regressão linear — "a régua equilibrada no meio dos pontos." Ajusta uma reta que passa o mais perto possível de todos os dados para prever um número. Como o corretor que criou a regra "cada m² a mais vale ~R$ 3.000". Prevê valores contínuos, é rápida e interpretável, mas só enxerga relações em linha reta.
Árvore de decisão — "as 20 perguntas do médico." Uma sequência de perguntas sim/não que vai afunilando até a resposta, escolhendo sempre a pergunta que melhor separa os dados. Fácil de visualizar, mas tende a "decorar" (overfitting) se crescer demais.
K-Means — "os centros que perseguem seus grupos." Você diz quantos grupos quer (K); ele joga centros ao acaso e repete: cada ponto vai ao centro mais próximo, cada centro pula para o meio do seu grupo, até estabilizar. Como observar aglomerados de pessoas numa festa. Não supervisionado, mas exige o K e só forma grupos arredondados.
DBSCAN — "as cidades vistas do satélite à noite." Agrupa por densidade: regiões amontoadas viram grupos, pontos isolados viram ruído. Descobre o número de grupos sozinho, captura formatos irregulares e detecta anomalias — ao custo de parâmetros (eps, min_samples) mais delicados.
O eixo que organiza tudo é a presença ou ausência de rótulos: no supervisionado você tem as respostas certas e ensina o modelo a reproduzi-las; no não supervisionado não há respostas, e o modelo precisa achar a estrutura escondida por conta própria.
Agora, o questionário para você testar seus conhecimentos. Montei com 12 questões (mistura de múltipla escolha e dissertativas) e o gabarito comentado logo abaixo — tente responder tudo antes de conferir.

# Questionário de estudo — Algoritmos de Machine Learning

Um guia de revisão sobre K-NN, Regressão Linear, Árvore de Decisão, K-Means e DBSCAN.
Responda a todas as questões antes de olhar o gabarito no final.

---

## Parte 1 — Múltipla escolha

**1.** Qual das opções abaixo é um algoritmo de aprendizado **não supervisionado**?

- a) Regressão linear
- b) Árvore de decisão
- c) K-Means X
- d) K-NN

**2.** No algoritmo K-NN, o que acontece se escolhermos um valor de K **muito pequeno** (por exemplo, K = 1)?

- a) O modelo fica mais lento para treinar
- b) O resultado fica muito sensível a um único vizinho, podendo ser enganado por uma exceção x
- c) O modelo passa a prever números contínuos em vez de classes
- d) A resposta fica "borrada" por incluir vizinhos distantes demais

**3.** A regressão linear pelo método dos mínimos quadrados busca a reta que:

- a) Passa exatamente por cima de todos os pontos
- b) Minimiza a soma dos erros (distâncias verticais) elevados ao quadrado
- c) Tem a maior inclinação possível X
- d) Cruza o eixo vertical na origem (b = 0)

**4.** Numa árvore de decisão, o que é uma **folha (leaf)**?

- a) A primeira pergunta, no topo da árvore
- b) Uma pergunta intermediária que divide os dados
- c) A ponta final da árvore, onde está a resposta X
- d) O caminho "sim" ou "não" que sai de um nó

**5.** No DBSCAN, um ponto classificado com o rótulo **-1** é:

- a) Um ponto núcleo (core)
- b) Um ponto de borda (border)
- c) Ruído (outlier), que não pertence a grupo nenhum X
- d) O centro de um cluster

**6.** Qual algoritmo você escolheria se **não soubesse quantos grupos** existem nos dados e suspeitasse de **formatos irregulares**?

- a) K-Means
- b) DBSCAN X
- c) Regressão linear
- d) K-NN

---

## Parte 2 — Verdadeiro ou Falso

**7.** O K-NN é chamado de algoritmo "preguiçoso" (lazy) porque não constrói um modelo antecipadamente — apenas guarda os dados e faz os cálculos na hora de classificar. ( V / F ) V

**8.** A regressão linear consegue capturar bem relações que têm um formato fortemente curvo (como um crescimento exponencial). ( V / F ) F

**9.** O overfitting em uma árvore de decisão acontece quando ela cria perguntas específicas demais e acaba "decorando" os dados de treino, funcionando mal com dados novos. ( V / F ) V

**10.** No K-Means, você precisa informar o número de grupos (o K) antes de rodar o algoritmo. ( V / F ) V

---

## Parte 3 — Dissertativas

**11.** Explique, com suas palavras, a diferença fundamental entre **aprendizado supervisionado** e **não supervisionado**. Dê um exemplo de algoritmo de cada tipo.


**12.** Descreva os **dois passos** que o K-Means repete a cada iteração até convergir. Você pode usar a analogia que preferir.

---
---

# GABARITO COMENTADO

*(Só continue depois de responder tudo acima!)*

**1. Resposta: c) K-Means.**
K-Means e DBSCAN são não supervisionados (trabalham sem rótulos). Regressão linear, árvore de decisão e K-NN são supervisionados.

**2. Resposta: b).**
Com K = 1 a classificação depende de um único vizinho; se por acaso esse vizinho for uma exceção/ruído, a previsão erra. Por isso se busca um K equilibrado (e ímpar, para evitar empates). A opção (d) descreve o problema oposto — um K grande demais.

**3. Resposta: b).**
O método dos mínimos quadrados escolhe a reta que minimiza a soma dos erros ao quadrado. Ela não passa exatamente por todos os pontos (isso quase nunca é possível), e o intercepto b não precisa ser zero.

**4. Resposta: c).**
A folha é a ponta final, onde está a resposta (a classe ou o valor previsto). A opção (a) descreve a raiz, (b) descreve um nó de decisão e (d) descreve um ramo.

**5. Resposta: c) Ruído (outlier).**
O rótulo -1 no DBSCAN marca pontos isolados que não pertencem a nenhum grupo — justamente o que torna o algoritmo útil para detectar anomalias e fraudes.

**6. Resposta: b) DBSCAN.**
Ele descobre o número de grupos sozinho e captura formatos irregulares (curvos, alongados). O K-Means exigiria que você informasse o K de antemão e só forma grupos arredondados.

**7. Verdadeiro.**
O K-NN não tem uma fase de "treino" real; ele só armazena os dados e faz todo o cálculo de distância no momento da classificação.

**8. Falso.**
A regressão linear assume uma relação aproximadamente em linha reta. Para dados fortemente curvos ela falha, e usam-se variações como a regressão polinomial.

**9. Verdadeiro.**
Esse é exatamente o conceito de overfitting: ótimo desempenho no treino, ruim no mundo real. Combate-se podando a árvore (limitando a profundidade, por exemplo).

**10. Verdadeiro.**
O "K" do K-Means é o número de grupos, que você define antes de rodar. Essa é justamente uma das diferenças em relação ao DBSCAN, que descobre o número sozinho.

**11. Resposta esperada:**
No aprendizado **supervisionado**, os dados vêm com rótulos (as respostas certas), e o modelo aprende a reproduzi-las — prevendo uma classe ou um número. Exemplos: K-NN, regressão linear, árvore de decisão.
No aprendizado **não supervisionado**, não há rótulos; o modelo precisa descobrir sozinho a estrutura escondida (os grupos naturais) apenas pelas semelhanças entre os dados. Exemplos: K-Means, DBSCAN.
O ponto-chave é a presença ou ausência de rótulos/respostas.

**12. Resposta esperada:**
A cada iteração, o K-Means repete:
1. **Atribuir:** cada ponto é ligado ao centro (centróide) mais próximo, formando os grupos provisórios.
2. **Mover/atualizar:** cada centro se desloca para a média (o "meio") de todos os pontos que capturou.
Esses dois passos se repetem até os centros pararem de se mover — o momento da convergência. (Analogia válida: observar de uma varanda os aglomerados de pessoas numa festa se formando por proximidade.)

---

## Como se avaliar

- **10–12 acertos:** Excelente! Você tem uma base conceitual sólida para aprofundar na matemática.
- **7–9 acertos:** Bom caminho. Revise os pontos que errou antes de avançar.
- **Menos de 7:** Vale reler o resumo da conversa e refazer o quiz — os conceitos vão fixar rápido.

## Sugestões para aprofundar

- Implemente cada algoritmo com a biblioteca `scikit-learn` (Python) usando datasets pequenos.
- Estude as métricas de impureza (entropia e índice de Gini) para árvores.
- Explore a evolução das árvores: Random Forest e Gradient Boosting.
- Teste o caso das "duas luas" para ver K-Means vs DBSCAN na prática.
- Aprenda o "método do cotovelo" para escolher o K no K-Means.
