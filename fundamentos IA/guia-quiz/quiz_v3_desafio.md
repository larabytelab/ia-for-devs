# Quiz de Machine Learning — Versão 3: Desafio e Conexões

> Responda todas as questões antes de conferir o gabarito no final.
> Nível: desafio — conecta conceitos, questões com pegadinha, e "qual está ERRADO?".
> Algumas alternativas erradas são parcialmente verdadeiras — leia com atenção!

---

## Questões

**1.** Qual das seguintes afirmações sobre o **K-NN** está **INCORRETA**?

- a) K-NN é um algoritmo "lazy" — ele não treina um modelo, apenas memoriza os dados
- b) K-NN é sensível à escala das features, portanto normalização é recomendada
- c) K-NN pertence ao aprendizado supervisionado
- d) K-NN descobre automaticamente o número ideal de K durante o treinamento

---

**2.** O Naive Bayes Gaussiano usa internamente quais dois conceitos estatísticos para modelar cada feature?

- a) Correlação e z-score
- b) Média e desvio padrão (distribuição normal)
- c) Mínimos quadrados e gradiente
- d) Entropia e ganho de informação

---

**3.** Você aplica K-Means em dados de clientes e obtém 3 clusters bem definidos. Um colega diz: "o K-Means funcionou porque os dados formavam grupos circulares e compactos." Por que essa observação é relevante?

- a) K-Means só funciona com dados bidimensionais
- b) K-Means é baseado em distância de centroide, o que favorece clusters arredondados e não lida bem com formatos irregulares
- c) Clusters circulares indicam que o DBSCAN teria sido melhor
- d) A observação não tem relevância — K-Means funciona em qualquer forma de cluster

---

**4.** Por que o Naive Bayes usa o **truque do logaritmo** ao multiplicar probabilidades?

- a) Para converter porcentagens em números inteiros
- b) Para evitar underflow — probabilidades muito pequenas multiplicadas resultam em zero na memória do computador
- c) Para normalizar os dados antes do cálculo
- d) Para acelerar o tempo de busca no índice invertido

---

**5.** Um dataset tem a nota de alunos e as horas de estudo. A correlação calculada é **+0.85**. Um gestor conclui: "vou aumentar as horas de estudo para garantir melhores notas." Qual é a crítica mais precisa?

- a) A correlação de +0.85 é fraca demais para qualquer conclusão
- b) A correlação mede relação, não causa — horas de estudo pode influenciar notas, mas há outros fatores e a causalidade exige estudos específicos
- c) O gestor deveria usar regressão logística, não correlação
- d) Correlação só é válida com dados normalizados

---

**6.** No HDFS (Hadoop Distributed File System), qual é a função das **réplicas** dos blocos de dados?

- a) Aumentar a velocidade de leitura em 3x
- b) Garantir tolerância a falhas — se um nó cair, os dados não são perdidos
- c) Comprimir os dados para economizar espaço em disco
- d) Permitir que o MapReduce execute em paralelo

---

**7.** Qual é a principal diferença entre o Apache **Spark** e o **MapReduce** do Hadoop?

- a) Spark usa SQL e MapReduce não
- b) Spark processa dados em memória RAM (mais rápido), enquanto MapReduce grava em disco entre etapas
- c) Spark só funciona com dados estruturados
- d) MapReduce é mais moderno e substituiu o Spark

---

**8.** Considere a seguinte analogia: "A Regressão Linear é como um **alfaiate** que tenta traçar a melhor linha reta entre os pontos." Qual é a técnica matemática que ela usa para encontrar essa "melhor linha"?

- a) Gradiente descendente estocástico
- b) Método dos mínimos quadrados — minimiza a soma dos quadrados dos erros
- c) Função de entropia cruzada
- d) Algoritmo de Dijkstra

---

**9.** Um modelo de Árvore de Decisão com profundidade máxima de 20 foi treinado. O conjunto de treino tem 100% de acerto, mas o teste tem 55%. Um colega sugere: "basta aumentar o dataset de treino que o problema se resolve." Você concorda? Qual resposta é mais completa?

- a) Sim, mais dados sempre resolvem overfitting
- b) Não. Mais dados ajudam, mas a raiz do problema é a profundidade excessiva — limitar a profundidade máxima (podagem) é mais eficaz
- c) Sim, o problema é o dataset e não a árvore
- d) Não — o correto seria usar DBSCAN no lugar da árvore

---

**10.** Um ponto de dados tem **z-score = +3.8**. O que isso indica sobre esse ponto?

- a) É um ponto exatamente na média do dataset
- b) É um ponto que está 3.8 desvios padrão abaixo da média — provavelmente um erro
- c) É um ponto que está 3.8 desvios padrão acima da média — provavelmente um outlier ou valor suspeito
- d) É um ponto dentro da faixa normal (até 2 desvios padrão)

---

**11.** Em qual situação o **DBSCAN** tem vantagem clara sobre o K-Means?

- a) Quando os dados têm exatamente 3 grupos esféricos bem definidos
- b) Quando o usuário sabe exatamente quantos grupos existem
- c) Quando os grupos têm formatos irregulares e há presença de outliers
- d) Quando se deseja prever valores contínuos

---

**12.** Qual das opções descreve **corretamente** o aprendizado **não supervisionado**?

- a) O modelo recebe dados com rótulos e aprende a classificar novos dados
- b) O modelo recebe dados sem rótulos e encontra padrões ou estruturas por conta própria
- c) O modelo só funciona sem dados — aprende pelas regras definidas pelo programador
- d) O modelo é treinado com metade dos rótulos para economizar tempo

---

**13.** Imagine que você quer construir um sistema que detecta fraudes em tempo real em cartões de crédito. O sistema funciona assim:
- Para cada transação nova, calcula o z-score dos valores
- Usa Naive Bayes para atualizar a probabilidade de fraude
- Marca transações com probabilidade > 90% como fraude

Qual conceito **NÃO** faz parte diretamente da lógica descrita acima?

- a) Z-score para identificar valores anômalos
- b) Probabilidade bayesiana para atualizar crença de fraude
- c) Limiar (threshold) de decisão em 90%
- d) Método dos mínimos quadrados para ajustar os pesos

---

**14.** Por que o índice invertido se chama "invertido"?

- a) Porque ele processa os dados de trás para frente, do último documento ao primeiro
- b) Porque ele inverte a lógica padrão: em vez de "documento → palavras", guarda "palavra → lista de documentos"
- c) Porque ele é o oposto de um índice de banco de dados relacional
- d) Porque ele foi criado invertendo a ordem alfabética das palavras

---

**15.** Qual das alternativas apresenta a sequência **CORRETA** do funcionamento do K-Means?

- a) Definir K → Atribuir todos os pontos ao centroide mais distante → Calcular nova média → Repetir
- b) Definir K → Escolher centroides aleatórios → Atribuir cada ponto ao centroide mais próximo → Mover centroide para a média do grupo → Repetir
- c) Escolher K automaticamente → Calcular densidade dos pontos → Formar grupos por densidade
- d) Definir K → Calcular z-score de cada ponto → Agrupar por z-score similares

---

**16.** Qual das afirmações sobre **aprendizado semissupervisionado** está **INCORRETA**?

- a) É útil quando rotular dados é caro ou demorado
- b) Combina poucos dados rotulados com muitos dados não rotulados
- c) É uma técnica que fica entre o supervisionado e o não supervisionado
- d) Elimina completamente a necessidade de dados rotulados — basta ter dados brutos suficientes

---

**17.** Um cientista de dados recebe três datasets para analisar:
- **Dataset A:** Tabela SQL com colunas "CPF, Nome, Renda, Cidade"
- **Dataset B:** 10.000 tweets coletados sobre um produto
- **Dataset C:** Arquivo XML com pedidos de e-commerce (cada pedido tem campos opcionais)

Qual classificação está **correta**?

- a) A = estruturado, B = semiestruturado, C = não estruturado
- b) A = estruturado, B = não estruturado, C = semiestruturado
- c) A = semiestruturado, B = não estruturado, C = estruturado
- d) A = temporal, B = estruturado, C = semiestruturado

---

**18.** Um modelo semissupervisionado é treinado para classificar reviews de produtos. O dataset tem 200 reviews rotulados (positivo/negativo) e 20.000 sem rótulo. O modelo usa os 200 para aprender padrões e, com isso, "propaga" rótulos para os 20.000. Qual é o **risco** dessa abordagem?

- a) O modelo vai ignorar completamente os dados não rotulados
- b) Se os 200 rótulos iniciais tiverem erros ou viés, esses erros serão amplificados ao rotular os 20.000
- c) Dados de texto nunca funcionam com aprendizado semissupervisionado
- d) O modelo vai ter performance idêntica ao supervisionado puro

---

**19.** Séries temporais têm uma característica que as torna diferentes de datasets comuns. Qual é ela?

- a) Os dados sempre têm mais de 1.000 colunas de features
- b) A ordem dos registros no tempo importa — um ponto depende dos anteriores (ex.: temperatura de ontem influencia a de hoje)
- c) Só podem ser analisadas com redes neurais profundas
- d) Os dados de série temporal são sempre não estruturados

---

## Gabarito

| Questão | Resposta | Explicação rápida |
|---------|----------|-------------------|
| 1 | d | K-NN nunca descobre K automaticamente — o usuário escolhe. |
| 2 | b | Naive Bayes Gaussiano modela cada feature com média + desvio padrão (curva normal). |
| 3 | b | K-Means usa distância de centroide — funciona melhor em clusters compactos e arredondados. |
| 4 | b | Underflow: probabilidades minúsculas multiplicadas viram zero no computador; log transforma em somas. |
| 5 | b | +0.85 é uma correlação forte, mas correlação não prova causalidade — é só uma pista. |
| 6 | b | Réplicas = tolerância a falhas. Se um nó falhar, outro nó tem a cópia dos dados. |
| 7 | b | Spark processa em memória RAM (muito mais rápido). MapReduce grava em disco entre fases. |
| 8 | b | A Regressão Linear usa o método dos mínimos quadrados para minimizar os erros. |
| 9 | b | Mais dados ajudam, mas o problema principal é a profundidade excessiva (overfitting). Poda a árvore. |
| 10 | c | Z-score +3.8 = 3.8 desvios acima da média — na regra 68-95-99, isso está muito além do normal. |
| 11 | c | DBSCAN é ideal para grupos irregulares e detecta outliers automaticamente. |
| 12 | b | Não supervisionado = sem rótulos; o modelo acha padrões por conta própria. |
| 13 | d | Mínimos quadrados é da Regressão Linear, não de Bayes ou z-score. |
| 14 | b | "Invertido" porque a lógica é invertida: palavra → documentos, não documento → palavras. |
| 15 | b | K-Means: define K → centroides aleatórios → atribui pontos → move centroide → repete. |
| 16 | d | Semissupervisionado SEMPRE precisa de algum dado rotulado — a afirmação d está incorreta. |
| 17 | b | Tabela SQL = estruturado; tweets = não estruturado; XML com campos opcionais = semiestruturado. |
| 18 | b | Propagação de rótulos amplifica erros — garbage in, garbage out em escala maior. |
| 19 | b | Em séries temporais, a ordem temporal é informação — embaralhar os dados destrói o significado. |

---

### Como se avaliar
- **16–19 acertos:** Impressionante! Você já conecta conceitos como um cientista de dados.
- **12–15 acertos:** Muito bom! Continue estudando os pontos fracos.
- **8–11 acertos:** Bom esforço! Revise os guias com foco nas conexões entre conceitos.
- **Menos de 8:** Volte às versões 1 e 2 e consolide a base antes do desafio.
