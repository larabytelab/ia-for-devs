# Guia de Estudo — Big Data, Estatística e Machine Learning

Um resumo de tudo o que conversamos, com analogias para fixar, seguido de um
questionário de múltipla escolha para você responder sozinho e o gabarito para
conferir depois.

> Dica de estudo: leia o mapa, tente responder o questionário **sem** olhar o
> gabarito, e só depois compare. Onde você errar, volte na explicação daquele conceito.

---
Mapa resumo — cada conceito pela sua analogia
Pense em toda a nossa conversa como uma escada: cada degrau se apoia no anterior.
Índice invertido (Lucene) — o índice remissivo no fim de um livro. Em vez de folhear tudo atrás de uma palavra, você já tem a lista de onde ela aparece. Ele inverte a lógica: de "documento → palavras" para "palavra → documentos".
Hadoop — o mutirão de um censo com mil recenseadores. Em vez de uma pessoa contar a cidade toda, cada um conta seu bairro ao mesmo tempo. Divide-se em HDFS (o armário que quebra os dados em blocos e guarda cópias), MapReduce (Map = cada um conta seu pedaço; Reduce = soma tudo) e YARN (o gerente que distribui as tarefas).
Extração de recursos — traduzir para a língua da máquina. A fruta vira "cor, peso, formato"; a compra vira "valor, hora, distância". Sem números, nenhum algoritmo funciona.
Desvio padrão — a régua da bagunça. Duas turmas com média 7 podem ser mundos diferentes: ele mede se os dados estão grudados na média ou espalhados.
Distribuição normal — o sino. A maioria fica no meio, os extremos são raros. O desvio padrão define se o sino é magro ou gordo (regra 68-95-99).
Bayes — o detetive (ou o médico). Parte de um palpite inicial e o revisa a cada nova pista que surge.
Correlação — o "andam de mãos dadas". Sorvete e afogamento sobem juntos, mas um não causa o outro — mãos dadas não é um empurrando o outro.
E no sistema de fraude, todos viram uma orquestra: extração vira números → desvio padrão mede a fuga do padrão → o sino diz que é raríssima → correlação vê a combinação suspeita → Bayes pesa tudo e decide. O detalhe genial é que o Bayes usa o desvio padrão e a curva normal por dentro dele.
Agora o guia de estudo:

## Parte 1 — Mapa resumo (com analogias)

Pense em toda a jornada como uma escada: cada degrau se apoia no anterior.

| Conceito | Analogia para lembrar | Essência em uma frase |
|---|---|---|
| Índice invertido (Lucene) | Índice remissivo no fim de um livro | Vira a lógica: em vez de "documento → palavras", guarda "palavra → documentos". |       
| Hadoop | Mutirão de um censo com mil recenseadores | Guarda e processa dados gigantes em muitos computadores comuns em paralelo. |
| Extração de recursos | Traduzir para a "língua" da máquina | Transforma dados crus (fruta, compra, texto) em números úteis (features). |
| Desvio padrão | A régua da bagunça | Mede se os dados estão grudados na média ou espalhados. |
| Distribuição normal | O sino (bell curve) | A maioria fica no meio; extremos são raros. O desvio define a largura do sino. |
| Teorema de Bayes | O detetive / o médico | Parte de um palpite inicial e o revisa a cada nova pista. |
| Correlação | "Andam de mãos dadas" | Mede se duas coisas variam juntas (−1 a +1). Cuidado: não é causa. |

### Como o Hadoop se divide
- HDFS: o "armário" que quebra o arquivo em blocos e guarda cópias em várias máquinas (tolerância a falhas).
- MapReduce: o processamento em duas fases — Map (cada um conta seu pedaço) e Reduce (soma tudo).
- YARN: o "gerente de recursos" que distribui as tarefas pelo cluster.
- Hoje o Apache Spark costuma substituir o MapReduce puro (processa na memória, mais rápido).

### Como os 5 conceitos de estatística se juntam no sistema de fraude
Extração de recursos (vira números) → desvio padrão (mede o quanto foge do padrão)
→ distribuição normal (diz que essa fuga é raríssima) → correlação (vê que as pistas
formam uma combinação típica de golpe) → Bayes (pesa todas as evidências e calcula a
probabilidade final). Detalhe genial: o Bayes usa o desvio padrão e a curva normal
**por dentro** para pesar cada pista (Naive Bayes Gaussiano).

---

## Parte 2 — Questionário de múltipla escolha (responda antes de ver o gabarito)

**1. Por que o índice do Lucene é chamado de "invertido"?**
- (a) Porque lê os documentos de trás para frente
- (b) Porque mapeia "palavra → documentos", invertendo a relação "documento → palavras" 
- (c) Porque inverte a ordem alfabética das palavras
- (d) Porque comprime os documentos para ocupar menos espaço

**2. Qual analogia representa melhor o índice invertido?**
- (a) Uma fila de banco
- (b) O índice remissivo no fim de um livro 
- (c) Uma pilha de pratos
- (d) Um dicionário de sinônimos

**3. Quais são os dois componentes centrais do Hadoop?**
- (a) HDFS e MapReduce 
- (b) Lucene e Solr
- (c) Spark e Kafka
- (d) YARN e Bayes

**4. Para que serve o HDFS?**
- (a) Processar os dados em duas fases (Map e Reduce)
- (b) Armazenar os dados quebrando-os em blocos, com cópias em várias máquinas 
- (c) Gerenciar os recursos do cluster
- (d) Fazer buscas rápidas por palavra

**5. Por que o Hadoop "leva o processamento até onde os dados estão"?**
- (a) Porque é mais barato comprar novos computadores
- (b) Porque mover grandes volumes de dados pela rede é lento e caro 
- (c) Porque os dados não podem ser copiados por lei
- (d) Porque o MapReduce exige isso por segurança

**6. Duas turmas têm média 7, mas a Turma B tem notas muito mais espalhadas. O que mede essa diferença?**
- (a) A média
- (b) A mediana
- (c) O desvio padrão 
- (d) A moda

**7. Na regra 68-95-99 da distribuição normal, quantos % dos dados ficam a ~2 desvios padrão da média?**
- (a) 50% 
- (b) 68%
- (c) 95%
- (d) 99%

**8. Qual é a ideia central do Teorema de Bayes?**
- (a) Somar todas as probabilidades até chegar a 100%
- (b) Atualizar uma crença inicial conforme surgem novas evidências
- (c) Provar a relação de causa entre duas variáveis
- (d) Encontrar a média de um conjunto de dados 

**9. No verão, vendas de sorvete e afogamentos sobem juntos. O que isso ilustra?**
- (a) Que o sorvete causa afogamentos
- (b) Correlação sem causalidade — uma terceira variável (o calor) causa os dois 
- (c) Uma distribuição normal perfeita
- (d) Um erro na extração de recursos

**10. O que significa "extrair features" (extração de recursos)?**
- (a) Apagar dados irrelevantes do banco
- (b) Transformar dados brutos em números que representam suas características 
- (c) Ordenar os dados por relevância
- (d) Criar cópias de segurança dos dados

**11. No sistema de fraude, uma compra teve z-score de 51. O que isso indica?**
- (a) Que ela tem 51% de chance de ser fraude
- (b) Que está a 51 desvios padrão da média — raríssimo, portanto suspeito 
- (c) Que existiram 51 compras parecidas no histórico
- (d) Que a correlação com fraude foi de 0,51

**12. No Naive Bayes Gaussiano do código, qual conceito funciona "por dentro" para pesar cada pista?**
- (a) O índice invertido
- (b) O MapReduce
- (c) A distribuição normal (usando a média e o desvio padrão) 
- (d) A mediana 

**13. (Aplicada) Uma compra de R$ 500 às 22h, a 3 km de casa, com 6h desde a última compra. Como tende a ser classificada?**
- (a) Fraude, porque o valor é alto X
- (b) Legítima, porque valor, horário, distância e intervalo batem com o padrão normal 
- (c) Fraude, apenas por causa do horário
- (d) Impossível saber sem mais dados

**14. (Desafio) Por que o código soma logaritmos das probabilidades em vez de multiplicá-las diretamente?**
- (a) Para deixar o código mais curto
- (b) Para evitar que números minúsculos multiplicados sejam arredondados a zero (underflow) 
- (c) Porque logaritmo é sempre mais rápido de calcular
- (d) Porque o Teorema de Bayes proíbe multiplicação

---

## Parte 3 — Gabarito

| Questão | Resposta | Por quê (resumido) |
|---|---|---|
| 1 | (b) | Ele inverte a relação: guarda "palavra → em quais documentos aparece". |
| 2 | (b) | É exatamente como procurar uma palavra no índice remissivo de um livro. |
| 3 | (a) | HDFS (armazenamento) + MapReduce (processamento) são o núcleo do Hadoop. |
| 4 | (b) | O HDFS quebra o arquivo em blocos e guarda cópias em máquinas diferentes. |
| 5 | (b) | Mover terabytes pela rede é caro; melhor levar o programa até o dado. |
| 6 | (c) | O desvio padrão mede o espalhamento que a média sozinha esconde. |
| 7 | (c) | ~68% a 1 desvio, ~95% a 2 desvios, ~99% a 3 desvios. |
| 8 | (b) | Bayes atualiza a probabilidade a cada nova evidência (como um médico). |
| 9 | (b) | Correlação não é causa; o calor é a variável que puxa os dois. |
| 10 | (b) | Features são as características numéricas extraídas dos dados brutos. |
| 11 | (b) | z-score = quantos desvios padrão longe da média; 51 é raríssimo. |
| 12 | (c) | A curva normal (com desvio padrão) mede a verossimilhança de cada pista. |
| 13 | (b) | Todas as pistas combinam com o padrão legítimo; Bayes daria prob. baixa. |
| 14 | (b) | Multiplicar muitas probabilidades minúsculas causa underflow; log evita. |

---

Bons estudos! Volte neste guia sempre que precisar reforçar as conexões entre os conceitos.
