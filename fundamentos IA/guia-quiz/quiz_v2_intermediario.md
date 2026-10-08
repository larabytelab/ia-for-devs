# Quiz de Machine Learning — Versão 2: Aplicação Prática

> Responda todas as questões antes de conferir o gabarito no final.
> Nível: intermediário — foco em "dado esse problema, qual ferramenta usar?"

---

## Questões

**1.** Uma startup quer recomendar filmes a novos usuários. O dataset tem filmes com rótulos (ação, comédia, terror, etc.). Qual algoritmo é mais adequado para classificar um filme novo?

- a) K-Means
- b) DBSCAN
- c) K-NN
- d) Regressão Linear

---

**2.** Você quer prever o salário de um funcionário com base em seus anos de experiência. Qual técnica é mais indicada?

- a) Árvore de Decisão
- b) Regressão Linear
- c) K-Means
- d) Naive Bayes Gaussiano

---

**3.** Um analista tem dados de clientes de um e-commerce, mas **sem nenhum rótulo** ou categoria definida. Ele quer descobrir grupos naturais de comportamento. Qual algoritmo faz mais sentido?

- a) K-NN
- b) Regressão Linear
- c) K-Means ou DBSCAN
- d) Árvore de Decisão

---

**4.** Você está analisando dados de temperatura em várias cidades. Algumas cidades têm valores muito extremos (ex.: 72°C registrado por erro). Qual ferramenta estatística ajuda a identificar esses outliers?

- a) Correlação de Pearson
- b) Teorema de Bayes
- c) Z-score
- d) Distribuição uniforme

---

**5.** Um estudo mostra que cidades com mais sorveterias têm mais afogamentos. O gestor conclui que "sorvete causa afogamentos". O que está errado nesse raciocínio?

- a) O gestor usou o algoritmo errado
- b) Correlação não implica causalidade — ambos são influenciados pelo calor no verão
- c) Afogamentos e sorvete têm correlação negativa, não positiva
- d) O estudo usou uma amostra muito pequena

---

**6.** Você treina uma Árvore de Decisão e ela tem 100% de acerto nos dados de treino, mas só 60% nos dados novos. Qual é o problema?

- a) Underfitting — o modelo é simples demais
- b) Overfitting — o modelo memorizou os dados de treino
- c) O dataset estava desbalanceado
- d) A árvore não tem folhas suficientes

---

**7.** Um sistema de e-mail quer classificar mensagens como "spam" ou "não spam" com base em palavras-chave. Qual algoritmo é clássico para esse tipo de problema?

- a) K-Means
- b) Naive Bayes
- c) Regressão Linear
- d) DBSCAN

---

**8.** Você tem dados de vendas de sorvete e temperatura. Ao calcular a correlação, encontra **+0.92**. O que isso indica?

- a) A temperatura causa o aumento nas vendas
- b) As duas variáveis têm forte relação negativa
- c) As duas variáveis têm forte relação positiva — quando uma sobe, a outra tende a subir também
- d) A correlação é fraca e não diz nada relevante

---

**9.** Ao aplicar K-NN para detectar fraudes, o sistema começa a confundir transações legítimas com fraudes. O engenheiro percebe que K=1. O que provavelmente está causando o problema?

- a) K=1 é muito grande e ignora os vizinhos próximos
- b) K=1 é muito pequeno e o modelo fica sensível demais ao ruído
- c) K-NN não serve para detecção de fraude
- d) O dataset não tem rótulos suficientes

---

**10.** Um pesquisador tem dados de GPS de clientes em uma cidade. Alguns clientes estão em regiões muito isoladas, longe de qualquer grupo. Qual algoritmo identifica melhor esses pontos isolados?

- a) K-NN com K pequeno
- b) Regressão Linear
- c) DBSCAN (que os marcará como ruído/outlier)
- d) K-Means com K grande

---

**11.** Em um sistema de busca textual (como o Google), por que usamos um **índice invertido** ao invés de varrer todos os documentos?

- a) Porque o índice invertido comprime os arquivos de texto
- b) Porque o índice invertido mapeia palavras → documentos, tornando a busca muito mais rápida
- c) Porque documentos muito grandes não podem ser lidos diretamente
- d) Porque o índice invertido elimina palavras duplicadas

---

**12.** Um médico quer prever se um paciente tem diabetes com base em 10 exames clínicos. Cada exame é uma "feature". Antes de treinar o modelo, os exames têm escalas muito diferentes (ex.: glicose em 80–150 vs. pressão em 70–200). Por que isso pode ser problemático para o K-NN?

- a) K-NN não aceita mais de 5 features
- b) K-NN usa distância — se as escalas forem muito diferentes, features com valores altos dominam a distância injustamente
- c) K-NN só funciona com features binárias
- d) Não há problema — K-NN é insensível à escala dos dados

---

**13.** No Apache Hadoop, o MapReduce processa dados em duas fases principais. Qual é a função da fase **Reduce**?

- a) Dividir os dados em blocos para distribuir nos nós
- b) Aplicar transformações independentes em cada pedaço do dado
- c) Agregar e combinar os resultados intermediários gerados pelo Map
- d) Enviar os dados processados de volta ao usuário

---

**14.** Uma empresa de saúde quer treinar um modelo para classificar exames de raio-X como "saudável" ou "doença". Eles têm **500 exames rotulados por médicos** e **50.000 exames sem rótulo**. Qual abordagem faz mais sentido?

- a) Supervisionado — usar apenas os 500 rotulados
- b) Não supervisionado — ignorar os rótulos e usar K-Means
- c) Semissupervisionado — usar os 500 rotulados + os 50.000 não rotulados juntos
- d) Regressão Linear — prever um valor numérico para cada exame

---

**15.** Um desenvolvedor recebe um arquivo de log de servidor no seguinte formato:

```
{"timestamp": "2024-01-01T10:00:00", "nivel": "erro", "mensagem": "timeout na conexão"}
{"timestamp": "2024-01-01T10:01:00", "nivel": "info", "mensagem": "requisição concluída"}
```

Qual tipo de dado esse arquivo representa?

- a) Dado estruturado — está em tabela com colunas fixas
- b) Dado não estruturado — não tem nenhuma organização
- c) Dado semiestruturado — usa JSON com chaves definidas, mas sem esquema rígido de tabela
- d) Dado temporal — porque tem timestamp

---

**16.** Vendas diárias de um produto durante 2 anos, coletadas automaticamente pelo sistema, constituem que tipo de dado? E qual característica especial eles têm?

- a) Dado estruturado sem nenhuma característica especial
- b) Dado não estruturado com padrões ocultos
- c) Dado temporal (série temporal) — a ordem e o tempo entre os registros importam para a análise
- d) Dado semiestruturado com tags de data

---

**17.** Por que o aprendizado semissupervisionado é útil na prática?

- a) Porque é sempre mais preciso do que o supervisionado puro
- b) Porque rotular dados manualmente é caro e demorado — aproveitar dados não rotulados reduz custo mantendo boa precisão
- c) Porque elimina a necessidade de qualquer dado rotulado
- d) Porque funciona apenas com dados de texto e imagem

---

## Gabarito

| Questão | Resposta | Explicação rápida |
|---------|----------|-------------------|
| 1 | c | K-NN é supervisionado e classifica com base nos vizinhos mais próximos. |
| 2 | b | Regressão Linear é ideal para prever valores contínuos. |
| 3 | c | Sem rótulos = não supervisionado; K-Means ou DBSCAN se encaixam. |
| 4 | c | Z-score indica quantos desvios padrão um ponto está da média — valores extremos (|z|>3) são suspeitos. |
| 5 | b | Correlação ≠ causalidade. Ambos sobem no verão por causa do calor. |
| 6 | b | Árvore muito profunda memoriza o ruído — overfitting clássico. |
| 7 | b | Naive Bayes é o algoritmo histórico para classificação de texto e spam. |
| 8 | c | +0.92 é correlação muito forte e positiva — as variáveis sobem juntas. |
| 9 | b | K=1 → apenas 1 vizinho decide — muito sensível a outliers e ruído. |
| 10 | c | DBSCAN rotula pontos isolados como -1 (ruído), perfeito para isso. |
| 11 | b | O índice invertido é o "índice do livro" digital: palavra → onde está. |
| 12 | b | K-NN é sensível à escala; features em escalas maiores dominam a distância. |
| 13 | c | Reduce agrega os resultados do Map — soma, contagem, combinação, etc. |
| 14 | c | Semissupervisionado aproveita poucos rótulos + muitos não rotulados — ideal quando rotular é caro. |
| 15 | c | JSON tem chaves e estrutura, mas não é uma tabela — é semiestruturado. Atenção: o timestamp é uma feature, não define o tipo do dado. |
| 16 | c | Série temporal: a ordem cronológica importa — tendências, sazonalidade e padrões ao longo do tempo. |
| 17 | b | O custo de anotação humana é o principal motivador do aprendizado semissupervisionado. |

---

### Como se avaliar
- **14–17 acertos:** Excelente! Você já pensa como um praticante. Passe para a Versão 3.
- **10–13 acertos:** Muito bom! Revise as questões erradas focando no "por quê".
- **Menos de 10:** Faça a Versão 1 primeiro e revise os conceitos de aplicação.
