"""
=====================================================================
 SISTEMA DE DETECÇÃO DE FRAUDE EM CARTÃO DE CRÉDITO (versão didática)
=====================================================================

Este programa usa os 5 conceitos que estudamos, cada um numa fase:

  FASE 1 - Extração de recursos   -> transforma a compra em números
  FASE 2 - Desvio padrão          -> mede o quanto a compra foge do padrão
  FASE 3 - Distribuição normal    -> diz o quão rara é essa fuga
  FASE 4 - Correlação             -> vê quais pistas andam junto com fraude
  FASE 5 - Bayes (Gaussian NB)    -> junta tudo numa probabilidade final

Escrito só com a biblioteca padrao do Python, de proposito, para que
as formulas fiquem visiveis. Basta rodar:  python3 deteccao_fraude.py
"""

import math
import random
import statistics

random.seed(42)  # deixa o resultado sempre igual, para estudo


# =====================================================================
# FASE 1 - EXTRACAO DE RECURSOS (feature extraction)
# ---------------------------------------------------------------------
# Uma "compra" no mundo real e uma coisa complexa. O computador nao a
# entende. Entao a transformamos numa lista de NUMEROS (as "features").
# =====================================================================

NOMES_FEATURES = ["valor", "hora", "distancia_km", "min_desde_ultima"]


def extrair_features(transacao: dict) -> list:
    """Pega uma compra (dicionario) e devolve so os numeros que importam."""
    return [
        transacao["valor"],             # quanto foi gasto (R$)
        transacao["hora"],              # hora do dia (0 a 23)
        transacao["distancia_km"],      # distancia da ultima compra (km)
        transacao["min_desde_ultima"],  # minutos desde a ultima compra
    ]


def gerar_historico():
    """Simula o passado que o banco ja viu: compras legitimas e fraudes."""
    legitimas, fraudes = [], []

    # Compras NORMAIS: valores baixos, de dia, pertinho, intervalos grandes.
    for _ in range(300):
        legitimas.append({
            "valor": max(1, random.gauss(150, 80)),
            "hora": min(23, max(0, random.gauss(14, 4))),
            "distancia_km": max(0, random.gauss(8, 10)),
            "min_desde_ultima": max(1, random.gauss(300, 150)),
        })

    # FRAUDES: valores altos, de madrugada, longe, em rajada (pouco intervalo).
    for _ in range(60):
        fraudes.append({
            "valor": max(1, random.gauss(2200, 1500)),
            "hora": random.choice([0, 1, 2, 3, 3, 4]),
            "distancia_km": max(0, random.gauss(500, 350)),
            "min_desde_ultima": max(1, random.gauss(18, 12)),
        })

    return legitimas, fraudes


# =====================================================================
# FASE 2 e 3 - DESVIO PADRAO + DISTRIBUICAO NORMAL
# ---------------------------------------------------------------------
# Aprendemos a media e o desvio padrao dos gastos normais do cliente.
# O "z-score" diz quantos desvios padrao a compra nova esta longe da
# media; a curva normal (o sino) diz o quao rara e essa distancia.
# =====================================================================

def z_score(valor, media, desvio):
    """Quantos desvios padrao o 'valor' esta longe da media."""
    if desvio == 0:
        return 0.0
    return (valor - media) / desvio


def densidade_normal(x, media, desvio):
    """
    A formula do sino (densidade da distribuicao normal).
    Alta quando x esta perto da media (comum); baixa quando esta longe (raro).
    """
    if desvio == 0:
        return 1.0
    coef = 1 / (desvio * math.sqrt(2 * math.pi))
    expoente = -((x - media) ** 2) / (2 * desvio ** 2)
    return coef * math.exp(expoente)


# =====================================================================
# FASE 4 - CORRELACAO (Pearson)
# ---------------------------------------------------------------------
# Um numero de -1 a +1 que diz se uma feature "anda junto" com a fraude.
# Implementado na mao para voce ver a formula por dentro.
# =====================================================================

def correlacao_pearson(xs, ys):
    media_x, media_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - media_x) * (y - media_y) for x, y in zip(xs, ys))
    var_x = math.sqrt(sum((x - media_x) ** 2 for x in xs))
    var_y = math.sqrt(sum((y - media_y) ** 2 for y in ys))
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y)


# =====================================================================
# FASE 5 - BAYES (Naive Bayes Gaussiano)
# ---------------------------------------------------------------------
# Aqui TODAS as ideias se abracam. Para cada classe (fraude/legitima)
# aprendemos media e desvio de cada feature. Para uma compra nova, Bayes
# combina o prior (quase tudo e legitimo) com as evidencias, usando a
# CURVA NORMAL como verossimilhanca. Ou seja: Bayes usa desvio padrao +
# distribuicao normal por dentro!
# =====================================================================

class DetectorBayes:
    def treinar(self, legitimas, fraudes):
        total = len(legitimas) + len(fraudes)
        self.prior = {
            "legitima": len(legitimas) / total,
            "fraude": len(fraudes) / total,
        }
        self.stats = {
            "legitima": self._stats_por_feature(legitimas),
            "fraude": self._stats_por_feature(fraudes),
        }

    def _stats_por_feature(self, transacoes):
        colunas = [extrair_features(t) for t in transacoes]
        resumo = []
        for i in range(len(NOMES_FEATURES)):
            valores = [linha[i] for linha in colunas]
            resumo.append((statistics.mean(valores), statistics.pstdev(valores)))
        return resumo

    def probabilidade_fraude(self, transacao):
        features = extrair_features(transacao)
        # Usamos log para nao "sumir" com numeros minusculos multiplicados.
        scores = {}
        for classe in ("legitima", "fraude"):
            log_prob = math.log(self.prior[classe])        # comeca no prior
            for i, x in enumerate(features):
                media, desvio = self.stats[classe][i]
                verossimilhanca = densidade_normal(x, media, desvio)  # altura no sino
                log_prob += math.log(verossimilhanca + 1e-12)         # evita log(0)
            scores[classe] = log_prob

        # Bayes: normaliza para virar probabilidade de 0 a 1.
        maior = max(scores.values())
        p_leg = math.exp(scores["legitima"] - maior)
        p_fra = math.exp(scores["fraude"] - maior)
        return p_fra / (p_leg + p_fra)


# =====================================================================
# JUNTANDO TUDO - a "orquestra dos 5 conceitos"
# =====================================================================

def main():
    print("=" * 62)
    print(" SISTEMA DE DETECCAO DE FRAUDE - os 5 conceitos em acao")
    print("=" * 62)

    legitimas, fraudes = gerar_historico()

    compra_suspeita = {
        "valor": 4000,          # R$ 4.000
        "hora": 3,              # 3h da manha
        "distancia_km": 800,    # 800 km de distancia
        "min_desde_ultima": 20, # 20 min desde a ultima compra
    }

    # ---------- FASE 1: EXTRACAO DE RECURSOS ----------
    print("\n[FASE 1] Extracao de recursos - a compra virou numeros:")
    for nome, valor in zip(NOMES_FEATURES, extrair_features(compra_suspeita)):
        print(f"    {nome:<18} = {valor:.0f}")

    # ---------- FASE 2 e 3: DESVIO PADRAO + NORMAL ----------
    valores_normais = [t["valor"] for t in legitimas]
    media = statistics.mean(valores_normais)
    desvio = statistics.pstdev(valores_normais)
    z = z_score(compra_suspeita["valor"], media, desvio)

    print("\n[FASE 2+3] Desvio padrao e distribuicao normal (so o valor):")
    print(f"    Gasto medio do cliente : R$ {media:.0f}")
    print(f"    Desvio padrao          : R$ {desvio:.0f}")
    print(f"    Compra suspeita        : R$ {compra_suspeita['valor']:.0f}")
    print(f"    -> z-score = {z:.1f}  (esta a {z:.0f} desvios padrao da media!)")
    print(f"    Pela curva normal, algo tao longe e rarissimo (<< 1%).")

    # ---------- FASE 4: CORRELACAO ----------
    todas = [(t, 0) for t in legitimas] + [(t, 1) for t in fraudes]
    rotulos = [rotulo for _, rotulo in todas]

    print("\n[FASE 4] Correlacao de cada pista com fraude (-1 a +1):")
    for i, nome in enumerate(NOMES_FEATURES):
        coluna = [extrair_features(t)[i] for t, _ in todas]
        corr = correlacao_pearson(coluna, rotulos)
        seta = "sobe -> mais fraude" if corr > 0 else "sobe -> menos fraude"
        print(f"    {nome:<18} corr = {corr:+.2f}  ({seta})")
    print("    (Lembre: correlacao != causa. E so mais uma evidencia.)")

    # ---------- FASE 5: BAYES ----------
    detector = DetectorBayes()
    detector.treinar(legitimas, fraudes)
    prob = detector.probabilidade_fraude(compra_suspeita)

    print("\n[FASE 5] Bayes junta TODAS as evidencias:")
    print(f"    Crenca inicial (prior) de fraude: {detector.prior['fraude']*100:.0f}%")
    print(f"    Depois de pesar todas as pistas -> {prob*100:.1f}% de chance de fraude")

    print("\n" + "=" * 62)
    veredito = "BLOQUEAR e avisar o cliente" if prob > 0.8 else "Aprovar"
    print(f" VEREDITO: {veredito}  (probabilidade de fraude: {prob*100:.1f}%)")
    print("=" * 62)

    # ---- Comparacao: uma compra normal ----
    compra_normal = {"valor": 120, "hora": 13, "distancia_km": 5, "min_desde_ultima": 400}
    prob_normal = detector.probabilidade_fraude(compra_normal)
    print(f"\n(Comparacao) Compra comum de R$120 ao meio-dia:"
          f" {prob_normal*100:.1f}% de fraude -> Aprovada.")


if __name__ == "__main__":
    main()