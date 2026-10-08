from sklearn.cluster import KMeans
import numpy as np

# Repare: NÃO existe um 'y' com rótulos. Só os dados.
X = np.array([
    [170,120],[180,90],[150,140],   # aglomerado 1
    [440,150],[420,180],[460,130],  # aglomerado 2
    [300,320],[280,340],[320,300],  # aglomerado 3
])

modelo = KMeans(n_clusters=3, random_state=42)
modelo.fit(X)

print(modelo.labels_)            # a que grupo cada ponto foi parar
print(modelo.cluster_centers_)   # a posição final dos centros (✕)

# Prever o grupo de um ponto novo
print(modelo.predict([[430, 160]]))   # provavelmente o grupo 2