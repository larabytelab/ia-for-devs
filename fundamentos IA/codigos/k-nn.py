from sklearn.neighbors import KNeighborsClassifier

# Dados de treino: [peso, doçura]
X = [
    [150, 7], [170, 6], [140, 8],   # maçãs
    [130, 4], [120, 3], [160, 5],   # laranjas
]
# Rótulos correspondentes
y = ["maçã", "maçã", "maçã", "laranja", "laranja", "laranja"]

# Cria o modelo com K = 3
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X, y)   # "treina" (na verdade só memoriza os dados)

# Classifica a fruta nova
fruta_nova = [[155, 7]]
print(modelo.predict(fruta_nova))          # -> ['maçã']
print(modelo.predict_proba(fruta_nova))    # probabilidade por classe