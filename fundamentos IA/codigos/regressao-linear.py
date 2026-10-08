from sklearn.linear_model import LinearRegression

# Dados: tamanho da casa (m²) -> preço (mil R$)
X = [[50], [70], [90], [110], [130]]   # tamanho
y = [200, 260, 330, 390, 450]          # preço

modelo = LinearRegression()
modelo.fit(X, y)   # encontra o melhor 'a' e 'b'

print("Inclinação (a):", modelo.coef_[0])      # ~ preço por m²
print("Intercepto (b):", modelo.intercept_)    # valor-base

# Prever o preço de uma casa de 100 m²
print("Previsão 100m²:", modelo.predict([[100]]))