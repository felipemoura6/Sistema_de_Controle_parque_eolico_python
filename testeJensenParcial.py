import numpy as np
import matplotlib.pyplot as plt

# Inicializando variáveis
i = 0
este = np.zeros(201)
est = 30
raio_esteira = 0        # Raio da esteira
r0 = 20      # Raio da turbina
var_x = np.zeros(201)

# Loop para calcular a área da esteira
for x in np.arange(-100, 101, 1):
    x += 0.00001
    xij = x
    
    if x <= 0:
        raio_esteira = -est
    else:
        raio_esteira = est
    
    if x <= 0:
        if x + r0 <= raio_esteira:
            As = 0
        elif (x + r0 > raio_esteira and x - r0 < raio_esteira):
            y = np.arccos((raio_esteira**2 + xij**2 - r0**2) / (2 * xij * raio_esteira))
            b = np.arccos((r0**2 + xij**2 - raio_esteira**2) / (2 * xij * r0))
            As = np.pi * r0**2 - r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
        elif x - r0 >= raio_esteira:
            As = np.pi * r0**2
    else:
        if x - r0 >= raio_esteira:
            As = 0
        elif (x - r0 < raio_esteira and x + r0 > raio_esteira):
            y = np.arccos((raio_esteira**2 + xij**2 - r0**2) / (2 * xij * raio_esteira))
            b = np.arccos((r0**2 + xij**2 - raio_esteira**2) / (2 * xij * r0))
            As = r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
        elif x + r0 <= raio_esteira:
            As = np.pi * r0**2
    
    este[i] = As
    var_x[i] = x
    i = i+1

# Plotando o gráfico
plt.plot(var_x, este)
plt.xlabel('Distância (x)')
plt.ylabel('Área da Esteira (As)')
plt.title('Área da Esteira em Função da Distância')
plt.grid(True)
plt.show()
