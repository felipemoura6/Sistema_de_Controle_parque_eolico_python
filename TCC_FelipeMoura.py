from matplotlib import pyplot as plt
import numpy as np
import random
import math


# ==================================================================================================================================
# Parâmetros do algoritmo genético

TAM_POPULACAO = 20      # Quantidade de indivíduos na população
NUM_GERACAO = 20        # Quantidade de gerações
TAXA_MUTACAO = 0.1       # Taxa de mutação
YAW_MIN = -30             # Ângulo mínimo do yaw
YAW_MAX = 30              # Ângulo máximo do yaw
COUNT = 10              # Número de repetições de gerações consecutivas para valores de aptidão iguais (CRITÉRIO DE PARADA)

u0=8.0          # Velocidade inicial do vento de 8m/s
r0=40.0         # Raio da turbina
D=2*r0
ct=0.8          # Coeficiente de arrasto
k=0.075         # Taxa de expansão da esteira (alfa)

layout_x = (0, 2, -2)        # Layout das coordenadas do eixo X
layout_y = (0, 100, 200)          # Layout das coordenadas do eixo Y
NUM_TURBINAS = len(layout_x)        # Número de turbinas no parque eólico

# True = Sim ---- False = Não
plotarAreaSombreada = False     # Plotar Area Sombreada
plotarOtimizacao = True        # Plotar Gráfico da otimização ao longo das gerações
printAG = False                 # Prints do Algoritmo Genético
plotarLayout = True             # Plotar Layout do parque
printAreaSombreada = True       # Prints sobre as Áreas Sombreadas das turbinas
# ==================================================================================================================================

def calcula_area_sombreada(x_i, x_j, r0, est):
    AsTotal=np.pi * r0**2
    xij = x_i - x_j   # Verifica em qual quadrante está
    if xij <= 0:
        raio_esteira = -est
    else:
        raio_esteira = est
    
    if xij <= 0:
        if xij + r0 <= raio_esteira:
            As = 0
            area_sombreada=0
        elif (xij + r0 > raio_esteira and xij - r0 < raio_esteira):
            y = np.arccos((raio_esteira**2 + xij**2 - r0**2) / (2 * xij * raio_esteira))
            b = np.arccos((r0**2 + xij**2 - raio_esteira**2) / (2 * xij * r0))
            As = np.pi * r0**2 - r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
            area_sombreada=100*As/AsTotal
        elif xij - r0 >= raio_esteira:
            As = AsTotal
            area_sombreada=100
    else:
        if xij - r0 >= raio_esteira:
            As = 0
            area_sombreada=0
        elif (xij - r0 < raio_esteira and xij + r0 > raio_esteira):
            y = np.arccos((raio_esteira**2 + xij**2 - r0**2) / (2 * xij * raio_esteira))
            b = np.arccos((r0**2 + xij**2 - raio_esteira**2) / (2 * xij * r0))
            As = r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
            area_sombreada=100*As/AsTotal
        elif xij + r0 <= raio_esteira:
            As = AsTotal
            area_sombreada=100

    if(printAreaSombreada == True):
        print("Area Sombreada(%): " , area_sombreada)
        print("Raio da esteira: ", est)

    return area_sombreada/100  # Percentual da área sombreada

def fitness_jensen(individuo, layout_x, layout_y, r0, u0, ct, k):
    
    u = [u0] * NUM_TURBINAS  # Inicializando velocidades
    Ve = [u0] * NUM_TURBINAS  # Inicializando velocidades
    
    for i in range(NUM_TURBINAS):
        x_i = layout_x[i]
        y_i = layout_y[i]
        for j in range(i): 
            x_j = layout_x[j] 
            y_j = layout_y[j]
                        
            if y_i > y_j:  # Apenas esteira para turbinas a jusante
                raio_esteira = r0 + k * (y_i - y_j)
                if abs(x_i - x_j)/2 < raio_esteira:  # Turbina está dentro da esteira
                  
                    if(printAreaSombreada): print('Calculo da area sombreada: Turbinas: T', j, ' - T', i)
                    As=calcula_area_sombreada(x_i, x_j, r0,raio_esteira)
                    
                    #numerador=(1-math.sqrt(1-ct)) * 2*r0**2
                    #denominador= np.pi*r0**2 * 2 * raio_esteira**2      
                    #Ve[i]= u0*(1-(numerador/denominador))   
                               
                    Ve[i]=u0*(1-(1-(math.sqrt(1-ct)))*(2*r0/(2*raio_esteira))**2)   ##  Formula da Tese de Jose Ricardo - pag 41   
                    
                    u[i]=Ve[i]*As+u0*(1-As) 
                    if(printAreaSombreada): print("AreaS=", As," V0=",u0, "||||||| Vsomb=", Ve[i]*As, " VNsomb=", u0*(1-As)," ||||||||| U[i]=", u[i])
                    
                    
        u[i] = u[i]*math.cos(math.radians(individuo[i]))    # Ajusta velocidade com base no yaw


    
    # Calcula a produção total de energia
    producao = sum(0.5 * 1.225 * math.pi * r0**2 * ui**3 for ui in u)  # Energia proporcional a v^3 sem descontar os ângulos
    producao_ajustada = producao * np.cos(np.radians(individuo[i]))**3      # Produção ajustada com o ângulo de inclinação de cada turbina
    return producao_ajustada/1000


# Inicialização da população
def populacao_inicial():
    
    # Cria a população inicial com ângulos yaw aleatórios para cada turbina.
    populacao = []
    for i in range(TAM_POPULACAO):
        # Cada indivíduo é um vetor de ângulos yaw (aleatórios entre YAW_MIN e YAW_MAX)
        individuo = [random.uniform(YAW_MIN, YAW_MAX) for j in range(NUM_TURBINAS)]
        populacao.append(individuo)
    print("")
    
    return populacao

# Seleção por torneio
def selecao(populacao, fitness_scores): 
    """
    Seleciona dois pais da população via seleção por torneio.
    """
    # Escolha de dois indivíduos aleatoriamente e selecionando o melhor
    tamanho_selecao = 3
    selected = random.sample(range(TAM_POPULACAO), tamanho_selecao)
    melhor_individual = max(selected, key=lambda x: fitness_scores[x])
    return populacao[melhor_individual]



# Cruzamento (Crossover)
def crossover(pai1, pai2):
    """
    pai1 = [-10, 5, 20]
    pai2 = [15, -5, 30]
    NUM_TURBINAS = 3
    Se o ponto de cruzamento selecionado for 1, o descendente será gerado como:

    Primeira parte do filho (até o ponto de cruzamento) vem de pai1: [-10]
    Segunda parte do filho (a partir do ponto de cruzamento) vem de pai2: [-5, 30]
              0    1   2
    filho = [-10, -5, 30]
    """
    
    crossover_point = random.randint(0, NUM_TURBINAS-1) # 0, 1 ou 2
    filho = pai1[:crossover_point] + pai2[crossover_point:]
    return filho

# Mutação
def mutacao(individuo):
    """
    Aplica mutação em um indivíduo.
    """
    for i in range(NUM_TURBINAS):
        if random.random() < TAXA_MUTACAO:
            individuo[i] = random.uniform(YAW_MIN, YAW_MAX)
    return individuo


# Algoritmo Genético
def genetic_algorithm():

    #fitness_jensen(layout_x, layout_y, r0, u0, ct, k)
    populacao = populacao_inicial()     # Inicializa a população
    melhor_fitness_anterior = 0
    count = 0
    melhor_fitness_historico=[]
    
    # Itera por várias gerações
    for geracao in range(NUM_GERACAO):
        if(printAG==True):
            print(f"\nGeração {geracao + 1}:")
            print("População e Aptidão:")

        # Calcula a aptidão de cada indivíduo
        fitness_scores = [fitness_jensen(individuo, layout_x, layout_y, r0, u0, ct, k) for individuo in populacao]
        
        # Itera e imprime cada indivíduo com uma numeração e sua aptidão
        for indice, (individuo, score) in enumerate(zip(populacao, fitness_scores), start=1):
            angulos_formatados = ", ".join(f"{angulo:.3f}°" for angulo in individuo)
            if(printAG==True): print(f"Indivíduo {indice}: [{angulos_formatados}], Fitness = {score:.2f} kW")
        
        # Encontra o melhor indivíduo da geração
        melhor_fitness = max(fitness_scores)
        melhor_individuo = populacao[fitness_scores.index(melhor_fitness)]
        melhor_fitness_historico.append(melhor_fitness)
        
        if(printAG==True): print(f"Melhor aptidão = {melhor_fitness:.2f}, Melhor indivíduo = {melhor_individuo}")
        
        # Critério de parada: verifica se a aptidão está saturada
        if melhor_fitness == melhor_fitness_anterior:
            count += 1
        else:
            count = 0
        melhor_fitness_anterior = melhor_fitness

        # Se a aptidão não alterar por COUNT gerações consecutivas
        if count >= COUNT:
            print(f"\nCritério de parada alcançado: Melhor aptidão não melhorou por {COUNT} gerações consecutivas.")
            break
        
        # Nova população
        nova_populacao = []
        
        # Mantém o melhor indivíduo (elitismo)
        nova_populacao.append(melhor_individuo)
        
        # Gera novos indivíduos
        while len(nova_populacao) < TAM_POPULACAO:
            # Seleciona dois pais
            pai1 = selecao(populacao, fitness_scores)
            pai2 = selecao(populacao, fitness_scores)
            
            # Realiza o cruzamento
            filho = crossover(pai1, pai2)
            
            # Aplica a mutação
            filho = mutacao(filho)
            
            # Adiciona o novo indivíduo à nova população
            nova_populacao.append(filho)
        
        # Substitui a população antiga pela nova
        populacao = nova_populacao
    
    # Melhor solução final encontrada
    if(printAG==True):
        print("\nMelhor configuração de ângulos yaw encontrada:", melhor_individuo)
        print(f'Aptidão da melhor solução:, {melhor_fitness:.2f}', "kW")
    
    # Plotando gráfico da otimização
    if(plotarOtimizacao==True):
        plt.plot(melhor_fitness_historico, marker='o', linestyle='-', color='b')
        plt.xlabel(f'Valor Final = {max(melhor_fitness_historico):.2f} kW') # Colocando o maior valor otimizado
        plt.ylabel('Melhor Aptidão (kW)')
        plt.title('Evolução da Aptidão ao Longo das Gerações')
        plt.grid(True)
        plt.show()

# Executa o algoritmo genético
print('==================================================')
print()
genetic_algorithm()



este = np.zeros(200) 
var_x = np.zeros(200) 
r0 = 30
est = 40
i = 0 
area_sombreada = np.zeros(200)

for x in np.arange(-100, 100, 1):
    x += 0.00001
    distancia = x
    AsTotal=np.pi * r0**2
    
    if x <= 0:
        raio_esteira = -est
    else:
        raio_esteira = est
    
    if x <= 0:
        if x + r0 <= raio_esteira:
            As = 0
            area_sombreada[i]=0
        elif (x + r0 > raio_esteira and x - r0 < raio_esteira):
            y = np.arccos((raio_esteira**2 + distancia**2 - r0**2) / (2 * distancia * raio_esteira))
            b = np.arccos((r0**2 + distancia**2 - raio_esteira**2) / (2 * distancia * r0))
            As = np.pi * r0**2 - r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
            area_sombreada[i]=100*As/AsTotal
        elif x - r0 >= raio_esteira:
            As = AsTotal
            area_sombreada[i]=100
    else:
        if x - r0 >= raio_esteira:
            As = 0
            area_sombreada[i]=0
        elif (x - r0 < raio_esteira and x + r0 > raio_esteira):
            y = np.arccos((raio_esteira**2 + distancia**2 - r0**2) / (2 * distancia * raio_esteira))
            b = np.arccos((r0**2 + distancia**2 - raio_esteira**2) / (2 * distancia * r0))
            As = r0**2 * (b - (np.sin(b) * np.cos(b))) + raio_esteira**2 * (y - (np.sin(y) * np.cos(y)))
            area_sombreada[i]=100*As/AsTotal
        elif x + r0 <= raio_esteira:
            As = AsTotal
            area_sombreada[i]=100
    
    este[i] = As
    var_x[i] = x
    i += 1 
    
# Plotando o gráfico
if(plotarAreaSombreada==True):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # 2 linha, 1 coluna

    # Plot da área da esteira
    axes[0].plot(var_x, este)
    axes[0].set_xlabel('Distância (x)')
    axes[0].set_ylabel('Área da Esteira (As)')
    axes[0].set_title('Área da Esteira em Função da Distância')
    axes[0].grid(True)

    # Plot da redução
    axes[1].plot(var_x, area_sombreada)
    axes[1].set_xlabel('Distância (x)')
    axes[1].set_ylabel('Área sombreada (%)')
    axes[1].set_title('Area sombreada em Função da Distância')
    axes[1].grid(True)


    plt.tight_layout()
    plt.show()


# Definir as coordenadas dos pontos
x = layout_x  # Coordenadas x dos pontos
y = layout_y  # Coordenadas y dos pontos

limitEsquerdoX = np.array(layout_x) - r0    # Posição do limite esquerdo das turbinas
limitDireitoX = np.array(layout_x) + r0     # Posição do limite direito das turbinas

cores = ['red', 'blue', 'green']    # Definindo as cores para cada turbina

# Plotar as turbinas
if(plotarLayout==True):
    plt.scatter(x, y, color="blue", label='Turbinas')
    plt.scatter(limitEsquerdoX, y, color="red", label='Limite pá')
    plt.scatter(limitDireitoX, y, color="red")
    # Adicionar rótulos ao gráfico
    plt.title('Parque 2D')
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')

    # Exibir a legenda
    plt.legend()

    # Exibir o gráfico
    plt.grid(True)
    plt.show()
    
    
print('==================================================')
print()
print()
