from matplotlib import pyplot as plt
import numpy as np
import random
import math


# ==================================================================================================================================
# Parâmetros do algoritmo genético
NUM_TURBINAS = 3        # Número de turbinas no parque eólico
TAM_POPULACAO = 20      # Quantidade de indivíduos na população
NUM_GERACAO = 10        # Quantidade de gerações
TAXA_MUTACAO = 0.1       # Taxa de mutação
YAW_MIN = -30             # Ângulo mínimo do yaw
YAW_MAX = 30              # Ângulo máximo do yaw
COUNT = 10              # Número de repetições de gerações consecutivas para valores de aptidão iguais (CRITÉRIO DE PARADA)

u0=8.0          # Velocidade inicial do vento de 8m/s
r0=40.0         # Raio da turbina
ct=0.8          # Coeficiente de arrasto
k=0.075         # Taxa de expansão da esteira (alfa)

layout_x = (0, 5, 10)        # Layout das coordenadas do eixo X
layout_y = (0, 0, 0)          # Layout das coordenadas do eixo Y

# ==================================================================================================================================



# # Função de aptidão
# def fitness(individuo):
#     """
#     Função de aptidão que simula a produção de energia com base nos ângulos yaw.
#     """
#     producao = 0
#     for i in range(NUM_TURBINAS):
#         # Mais próximo do ângulo ótimo (0), maior a energia.
#         producao += 100 - abs(individuo[i])
    
    
#     # Penalização por turbulência
#     penalidade = sum(abs(individuo[i] - individuo[i-1]) for i in range(1, NUM_TURBINAS))
    
#     return producao - penalidade



def fitness_jensen(individuo, layout_x, layout_y, r0, u0, ct, k):
    
    deltinha = np.zeros(NUM_TURBINAS)       # Inicializando o vetor de reduções da velocidade do vento
    
    u = [u0] * NUM_TURBINAS  # Inicializando velocidades
    
    for i in range(NUM_TURBINAS):
        x_i = layout_x[i]
        y_i = layout_y[i]
        for j in range(i): 
            x_j = layout_x[j] 
            y_j = layout_y[j]
            distancia = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
            
            if x_i > x_j:  # Apenas esteira para turbinas a jusante
                raio_esteira = r0 + k * distancia
                if abs(y_i - y_j)/2 < raio_esteira:  # Turbina está dentro da esteira
                    fator_reducao = 1-((1 - np.sqrt(1 - ct)) / (1 + k * distancia / r0)**2)   
                    
                    if(i==0):   # Verificação se é a primeira turbina
                        u[i]=u0*(1-fator_reducao)
                        deltinha[i]=0
                        
                    if(i>0):    # Exceto primeira turbina
                        u[i]=u[i-1]*(1-fator_reducao)
                        deltinha[i]=1-u[i]/u[i-1]
                        
                    soma_quadrados = sum([x**2 for x in deltinha])  # Somatório dos quadrados
                    deltinha[-1] = math.sqrt(soma_quadrados)    # deltinha médio

        # Ajusta velocidade com base no yaw
        u[i] = u[i]*math.cos(math.radians(individuo[i]))
    print("Velocidades: " , u)
    print("Deltinha: " , deltinha)
    
    
    # Calcula a produção total de energia
    producao = sum(0.5 * 1.225 * math.pi * r0**2 * ui**3 for ui in u)  # Energia proporcional a v^3 sem descontar os ângulos
    producao_ajustada = producao * np.cos(np.radians(individuo[i]))**3      # Produção ajustada com o ângulo de inclinação de cada turbina
    return producao_ajustada/10000


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
        print(f"\nGeração {geracao + 1}:")
        print("População e Aptidão:")

        # Calcula a aptidão de cada indivíduo
        fitness_scores = [fitness_jensen(individuo, layout_x, layout_y, r0, u0, ct, k) for individuo in populacao]
        
        # Itera e imprime cada indivíduo com uma numeração e sua aptidão
        for indice, (individuo, score) in enumerate(zip(populacao, fitness_scores), start=1):
            angulos_formatados = ", ".join(f"{angulo:.3f}°" for angulo in individuo)
            print(f"Indivíduo {indice}: [{angulos_formatados}], Fitness = {score:.2f} kW")
        
        # Encontra o melhor indivíduo da geração
        melhor_fitness = max(fitness_scores)
        melhor_individuo = populacao[fitness_scores.index(melhor_fitness)]
        melhor_fitness_historico.append(melhor_fitness)
        
        print(f"Melhor aptidão = {melhor_fitness:.2f}, Melhor indivíduo = {melhor_individuo}")
        
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
    print("\nMelhor configuração de ângulos yaw encontrada:", melhor_individuo)
    print("Aptidão da melhor solução:", melhor_fitness)
    
    # Plotando gráfico da otimização
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
print('==================================================')
print()
print()

