import numpy as np
import random

# Parâmetros do algoritmo genético
NUM_TURBINAS = 3        # Número de turbinas no parque eólico
TAM_POPULACAO = 20      # Quantidade de indivíduos na população
NUM_GERACAO = 10     # Quantidade de gerações
TAXA_MUTACAO = 0.1       # Taxa de mutação
YAW_MIN = -30             # Ângulo mínimo do yaw
YAW_MAX = 30              # Ângulo máximo do yaw

# Função de aptidão
def fitness(individuo):
    """
    Função de aptidão que simula a produção de energia com base nos ângulos yaw.
    Penaliza grandes diferenças de ângulos que causam turbulência.
    """
    energia = 0
    for i in range(NUM_TURBINAS):
        # Mais próximo do ângulo ótimo (0), maior a energia.
        energia += 100 - abs(individuo[i])
    
    
    # Penalização por turbulência (diferenças bruscas de ângulos entre turbinas)
    penalidade = sum(abs(individuo[i] - individuo[i-1]) for i in range(1, NUM_TURBINAS))
    
    return energia - penalidade

# Inicialização da população
def populacao_inicial():
    """
    Cria a população inicial com ângulos yaw aleatórios para cada turbina.
    """
    populacao = []
    print("População Inicial: ")
    for _ in range(TAM_POPULACAO):
        # Cada indivíduo é um vetor de ângulos yaw (aleatórios entre YAW_MIN e YAW_MAX)
        individuo = [random.uniform(YAW_MIN, YAW_MAX) for _ in range(NUM_TURBINAS)]
        print("Individuo", _, ": ", individuo)
        populacao.append(individuo)
    
    return populacao

# Seleção por torneio
def selecao(populacao, fitness_scores):
    """
    Seleciona dois pais da população via seleção por torneio.
    """
    # Escolhemos dois indivíduos aleatoriamente e pegamos o melhor
    tamanho_selecao = 3
    selected = random.sample(range(TAM_POPULACAO), tamanho_selecao)
    melhor_individual = max(selected, key=lambda x: fitness_scores[x])
    return populacao[melhor_individual]

# Cruzamento (Crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(0, NUM_TURBINAS-1)
    filho = parent1[:crossover_point] + parent2[crossover_point:]
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
    # Inicializa a população
    populacao = populacao_inicial()
    
    # Itera por várias gerações
    for geracao in range(NUM_GERACAO):
        # Calcula a aptidão de cada indivíduo
        fitness_scores = [fitness(individuo) for individuo in populacao]
        
        # Encontra o melhor indivíduo da geração
        melhor_fitness = max(fitness_scores)
        melhor_individuo = populacao[fitness_scores.index(melhor_fitness)]

        print(f"Geração {geracao+1}: Melhor aptidão = ", "{:.2f}".format(melhor_fitness), melhor_individuo)
        
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
            child = crossover(pai1, pai2)
            
            # Aplica a mutação
            child = mutacao(child)
            
            # Adiciona o novo indivíduo à nova população
            nova_populacao.append(child)
        
        # Substitui a população antiga pela nova
        populacao = nova_populacao
    
    # Melhor solução final
    melhor_fitness = max([fitness(individuo) for individuo in populacao])
    melhor_individuo = populacao[[fitness(individuo) for individuo in populacao].index(melhor_fitness)]
    
    print("\nMelhor configuração de ângulos yaw encontrada:", melhor_individuo)
    print("Aptidão da melhor solução:", melhor_fitness)

# Executa o algoritmo genético
genetic_algorithm()
