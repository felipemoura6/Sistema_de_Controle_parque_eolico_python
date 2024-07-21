import matplotlib.pyplot as plt
import numpy as np
import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface, WindRose
from windrose import WindroseAxes
from floris.tools.layout_functions import visualize_layout
import csv
from datetime import datetime
import time

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Registre o tempo de início
start_time = time.time()
ang_inferior=-31
ang_superior=30

fi = FlorisInterface("inputs/gch.yaml") # Criando a Interface Floris com valores iniciais importados do arquivo gch.yaml
farm_powers = None

# Read in the wind rose using the class
wind_rose = WindRose()


#Variavel para plotagens: True = Plotar ::: False = Não plotar
plot_vv=False
plot_visualization=False
plot_potencia=False
plot_windrose=False


def calculo_producao_total_casos(casos, vet_yaw, wind_speeds, wind_directions, fi):
    global farm_powers
    farm_powers = np.zeros(casos)  # Inicializa um array para armazenar as potências
    fi.reinitialize(wind_speeds=[wind_speeds], wind_directions=[wind_directions])
    
    for i in range(casos):
        # Configurando os ângulos de inclinação para todas as turbinas
        yaw_angles[0,0,0] = vet_yaw[i]
            
        fi.calculate_wake(yaw_angles=yaw_angles)
        farm_power = fi.get_farm_power()
        farm_powers[i] = np.squeeze(farm_power) / 1e3  # Convertendo para kW e salvando o valor na posição atual de farm_powers

    print(" ".join(f"{power:.2f}" for power in farm_powers))
    producao_total = np.sum(farm_powers)
    return producao_total

## ================================== Subrotina de calculo de produção total do parque =====================================

def calculo_producao_total(num_turbinas, vet_yaw, wind_speeds, wind_directions, fi):
    global farm_powers
    farm_powers = np.zeros(len(wind_directions))  # Inicializa um array para armazenar as potências
    
    for i, (ws, wd) in enumerate(zip(wind_speeds, wind_directions)):
        fi.reinitialize(wind_speeds=[ws], wind_directions=[wd])  # Passando a velocidade e direção do vento atual para o FlorisInterface
        
        # Configurando os ângulos de inclinação para todas as turbinas
        for j in range(num_turbinas):
            yaw_angles[0,0,0] = vet_yaw[i,j]
            
        fi.calculate_wake(yaw_angles=yaw_angles)
        farm_power = fi.get_farm_power()
        farm_powers[i] = np.squeeze(farm_power) / 1e3  # Convertendo para kW e salvando o valor na posição atual de farm_powers

    print('Vetor de potência: ' + farm_powers)
    producao_total = np.sum(farm_powers)
    return producao_total


## ============================================ =============================================================================


## =================================== Subrotina de calculo MPPT do ângulo individual ======================================

def calculo_opt_yaw(yaw_angles, vet_yaw, fi, j, i):
    melhor_angulo = 0
    melhor_potencia = 0
    
    for angulo in range(ang_inferior, ang_superior):
        yaw_angles[0, 0, j] = angulo  # Corrigindo a forma de acesso aos valores de yaw_angles
        fi.calculate_wake(yaw_angles=yaw_angles)
        farm_power = fi.get_farm_power()  # Obtém a potência para o ângulo atual
        
        # Verifica se a potência atual é maior que a melhor potência até agora
        if farm_power > melhor_potencia:
            melhor_potencia = farm_power
            melhor_angulo = angulo
            
        vet_yaw[i, j] = melhor_angulo  # Atualiza o vetor de yaw com o melhor ângulo encontrado
      
## =========================================================================================================================



## ======================================== LEITURA DOS DADOS DE ENTRADA ===================================================

# Nome do arquivo de texto
arquivo1 = 'Dados_2023.CSV'
arquivo2 = 'Dados_2023_Small.CSV'
arquivo3 = 'Teste_Dados_Reduced.CSV'

# Listas para armazenar os valores das colunas
data = []
hora = []
wind_speeds = []
wind_directions = []
wmax = []

# Abra o arquivo e leia as colunas desejadas
with open(arquivo3, 'r', encoding='utf-8') as arquivo:
    leitor_csv = csv.reader(arquivo, delimiter=';')
    for linha in leitor_csv:
        # Se houver pelo menos cinco colunas na linha
        if len(linha) >= 5:
            # Adicione os valores da quarta e da quinta colunas aos vetores
            data.append((linha[0]))  # Data
            hora.append((linha[1]))  # Hora
            wind_speeds.append(float(linha[18]))  # Vento
            wind_directions.append(float(linha[16])) # Direção em graus
            wmax.append((linha[17])) # Velocidade média do vento

#print(wind_directions)
# Converter para o formato de data e hora
x_values = np.arange(len(data)) # Cria o vetor do eixo x com os dados das datas
hora_sem_utc = [h.replace(' UTC', '') for h in hora]
hora_formatada = [h.zfill(4)[:2] + ':' + h.zfill(4)[2:] for h in hora_sem_utc]  # Coloca a hora no formato HH:MM
datas_horas = [datetime.strptime(data[i] + " " + h, "%Y/%m/%d %H:%M") for i, h in enumerate(hora_formatada)] # Adiciona a data com a hora

# Plotando a velocidade do vento x data
if(plot_vv==True):
    fig, ax = plt.subplots()
    ax.plot(datas_horas, wind_speeds, marker='o', linestyle='-', label='Velocidade do vento (m/s)')
    ax.plot(datas_horas, wind_directions, marker='o', linestyle='-', label='Direção do vento (°)')
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('Direção do vento (graus)')
    ax.set_ylabel('Potência (kW)')


# Plotando a rosa do vento
if(plot_windrose==True):
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_directions, wind_speeds, normed=True, opening=0.8, edgecolor='white')

    # Adicionar rótulos e título
    ax.set_legend()  # Chama set_legend sem argumentos
    plt.legend(title='Velocidade do vento (m/s)')
    plt.title('Rosa dos Ventos de Probabilidade')
## ================================================================================================================


## ============================================ PARÂMETROS DE SIMULAÇÃO ===========================================

fi.reinitialize(layout_x=[0.], layout_y=[0.])  # Layout do parque
#fi.reinitialize(layout_x=[0, 500, 1000], layout_y=[50., 50., 50.])  # Layout do parque
turbine_name = []  # Lista para armazenar os nomes das turbinas

for i in range(len(fi.layout_x)):       # Laço para nomear as turbinas no formato 'T01' usando o length do layout
    turbine_name.append('T{:02d}'.format(i+1))
    
D = 126.0 # Diâmetro do rotor NREL 5 MW
fi.reinitialize(wind_directions=wind_directions)
num_wd = len(wind_directions)  # Quantidade de posições do vetor: Wind Directions
num_ws = len(wind_speeds)  # Quantidade de posições do vetor: Wind 
num_turbine = len(fi.layout_x)  # Quantidade de turbinas no parque eólico
yaw_angles = np.zeros((1, 1, num_turbine))
num_yaw = len(yaw_angles)
vet_yaw_opt = np.zeros((num_ws,num_turbine))


## ========================================= LAYOUT DO PARQUE ====================================================

# Visualização do parque
if(plot_visualization==True):
    ax2=visualize_layout(
        fi,
        show_wake_lines=False,
        lim_lines_per_turbine=2,
        plot_rotor=True,
        black_and_white=True,
        turbine_names=turbine_name
    )

## ===============================================================================================================



## ===================================== SIMULANDO A POTÊNCIA DO PARQUE ==========================================
print('')
print('====================================')
print(f"Vento: {wind_speeds[0]}m/s - {wind_directions[0]}°")
print('')

start = -42
end = 42
increment = 3

# Crie o vetor de ângulos usando np.arange
vet_yaw_nom = np.arange(start, end + increment, increment)
casos=len(vet_yaw_nom)


## PRODUÇÃO NOMINAL DO PARQUE
farm_powers_nom = np.zeros(casos)  # Inicializa um array para armazenar as potências
    
print(vet_yaw_nom)
producao_total_nom = calculo_producao_total_casos(casos, vet_yaw_nom, wind_speeds, wind_directions, fi)
farm_powers_nom = farm_powers

      


    
    

## PRODUÇÃO OTIMIZADA INDIVIDUALMENTE DAS TURBINAS NO PARQUE
producao_total_opt=0 # Inicializando a potência otimizada somada do parque com valor zerado
farm_powers_opt = np.zeros(len(wind_directions))  # Inicializa um array para armazenar as potências otimizadas
vet_yaw_opt = np.zeros((num_ws,num_turbine))


## ==============================================================================================================


# ===============================================================================================================
# ===============================================================================================================

for i in range(0,casos,3):
    
    horizontal_plane2 = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        yaw_angles=np.array([[[float(vet_yaw_nom[i])]]]),
    )
    
    
    # Create the plots
    plt.figure(figsize=(10, 6))

    title = f"Ângulo: {vet_yaw_nom[i]}° - Produção: {farm_powers_nom[i]:.2f}kW"
    wakeviz.visualize_cut_plane(
        horizontal_plane2,
        ax=plt.gca(),
        label_contours=True,
        title=title
    )



# Crie o gráfico
plt.figure(figsize=(10, 6))
plt.plot(vet_yaw_nom, farm_powers_nom, marker='o', linestyle='-', color='b')
plt.title('Potência X Ângulo de Yaw')
plt.xlabel('Ângulo de Yaw (graus)')
plt.ylabel('Potência (kW)')
plt.grid(True)
plt.show()


# Registre o tempo de término
end_time = time.time()

# Calcule o tempo decorrido
elapsed_time = end_time - start_time

print(f"Tempo de simulação decorrido: {elapsed_time} segundos")

print('')
print('====================================')
print('')

wakeviz.show_plots()
plt.show()
