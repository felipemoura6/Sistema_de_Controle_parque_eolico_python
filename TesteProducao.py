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


## ================================== Subrotina de calculo de produção total do parque =====================================

def calculo_producao_total(num_turbinas, vet_yaw, wind_speeds, wind_directions, fi):
    global farm_powers
    farm_powers = np.zeros(len(wind_directions))  # Inicializa um array para armazenar as potências
    
    for i, (ws, wd) in enumerate(zip(wind_speeds, wind_directions)):
        fi.reinitialize(wind_speeds=[ws], wind_directions=[wd])  # Passando a velocidade e direção do vento atual para o FlorisInterface
        
        # Configurando os ângulos de inclinação para todas as turbinas
        for j in range(num_turbinas):
            yaw_angles[0,0,j] = vet_yaw[i,j]
            
        fi.calculate_wake(yaw_angles=yaw_angles)
        farm_power = fi.get_farm_power()
        farm_powers[i] = np.squeeze(farm_power) / 1e3  # Convertendo para kW e salvando o valor na posição atual de farm_powers

    producao_total = np.sum(farm_powers)
    return producao_total


## ============================================ =============================================================================


## ======================================== IMPRIME RESULTADOS ===================================================

def imprime_resultados(num_turbinas, vet_yaw, wind_speeds, wind_directions, fi, potencia_parque):
    print("")
    # Cabeçalho
    cabeçalho = "| {:<19} |".format("DATA - HORA") + " {:<15} |".format("VENTO")
    for i in range(1, len(turbine_name) + 1):   # Printa no cabeçalho as colunas de todas as turbinas
        cabeçalho += " {:<6} |".format(f"T{i}")
    print(cabeçalho)

    # Resultados
    for i in range(len(wind_speeds)):
        row = "| {:<19} |".format(f"{data[i]} - {hora_formatada[i]}") + " {:<15} |".format(f"{wind_speeds[i]}m/s / {wind_directions[i]}°")

        for yaw in vet_yaw[i]:
            row += " {:<6} |".format(f"{yaw}°")
        print(row)
    print("")

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

## ================================================================================================================


## ============================================ PARÂMETROS DE SIMULAÇÃO ===========================================

fi.reinitialize(layout_x=[0, 0, 0, 500, 500, 500, 1000, 1000, 1000], layout_y=[50., 220., 440., 50., 220., 440., 50., 220., 440.])  # Layout do parque
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

## ===================================== SIMULANDO A POTÊNCIA DO PARQUE ==========================================


vet_yaw_nom = np.zeros((num_ws,num_turbine))


## ==============================================================================================================
vet_yaw_nom[0] = [
    0, 0, 0,
    0, 0, 0,
    0, 0, 0
]
producao_total_nom = calculo_producao_total(num_turbine, vet_yaw_nom, wind_speeds, wind_directions, fi)
print('')
print('==================================================================')
print(f"Vento: {wind_speeds[0]}m/s - {wind_directions[0]}°")
print('')
print(f"Vetor Yaw Zerado: {vet_yaw_nom}")
print("Produção total nominal: ", "{:.2f}".format(producao_total_nom),"kW")
print('')

vet_yaw_nom[0] = [
    -6.9264992, 0.50648989, 0.22401308,
    28.91013223, 20.73122991, 22.49044182,
    28.14692902, 27.00129826, 25.57046741
]
print(f"Vetor Yaw Alterado: {vet_yaw_nom}")
producao_total_opt = calculo_producao_total(num_turbine, vet_yaw_nom, wind_speeds, wind_directions, fi)
print("Produção total parque: ", producao_total_opt,"kW")

saldo=((producao_total_opt-producao_total_nom)/producao_total_nom)*100
print("Saldo: ", saldo,"%")
print('==================================================================')
print('')
    
# Registre o tempo de término
end_time = time.time()

# Calcule o tempo decorrido
elapsed_time = end_time - start_time

print(f"Tempo de simulação decorrido: {elapsed_time} segundos")


wakeviz.show_plots()
plt.show()
