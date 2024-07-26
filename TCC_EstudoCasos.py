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
start=0
end = 0
increment = 1

fi = FlorisInterface("inputs/gch.yaml") # Criando a Interface Floris com valores iniciais importados do arquivo gch.yaml
farm_powers = None

# Read in the wind rose using the class
wind_rose = WindRose()

# Variavel para plotagens: True = Plotar ::: False = Não plotar
plot_vv = False
plot_visualization = False
plot_potencia = True
plot_windrose = False
plot_wake = True

# Controle de simulação
caso_simulacao = 0
vet_yaw_nom = None


## ================================== Subrotina de configuração de parâmetros ==============================================

def parametros_caso(caso_simulacao):
    global vet_yaw_nom, vet_layout_x, num_turbine  # Defina as variáveis como globais para que possam ser modificadas
    
    
    match caso_simulacao:
        
        
        case 1: # Turbinas: 1 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variado // Layout = Fixo
            print(f'Caso: {caso_simulacao}: Turbinas: 1 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variado // Layout = Fixo')
            fi.reinitialize(layout_x=[0.], layout_y=[0.])  # Layout do parque
            print(len(fi.layout_x))
            for i in range(len(fi.layout_x)):       # Laço para nomear as turbinas no formato 'T01' usando o length do layout
                turbine_name.append('T{:02d}'.format(i+1))
                
            start_yaw = -30
            increment_yaw = 3
            vet_yaw_nom = 0*np.ones((situacoes,num_turbine))
            vet_layout_x = 3*D*np.ones((situacoes))
            
            for i in range(situacoes):
                vet_yaw_nom[i] = start_yaw + increment_yaw*i
            print(vet_yaw_nom)
            return
        
        
        case 2: # Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Fixo // Layout = Variando em X
            print(f'Caso: {caso_simulacao}: Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Fixo // Layout = Variando em X')
            
            start_layoutx = 3*D
            increment_layoutx = 50
            num_turbine=2
            
            fi.reinitialize(layout_x=[0.,100], layout_y=[0.,0.])  # Layout do parque
            
            for i in range(len(fi.layout_x)):       # Laço para nomear as turbinas no formato 'T01' usando o length do layout
                turbine_name.append('T{:02d}'.format(i+1))
            
            vet_yaw_nom = 0*np.ones((situacoes,num_turbine))
            vet_layout_x = 3*D*np.ones((situacoes))
            
            for i in range(situacoes):
                vet_layout_x[i] = start_layoutx+ i*increment_layoutx
            return
        
        
        case 3: # Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variando // Layout = Fixo
            print(f'Caso: {caso_simulacao}: Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variando // Layout = Fixo')
            
            start_layoutx = 3*D
            increment_layoutx = 20
            num_turbine=2
            
            fi.reinitialize(layout_x=[0.,100], layout_y=[0.,0.])  # Layout do parque
            
            for i in range(len(fi.layout_x)):       # Laço para nomear as turbinas no formato 'T01' usando o length do layout
                turbine_name.append('T{:02d}'.format(i+1))
            
            vet_yaw_nom = 0*np.ones((situacoes,num_turbine))
            vet_layout_x = 3*D*np.ones((situacoes))
            
            for i in range(situacoes):
                vet_yaw_nom[i,1]=3*i
            return
        
        case 4: # Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variando // Layout = Variando em X
            print(f'Caso: {caso_simulacao}: Turbinas: 2 // Velocidade do Vento = Fixa // Direção do Vento = Fixa // Ângulo Yaw = Variando // Layout = Variando')
            
            start_layoutx = 3*D
            increment_layoutx = 20
            num_turbine=2
            
            fi.reinitialize(layout_x=[0.,100], layout_y=[0.,0.])  # Layout do parque
            
            for i in range(len(fi.layout_x)):       # Laço para nomear as turbinas no formato 'T01' usando o length do layout
                turbine_name.append('T{:02d}'.format(i+1))
            
            vet_yaw_nom = 0*np.ones((situacoes,num_turbine))
            vet_layout_x = 3*D*np.ones((situacoes))
            
            for i in range(situacoes):
                vet_layout_x[i] = start_layoutx+ i*increment_layoutx
                vet_yaw_nom[i,1]=3*i
            return


## =========================================================================================================================




def calculo_producao_total(num_turbinas, vet_layout_x, vet_yaw, wind_speeds, wind_directions, fi):
    global farm_powers
    global farm_t

    num_casos = len(vet_yaw)
    if(num_turbinas == 2): farm_t = np.zeros((num_turbinas, num_casos))

    farm_powers = np.zeros(num_casos)  # Inicializa um array para armazenar as potências
    
    for i in range(num_casos):
        fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)  # Passando a velocidade e direção do vento atual para o FlorisInterface
        
        if(num_turbinas == 2): # Coloca layout em x para duas turbinas exceto no caso 1
            fi.reinitialize(layout_x=[0., vet_layout_x[i]], wind_directions=wind_directions)
        
        
        u_points = fi.floris.flow_field.u
        fi.calculate_wake()
        # print(fi.turbine_average_velocities)
        
        # Configurando os ângulos de inclinação para todas as turbinas
        yaw_angles = np.zeros((1, 1, num_turbinas))  # Ajustando a forma de yaw_angles
        

        
        for j in range(num_turbinas):
            yaw_angles[0, 0, j] = vet_yaw[i, j]
            fi.calculate_wake(yaw_angles=yaw_angles)
            
        farm_power = fi.get_farm_power()
        turbine_powers = fi.get_turbine_powers() / 1000  # Convertendo para kW

        if(num_turbinas == 2):
            farm_t[:, i] = turbine_powers  # Armazenando as potências das turbinas em farm_t
        
        farm_powers[i] = np.squeeze(farm_power) / 1e3  # Convertendo para kW e salvando o valor na posição atual de farm_powers
        
        if plot_wake:
            horizontal_plane2 = fi.calculate_horizontal_plane(
                x_resolution=200,
                y_resolution=100,
                height=90.0,
                yaw_angles=yaw_angles,
            )
            
            # Create the plots
            plt.figure(figsize=(10, 6))
            title = f"Distância entre T: {vet_layout_x[i]}m - Ângulo: {vet_yaw[i]}° - Produção: {farm_powers[i]:.2f}kW"
            wakeviz.visualize_cut_plane(
                horizontal_plane2,
                ax=plt.gca(),
                label_contours=True,
                title=title
            )

    # Imprimindo farm_powers
    formatted_farm_powers = ", ".join([f"{fp:.2f}" for fp in farm_powers])
    print(f"farm_powers: [{formatted_farm_powers}]")

    # Imprimindo farm_t
    if(num_turbinas == 2):
        for i, row in enumerate(farm_t):
            formatted_row = ", ".join([f"{value:.2f}" for value in row])
            print(f"Produção T{i+1}: [{formatted_row}]")
        
    producao_total = np.sum(farm_powers)
    return producao_total
## ==========================================================================================================================



## ======================================== LEITURA DOS DADOS DE ENTRADA ===================================================

# Nome do arquivo de texto
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


## ===================================== INCIALIZAÇÃO DOS PARÂMETROS DE SIMULAÇÃO =================================

fi.reinitialize(layout_x=[0.], layout_y=[0.])  # Layout do parque
global turbine_name 
turbine_name = []  # Lista para armazenar os nomes das turbinas

D = 126.0 # Diâmetro do rotor NREL 5 MW
fi.reinitialize(wind_directions=wind_directions)
num_wd = len(wind_directions)  # Quantidade de posições do vetor: Wind Directions
num_ws = len(wind_speeds)  # Quantidade de posições do vetor: Wind 
num_turbine = len(fi.layout_x)  # Quantidade de turbinas no parque eólico
yaw_angles = np.zeros((1, 1, num_turbine))
num_yaw = len(yaw_angles)




## ===================================== SIMULANDO A POTÊNCIA DO PARQUE ==========================================
print('')
print('====================================')
print(f"Vento: {wind_speeds[0]}m/s - {wind_directions[0]}°")
print('')

situacoes=21
caso_simulacao=3

parametros_caso(caso_simulacao)


## PRODUÇÃO NOMINAL DO PARQUE
farm_powers_nom = np.zeros(situacoes)  # Inicializa um array para armazenar as potências
producao_total_nom = calculo_producao_total(num_turbine, vet_layout_x, vet_yaw_nom, wind_speeds, wind_directions, fi)
farm_powers_nom = farm_powers

      

## ==============================================================================================================


# ===============================================================================================================



# Crie o gráfico
plt.figure(figsize=(10, 6))
if(caso_simulacao==1):
    plt.plot(vet_yaw_nom, farm_powers_nom, marker='o', linestyle='-', color='b', label='Farm Powers Nom')
    plt.title('Potência X Ângulo Yaw')
    plt.xlabel('Ângulo Yaw (°)')
    plt.ylabel('Potência (kW)')
    plt.grid(True)
    plt.legend()  # Adiciona a legenda
    plt.show()
    
if(caso_simulacao==2):
    plt.plot(vet_layout_x, farm_powers_nom, marker='o', linestyle='-', color='b', label='Farm Powers Nom')
    if(num_turbine == 2): plt.plot(vet_layout_x, farm_t[0, :], marker='o', linestyle='-', color='r', label='T1')
    if(num_turbine == 2): plt.plot(vet_layout_x, farm_t[1, :], marker='o', linestyle='-', color='g', label='T2')
    plt.title('Potência X Distância')
    plt.xlabel('Distância em X (m)')
    plt.ylabel('Potência (kW)')
    plt.grid(True)
    plt.legend()  # Adiciona a legenda
    plt.show()
    
if(caso_simulacao==3):
    print('3')
    plt.plot(vet_yaw_nom, farm_powers_nom, marker='o', linestyle='-', color='b', label='Farm Powers Nom')
    if(num_turbine == 2): plt.plot(vet_yaw_nom, farm_t[0, :], marker='o', linestyle='-', color='r', label='T1')
    if(num_turbine == 2): plt.plot(vet_yaw_nom, farm_t[1, :], marker='o', linestyle='-', color='g', label='T2')
    plt.title('Potência X Ângulo Yaw')
    plt.xlabel('Ângulo Yaw (°)')
    plt.ylabel('Potência (kW)')
    plt.grid(True)
    plt.legend()  # Adiciona a legenda
    plt.show()
    
if(caso_simulacao==4):
    print('4')
    plt.plot(vet_yaw_nom, farm_powers_nom, marker='o', linestyle='-', color='b', label='Farm Powers Nom')
    if(num_turbine == 2): plt.plot(vet_yaw_nom, farm_t[0, :], marker='o', linestyle='-', color='r', label='T1')
    if(num_turbine == 2): plt.plot(vet_yaw_nom, farm_t[1, :], marker='o', linestyle='-', color='g', label='T2')
    plt.title('Potência X Ângulo Yaw')
    plt.xlabel('Ângulo Yaw (°)')
    plt.ylabel('Potência (kW)')
    plt.grid(True)
    plt.legend()  # Adiciona a legenda
    plt.show()
    


## ========================================= LAYOUT DO PARQUE ====================================================


if(plot_visualization==True & caso_simulacao==2):
    for i in range(len(vet_layout_x)):

        # Inicializa a figura
        fig, ax = plt.subplots(figsize=(10, 6))

        # Chama a função visualize_layout passando o eixo ax
        visualize_layout(
            fi,
            show_wake_lines=False,
            lim_lines_per_turbine=2,
            plot_rotor=True,
            black_and_white=True,
            turbine_names=turbine_name,
            ax=ax
        )

        # Define o título
        ax.set_title(f"Distância entre Turbinas: {vet_layout_x[i]}m - Produção Total: {farm_powers_nom[i]:.2f}kW")

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