import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from ACO_funcs import *
from dados_cidades import carregar_dados_viagem

coordenadas, distancia_cidades, tempo_viagem, custo_viagem = carregar_dados_viagem()

if distancia_cidades.size == 0:
    exit()

NUM_CIDADES: tp.Final[int] = distancia_cidades.shape[0]
cidades = np.arange(NUM_CIDADES)
inicio = np.copy(cidades)
tours = np.empty((NUM_CIDADES, NUM_CIDADES+1))
EPOCAS: tp.Final[int] = 10
matriz_multiobjetivo: tp.final[np.array]
custos = np.zeros(NUM_CIDADES)
melhor_agente = -1
qtde_feromonio = np.zeros(NUM_CIDADES)

# constantes do ACO
A: tp.Final[float] = 0.5 # influência do feromônio
B: tp.Final[float] = 0.5 # influência da heurística
R: tp.Final[float] = 0.5 # Taxa de evaporação do feromônio
Q: tp.Final[float] = 1.0 # intensidade/quantidade de feromônio

# pesos
PESO_DISTANCIA: tp.Final[float] = 0.2 
PESO_TEMPO: tp.Final[float] = 0.5
PESO_CUSTO: tp.Final[float] = 0.3

# feromonios iniciais
feromonios = np.array([
              [9.99, 0.30, 0.25, 0.20, 0.30, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 9.99, 0.20, 0.20, 0.30, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.25, 0.20, 9.99, 0.10, 0.15, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.20, 0.20, 0.10, 9.99, 0.45, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 9.99, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.22, 9.99, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.15, 0.30, 9.99, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.30, 0.30, 0.25, 9.99, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.40, 0.30, 0.25, 0.20, 9.99, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.15, 0.30, 0.25, 0.20, 0.30, 9.99]
             ])


if __name__ == '__main__':
    matriz_multiobjetivo = criar_matriz_custo(distancia_cidades,tempo_viagem,custo_viagem,PESO_CUSTO,PESO_DISTANCIA,PESO_TEMPO)

    epsilon = 1e-10  # Para evitar divisão por zero
    visibilidade = 1 / (matriz_multiobjetivo + epsilon)
    np.fill_diagonal(visibilidade, 0) # Nenhuma cidade é visível para si mesma

    print(matriz_multiobjetivo)

    for i in range(EPOCAS):
        tours.fill(-1)
        # cidade inicial para cada formiga
        np.random.shuffle(inicio)

        for f in range(NUM_CIDADES):        
            print("Formiga", f, "iniciando tour na cidade", inicio[f])

            # fazendo o tour
            t = 0
            tours[f][t] = inicio[f]
            while(True):
                t = t+1
                if(t < NUM_CIDADES):
                    tours[f][t] = prox_cidade(tours[f][t-1].astype(int), tours[f], feromonios,visibilidade)
                else:
                    tours[f][t] = inicio[f]
                    break
        
            print(tours[f])
    
        custos, melhor_agente = calcular_custos_tours(tours,matriz_multiobjetivo)
        print("CUSTOS:", custos)
        print("MELHOR AGENTE:", melhor_agente)

        # plotar todos os tours
        str = ['Santos: 0','Campinas: 1','Sorocaba: 2','Ribeirão Preto: 3','Adamantina: 4','São José dos Campos: 5','Caçapava: 6','Avaré: 7','Areias: 8', 'Holambra: 9']
        x, y = coordenadas.T
        plt.plot(x, y, color='black', marker='o', markersize=5)
        for i in range(NUM_CIDADES):
            plt.annotate(str[i], (x[i], y[i]), xytext=(x[i]+0.03, y[i]+0.1), bbox=dict(boxstyle="round", alpha=0.1), color="black", size=10, fontweight="bold")

        graph = np.empty((NUM_CIDADES+1, 2))
        for f in range(NUM_CIDADES):
            for c in range(NUM_CIDADES+1):
                graph[c] = coordenadas[tours[f][c].astype(int)]

            x, y = graph.T
            plt.plot(x, y, linestyle='dashed', color='blue')

        # plotar melhor tour
        for c in range(NUM_CIDADES+1):
            graph[c] = coordenadas[tours[melhor_agente][c].astype(int)]

        x, y = graph.T
        #plt.ion()
        plt.plot(x, y, color='red')
        #plt.draw()
        #plt.pause(0.005)
        plt.show()

        atualizar_feromonio(feromonios, tours, custos, melhor_agente, Q, R)