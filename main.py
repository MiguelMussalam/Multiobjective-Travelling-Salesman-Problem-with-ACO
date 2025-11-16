import typing as tp
import numpy as np
import pandas as pd
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
A: tp.Final[float] = 1.0 # influência do feromônio
B: tp.Final[float] = 1.0 # influência da heurística
R: tp.Final[float] = 1.0 # Taxa de evaporação do feromônio
Q: tp.Final[float] = 1.0 # intensidade/quantidade de feromônio

# pesos
PESO_DISTANCIA: tp.Final[float] = 0.3 
PESO_TEMPO: tp.Final[float] = 0.5
PESO_CUSTO: tp.Final[float] = 0.2

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

nomes_indices = ['SANTOS','CAMPINAS','SOROCABA','RIBEIRÃO PRETO','ADAMANTINA',
                 'SÃO JOSÉ DOS CAMPOS', 'CAÇAPAVA', 'AVARÉ', 'AREIAS', 'HOLAMBRA']

if __name__ == '__main__':
    matriz_multiobjetivo = criar_matriz_custo(distancia_cidades,tempo_viagem,custo_viagem,PESO_CUSTO,PESO_DISTANCIA,PESO_TEMPO)

    epsilon = 1e-10  # Para evitar divisão por zero
    visibilidade = 1 / (matriz_multiobjetivo + epsilon)
    np.fill_diagonal(visibilidade, 0) # Nenhuma cidade é visível para si mesma

    matriz_multiobjetivo_df = pd.DataFrame(visibilidade,nomes_indices,nomes_indices)
    print(matriz_multiobjetivo_df)

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
                    tours[f][t] = prox_cidade(tours[f][t-1].astype(int), tours[f], feromonios,visibilidade, NUM_CIDADES, A, B)
                else:
                    tours[f][t] = inicio[f]
                    break
        
            print(tours[f])
    
        custos, melhor_agente = calcular_custos_tours(tours,matriz_multiobjetivo)
        print("CUSTOS:", custos)
        print("MELHOR AGENTE:", melhor_agente)

        # 1. Prepara a base do gráfico (pontos e nomes das cidades)
        plt.figure(figsize=(16, 10)) # Gráfico maior para melhor visualização
        
        str_cidades = [
            'Santos', 'Campinas', 'Sorocaba', 'R. Preto', 'Adamantina', 
            'SJC', 'Caçapava', 'Avaré', 'Areias', 'Holambra'
        ]
        x_coords, y_coords = coordenadas.T
        
        # Plota os pontos pretos das cidades
        plt.plot(x_coords, y_coords, color='black', marker='o', markersize=7, linestyle='')
        
        # Adiciona os nomes das cidades
        for idx in range(NUM_CIDADES):
            plt.annotate(
                f"{str_cidades[idx]}: {idx}", 
                (x_coords[idx], y_coords[idx]), 
                xytext=(x_coords[idx], y_coords[idx] + 0.1), # Posição do texto
                ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7)
            )

        # 2. (Opcional) Plota os tours das formigas "não-melhores" em azul claro
        for f in range(NUM_CIDADES):
            if f != melhor_agente:
                tour_ruim = tours[f].astype(int)
                graph = coordenadas[tour_ruim]
                x_tour, y_tour = graph.T
                plt.plot(x_tour, y_tour, linestyle='dashed', color='blue', alpha=0.2) # Alpha baixo

        # 3. Plota o MELHOR TOUR com destaque e gradiente de cor (FADE)
        melhor_tour_indices = tours[melhor_agente].astype(int)
        
        # Pega o mapa de cores que vai do branco ao vermelho
        cmap = plt.get_cmap('Reds')

        # Itera sobre cada SEGMENTO do melhor tour para plotá-lo individualmente
        for passo in range(NUM_CIDADES): # O tour tem 10 segmentos (passos)
            ponto_origem_idx = melhor_tour_indices[passo]
            ponto_destino_idx = melhor_tour_indices[passo+1]
            
            ponto_origem_coords = coordenadas[ponto_origem_idx]
            ponto_destino_coords = coordenadas[ponto_destino_idx]

            # Calcula a cor para este segmento. O valor de progresso vai de 0.0 a 1.0
            progresso = passo / (NUM_CIDADES - 1)
            cor_segmento = cmap(progresso)

            # Plota o segmento
            plt.plot(
                [ponto_origem_coords[0], ponto_destino_coords[0]], # Coordenadas X
                [ponto_origem_coords[1], ponto_destino_coords[1]], # Coordenadas Y
                color=cor_segmento,
                linewidth=3, # Linha mais grossa para destacar
                solid_capstyle='round'
            )

        plt.title(f"Melhor Rota da Época {i+1} (Agente {melhor_agente})", fontsize=16)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.show()

        feromonios = atualizar_feromonio(feromonios, tours, custos, melhor_agente, Q, R)
        feromonio_fd = pd.DataFrame(feromonios,nomes_indices,nomes_indices)
        print(feromonio_fd)