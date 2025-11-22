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
EPOCAS: tp.Final[int] = 50
matriz_multiobjetivo: tp.Final[np.array]
custos = np.zeros(NUM_CIDADES)
melhor_agente = -1
qtde_feromonio = np.zeros(NUM_CIDADES)

# constantes do ACO
A: tp.Final[float] = 1.0 # influência do feromônio
B: tp.Final[float] = 1.0 # influência da heurística
R: tp.Final[float] = 0.2 # Taxa de evaporação do feromônio
Q: tp.Final[float] = 1.0 # intensidade/quantidade de feromônio

# pesos
PESO_DISTANCIA: tp.Final[float] = 0.3 
PESO_TEMPO: tp.Final[float] = 0.5
PESO_CUSTO: tp.Final[float] = 0.2

# Parada quase tenha N épocas sem melhoria
PARADA: tp.Final[int] = 10
contador_parada: tp.Final[int] = 0
ultimo_melhor_custo: tp.Final[float] = 0.0

# feromonios iniciais
feromonios = np.array([
              [0.0, 0.30, 0.25, 0.20, 0.30, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.0, 0.20, 0.20, 0.30, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.25, 0.20, 0.0, 0.10, 0.15, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.20, 0.20, 0.10, 0.0, 0.45, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.0, 0.30, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.22, 0.0, 0.25, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.15, 0.30, 0.0, 0.20, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.30, 0.30, 0.25, 0.0, 0.30, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.40, 0.30, 0.25, 0.20, 0.0, 0.20],
              [0.30, 0.30, 0.15, 0.45, 0.15, 0.30, 0.25, 0.20, 0.30, 0.0]
             ])
nomes_indices = ['SANTOS','CAMPINAS','SOROCABA','RIBEIRÃO PRETO','ADAMANTINA',
                 'SÃO JOSÉ DOS CAMPOS', 'CAÇAPAVA', 'AVARÉ', 'AREIAS', 'HOLAMBRA']

if __name__ == '__main__':
    matriz_multiobjetivo = criar_matriz_custo(distancia_cidades,tempo_viagem,custo_viagem,PESO_CUSTO,PESO_DISTANCIA,PESO_TEMPO)

    epsilon = 1e-10  # evitar divisão por zero
    visibilidade = 1 / (matriz_multiobjetivo + epsilon)
    np.fill_diagonal(visibilidade, 0) # Nenhuma cidade é visível pra si mesma

    matriz_multiobjetivo_df = pd.DataFrame(visibilidade,nomes_indices,nomes_indices)
    print("MATRIZ MULTIOBJETIVO:")
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
        
        if(custos[melhor_agente] == ultimo_melhor_custo):
            contador_parada += 1
            if (contador_parada == PARADA):
                print(f"Sem melhoria a {PARADA} épocas, atingindo condição de parada")
                break
        else: contador_parada = 0
        ultimo_melhor_custo = custos[melhor_agente]
        print("CUSTOS:", custos)
        print("MELHOR AGENTE:", melhor_agente)
        feromonio_df = pd.DataFrame(feromonios, index=nomes_indices, columns=nomes_indices)
        print(feromonio_df.round(2)) # arredondar saidda
        feromonios = atualizar_feromonio(feromonios, tours, custos, melhor_agente, Q, R)

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    
    ax0 = fig.add_subplot(gs[0]) # Eixo para o mapa
    ax1 = fig.add_subplot(gs[1]) # Eixo para a matriz de feromônios

    y_coords = coordenadas[:, 0]
    x_coords = coordenadas[:, 1]
    
    ax0.plot(x_coords, y_coords, color='black', marker='o', markersize=7, linestyle='')
    for idx in range(NUM_CIDADES):
        ax0.annotate(f"{nomes_indices[idx]}: {idx}", (x_coords[idx], y_coords[idx]), 
                     xytext=(x_coords[idx], y_coords[idx] + 0.1), ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.7))

    # formigas "não-melhores"
    for f in range(NUM_CIDADES):
        if f != melhor_agente:
            tour_ruim = tours[f].astype(int)
            graph = coordenadas[tour_ruim]
            ax0.plot(graph[:, 1], graph[:, 0], linestyle='dashed', color='blue', alpha=0.2)

    melhor_tour_indices = tours[melhor_agente].astype(int)
    cmap_reds = plt.get_cmap('Reds_r')

    for passo in range(NUM_CIDADES):
        ponto_origem_coords = coordenadas[melhor_tour_indices[passo]]
        ponto_destino_coords = coordenadas[melhor_tour_indices[passo+1]]
        progresso = passo / (NUM_CIDADES - 1)
        cor_segmento = cmap_reds(progresso)
        ax0.plot([ponto_origem_coords[1], ponto_destino_coords[1]], 
                 [ponto_origem_coords[0], ponto_destino_coords[0]],
                 color=cor_segmento, linewidth=3, solid_capstyle='round')

    melhor_tour_indices = tours[melhor_agente].astype(int)
    melhor_tour_nomes = [nomes_indices[idx] for idx in melhor_tour_indices]
    ax0.set_title(
        f"Melhor Rota da Época {i+1} (Agente {melhor_agente})",
        fontsize=16,
        pad=50
    )

    ax0.text(
        0.5, 1.05,
        " → ".join(melhor_tour_nomes),
        transform=ax0.transAxes,
        ha="center",
        fontsize=10
    )

    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.grid(True)
    ax0.invert_yaxis()
    ax0.invert_xaxis()
        
    im = ax1.imshow(feromonios, cmap='hot', interpolation='nearest')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
    ax1.set_xticks(np.arange(len(nomes_indices)))
    ax1.set_yticks(np.arange(len(nomes_indices)))
    ax1.set_xticklabels(nomes_indices)
    ax1.set_yticklabels(nomes_indices)
        
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax1.set_title("Matriz de Feromônios", fontsize=16)
        
    fig.tight_layout()
    
    plt.show()