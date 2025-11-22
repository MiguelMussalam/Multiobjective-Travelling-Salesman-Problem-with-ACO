import numpy as np
import typing as tp

def criar_matriz_custo(
    distancia_cidades: np.ndarray, 
    tempo_viagem: np.ndarray, 
    custo_viagem: np.ndarray, 
    PESO_CUSTO: float, 
    PESO_DISTANCIA: float, 
    PESO_TEMPO: float
) -> np.ndarray:

    dist_min = np.min(distancia_cidades[np.nonzero(distancia_cidades)])
    dist_max = np.max(distancia_cidades)
    distancia_norm = (distancia_cidades - dist_min) / (dist_max - dist_min)
    np.fill_diagonal(distancia_norm, 0)
    #print(distancia_norm)

    tempo_min = np.min(tempo_viagem[np.nonzero(tempo_viagem)])
    tempo_max = np.max(tempo_viagem)
    tempo_norm = (tempo_viagem - tempo_min) / (tempo_max - tempo_min)
    np.fill_diagonal(tempo_norm, 0)
    #print(tempo_norm)

    custo_min = np.min(custo_viagem[np.nonzero(custo_viagem)])
    custo_max = np.max(custo_viagem)
    custo_norm = (custo_viagem - custo_min) / (custo_max - custo_min)
    np.fill_diagonal(custo_norm, 0)
    #print(custo_norm)

    matriz_custo_final = (
        PESO_DISTANCIA * distancia_norm +
        PESO_TEMPO * tempo_norm +
        PESO_CUSTO * custo_norm
    )
    
    # --- 3. Retorna o resultado ---
    return matriz_custo_final

def prox_cidade(
    cidade_atual: int, 
    tour_parcial: np.ndarray, 
    feromonios: np.ndarray,
    visibilidade: np.ndarray,
    NUM_CIDADES : int,
    A : float,
    B : float
) -> int:
    prob_maxima = -1.0
    proxima_cidade_idx = -1

    soma = 0.0
    for c in range(NUM_CIDADES):
        if c not in tour_parcial:
            
            fator_feromonio = feromonios[cidade_atual, c] ** A
            fator_visibilidade = visibilidade[cidade_atual, c] ** B
            
            soma += (fator_visibilidade * fator_feromonio)
    
    if soma == 0:
        disponiveis = np.setdiff1d(np.arange(NUM_CIDADES), tour_parcial)
        return disponiveis[0] if len(disponiveis) > 0 else -1

    for c in range(NUM_CIDADES):
        if c not in tour_parcial:
            
            fator_feromonio = feromonios[cidade_atual, c] ** A
            fator_visibilidade = visibilidade[cidade_atual, c] ** B
            
            prob_caminho = ((fator_visibilidade * fator_feromonio) / soma)
            
            if prob_caminho > prob_maxima:
                prob_maxima = prob_caminho
                proxima_cidade_idx = c
                
    return proxima_cidade_idx

def calcular_custos_tours(tours: np.ndarray, custo_combinado_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    num_formigas = tours.shape[0]
    num_passos_tour = tours.shape[1] - 1 # O tour tem 11 colunas, então são 10 passos
    custos = np.zeros(num_formigas)

    for f in range(num_formigas):
        for c in range(num_passos_tour):
            origem = int(tours[f, c])
            destino = int(tours[f, c+1])
            custos[f] += custo_combinado_matrix[origem, destino]
    
    melhor_agente_idx = np.argmin(custos)
    
    return custos, melhor_agente_idx

def atualizar_feromonio(
    feromonios: np.ndarray, 
    tours: np.ndarray, 
    custos: np.ndarray, 
    melhor_agente_idx: int, 
    Q: float, 
    R: float
) -> np.ndarray:

    feromonios_atualizado = feromonios * (1 - R)
    
    melhor_tour = tours[melhor_agente_idx]
    melhor_custo = custos[melhor_agente_idx]
    
    qtd_deposito = Q / melhor_custo
    
    for c in range(len(melhor_tour) - 1):
        origem = int(melhor_tour[c])
        destino = int(melhor_tour[c+1])
        
        feromonios_atualizado[origem, destino] += qtd_deposito
        feromonios_atualizado[destino, origem] += qtd_deposito
    
    np.fill_diagonal(feromonios_atualizado, 0)
        
    return feromonios_atualizado