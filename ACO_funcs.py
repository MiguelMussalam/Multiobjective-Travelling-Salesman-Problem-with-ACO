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
    """
    Cria uma matriz de custo combinado normalizando e ponderando as matrizes de
    distância, tempo e custo.

    Args:
        distancia_cidades: Matriz com as distâncias (km).
        tempo_viagem: Matriz com os tempos (horas).
        custo_viagem: Matriz com os custos (R$).
        PESO_CUSTO: A importância do custo (0.0 a 1.0).
        PESO_DISTANCIA: A importância da distância (0.0 a 1.0).
        PESO_TEMPO: A importância do tempo (0.0 a 1.0).

    Returns:
        Uma única matriz (np.ndarray) de custo combinado para ser usada pelo ACO.
    """
    # Normaliza a Distância
    dist_min = np.min(distancia_cidades[np.nonzero(distancia_cidades)])
    dist_max = np.max(distancia_cidades)
    distancia_norm = (distancia_cidades - dist_min) / (dist_max - dist_min)
    np.fill_diagonal(distancia_norm, 0) # Garante que a diagonal permaneça zero
    #print(distancia_norm)

    # Normaliza o Tempo
    tempo_min = np.min(tempo_viagem[np.nonzero(tempo_viagem)])
    tempo_max = np.max(tempo_viagem)
    tempo_norm = (tempo_viagem - tempo_min) / (tempo_max - tempo_min)
    np.fill_diagonal(tempo_norm, 0)
    #print(tempo_norm)

    # Normaliza o Custo
    custo_min = np.min(custo_viagem[np.nonzero(custo_viagem)])
    custo_max = np.max(custo_viagem)
    custo_norm = (custo_viagem - custo_min) / (custo_max - custo_min)
    np.fill_diagonal(custo_norm, 0)
    #print(custo_norm)

    # --- 2. Soma Ponderada para Criar a Matriz Final ---
    # Combina as três matrizes normalizadas usando os pesos definidos.
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
    feromonios: np.array, 
    visibilidade: np.ndarray
) -> int:
    """
    Decide a próxima cidade a ser visitada usando uma regra GULOSA (determinística),
    mas adaptada para o contexto multiobjetivo e sem variáveis globais.

    Args:
        cidade_atual: O índice da cidade onde a formiga está.
        tour_parcial: Um array com os índices das cidades já visitadas.
        feromonios: A matriz de feromônios atual.
        visibilidade: A matriz de atratividade (1 / custo_combinado).

    Returns:
        O índice da próxima cidade a ser visitada.
    """
    num_cidades = feromonios.shape[0]
    prob_maxima = -1.0
    proxima_cidade_idx = -1

    soma_desejabilidades = 0.0
    for c in range(num_cidades):
        if c not in tour_parcial:
            desejabilidade = visibilidade[cidade_atual, c] * feromonios[cidade_atual, c]
            soma_desejabilidades += desejabilidade
    
    if soma_desejabilidades == 0:
        # Se não há caminho, escolhe a primeira cidade disponível que não esteja no tour
        disponiveis = np.setdiff1d(np.arange(num_cidades), tour_parcial)
        return disponiveis[0] if len(disponiveis) > 0 else -1

    for c in range(num_cidades):
        if c not in tour_parcial:
            # Calcula a probabilidade (desejabilidade / soma total)
            prob_caminho = (visibilidade[cidade_atual, c] * feromonios[cidade_atual, c]) / soma_desejabilidades
            
            # Se a probabilidade deste caminho for a maior encontrada até agora, guarda
            if prob_caminho > prob_maxima:
                prob_maxima = prob_caminho
                proxima_cidade_idx = c
                
    return proxima_cidade_idx

def calcular_custos_tours(tours: np.ndarray, custo_combinado_matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Calcula os custos dos tours de todas as formigas e encontra o índice do melhor agente.

    Args:
        tours (np.ndarray): Matriz com os caminhos percorridos por cada formiga.
        custo_combinado_matrix (np.ndarray): A matriz de custo multiobjetivo.

    Returns:
        tuple[np.ndarray, int]: Uma tupla contendo:
                                1. Um array com os custos de cada tour.
                                2. O índice do melhor agente (formiga com o menor custo).
    """
    num_formigas = tours.shape[0]
    num_passos_tour = tours.shape[1] - 1 # O tour tem 11 colunas, então são 10 passos
    custos = np.zeros(num_formigas)

    # 1. Calcula o custo de cada tour usando a matriz de custo combinado
    for f in range(num_formigas):
        for c in range(num_passos_tour):
            origem = int(tours[f, c])
            destino = int(tours[f, c+1])
            custos[f] += custo_combinado_matrix[origem, destino]
    
    # 2. Encontra o índice do agente com o menor custo de forma eficiente
    melhor_agente_idx = np.argmin(custos)
    
    # 3. Retorna os resultados em vez de usar globais
    return custos, melhor_agente_idx


def atualizar_feromonio(
    feromonios: np.ndarray, 
    tours: np.ndarray, 
    custos: np.ndarray, 
    melhor_agente_idx: int, 
    Q: float, 
    R: float
) -> np.ndarray:
    """
    Atualiza a matriz de feromônios usando a estratégia "Elitist Ant System".
    1. Evapora o feromônio em todas as arestas.
    2. Apenas a melhor formiga da iteração deposita feromônio.

    Args:
        feromonios: A matriz de feromônios atual.
        tours: Os caminhos percorridos por todas as formigas.
        custos: Um array com o custo do tour de cada formiga.
        melhor_agente_idx: O índice da formiga que teve o menor custo.
        Q: A constante de intensidade do feromônio.
        R: A taxa de evaporação do feromônio.

    Returns:
        A nova matriz de feromônios atualizada.
    """
    # 1. Evaporação
    feromonios_atualizado = feromonios * (1 - R)
    
    # 2. Deposição de feromônio pela melhor formiga
    melhor_tour = tours[melhor_agente_idx]
    melhor_custo = custos[melhor_agente_idx]
    
    # Calcula a quantidade de feromônio a ser depositada
    qtde_deposito = Q / melhor_custo
    
    # Adiciona o feromônio em cada aresta do melhor tour
    for c in range(len(melhor_tour) - 1):
        origem = int(melhor_tour[c])
        destino = int(melhor_tour[c+1])
        
        feromonios_atualizado[origem, destino] += qtde_deposito
        # Adiciona também no caminho de volta para problemas simétricos
        feromonios_atualizado[destino, origem] += qtde_deposito
        
    return feromonios_atualizado