import pandas as pd
import numpy as np
import typing as tp

def carregar_dados_viagem() -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega os dados de coordenadas, distância, tempo e custo de uma planilha Excel.
    """
    try:
        info_cidades_path = "Informação de cidades.xlsx"
        
        coordenadas_df = pd.read_excel(info_cidades_path, skiprows=1, sheet_name=0, header=0)
        distancia_df = pd.read_excel(info_cidades_path, skiprows=1, sheet_name=1, header=0)
        tempo_df = pd.read_excel(info_cidades_path, skiprows=1, sheet_name=2, header=0)
        custo_df = pd.read_excel(info_cidades_path, skiprows=1, sheet_name=3, header=0)

        # Remove a coluna de nomes para ter apenas matrizes numéricas
        nome_coluna_cidades = distancia_df.columns[0]
        
        coordenadas_matrix = coordenadas_df.drop(columns=[nome_coluna_cidades]).to_numpy()
        distancia_matrix = distancia_df.drop(columns=[nome_coluna_cidades]).to_numpy()
        tempo_matrix = tempo_df.drop(columns=[nome_coluna_cidades]).to_numpy()
        custo_matrix = custo_df.drop(columns=[nome_coluna_cidades]).to_numpy()
        
        return coordenadas_matrix, distancia_matrix, tempo_matrix, custo_matrix

    except FileNotFoundError:
        print(f"Erro: Arquivo '{info_cidades_path}' não encontrado.")
        return np.array([]), np.array([]), np.array([]), np.array([])

if __name__ == '__main__':
    coordenadas, distancia, tempo, custo = carregar_dados_viagem()
    print(coordenadas)
    print(distancia)
    print(tempo)
    print(custo)