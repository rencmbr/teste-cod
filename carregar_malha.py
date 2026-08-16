import pandas as pd

def carregar_malha(filepath):
    """
    Lê o arquivo de dados dos nós.
    Espera-se um arquivo CSV (ou TXT separado por vírgulas) com as colunas:
    id, x, y, tx, ty
    onde (tx, ty) são as componentes do vetor unitário.
    """
    # Utilizando pandas para leitura rápida de dados tabulares
    df = pd.read_csv(filepath)
    
    # Extraindo as coordenadas espaciais e os vetores de direção
    nodes_coords = df[['x', 'y']].to_numpy()
    nodes_vectors = df[['tx', 'ty']].to_numpy()
    
    return nodes_coords, nodes_vectors
