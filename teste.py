import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import itertools
import io

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

def selecionar_nos_vnmm_2d(P, nodes_coords, nodes_vectors, K=15, Tol_det=1e-3):
    """
    Algoritmo incremental heurístico para seleção de 3 nós de suporte.
    
    Parâmetros:
    - P: Coordenada do ponto de avaliação (x, y).
    - nodes_coords: Array numpy com as coordenadas de todos os nós globais.
    - nodes_vectors: Array numpy com as componentes vetoriais (tx, ty).
    - K: Número de vizinhos mais próximos a serem recuperados pela kd-tree.
    - Tol_det: Tolerância mínima aceitável para o determinante de A.
    
    Retorna:
    - trio_indices: Índices originais (no array global) dos 3 nós selecionados.
    - det_A: O determinante absoluto da matriz A selecionada.
    - A_mat: A matriz de momento direcional 3x3 que foi aprovada.
    """
    P = np.array(P)
    N_total = len(nodes_coords)
    K_busca = min(K, N_total)
    
    # Fase 0 e 1: Estrutura de busca espacial e recuperação de vizinhos
    # A kd-tree possui complexidade de busca O(log N)
    arvore = KDTree(nodes_coords)
    distancias, indices_vizinhos = arvore.query(P, k=K_busca)
    
    # Tratamento caso retorne apenas 1 vizinho (para evitar erro de iteração)
    if K_busca == 1:
        indices_vizinhos = [indices_vizinhos]

    # Fase 2: Busca Incremental com Falback da Âncora
    # Começamos tentando o nó mais próximo (índice 0 da busca) como Âncora
    for idx_ancora_local in range(K_busca - 2):
        ancora_global = indices_vizinhos[idx_ancora_local]
        
        # O subconjunto de nós que serão combinados com a âncora
        indices_restantes = range(idx_ancora_local + 1, K_busca)
        
        # Gerando combinações 2 a 2 dos nós restantes
        for comb in itertools.combinations(indices_restantes, 2):
            idx2_global = indices_vizinhos[comb[0]]
            idx3_global = indices_vizinhos[comb[1]]
            
            trio_candidato = [ancora_global, idx2_global, idx3_global]
            
            # Fase 3: Construção da Matriz de Interpolação A
            A = np.zeros((3, 3))
            for i, idx_global in enumerate(trio_candidato):
                # Translação para o sistema de coordenadas local
                x_local = nodes_coords[idx_global][0] - P[0]
                y_local = nodes_coords[idx_global][1] - P[1]
                
                tx = nodes_vectors[idx_global][0]
                ty = nodes_vectors[idx_global][1]
                
                A[i, 0] = tx
                A[i, 1] = ty
                A[i, 2] = y_local * tx - x_local * ty
                
            # Fase 4: Avaliação Algébrica
            det_A = np.abs(np.linalg.det(A))
            
            # Critério de Parada Antecipada
            if det_A >= Tol_det:
                return trio_candidato, det_A, A
                
    # Caso nenhuma combinação atenda à tolerância dentro da vizinhança K
    return None, 0.0, None

# ==========================================
# EXEMPLO DE USO
# ==========================================
if __name__ == "__main__":
    # Simulando a leitura de um arquivo CSV (dados fictícios)
    csv_simulado = """id,x,y,tx,ty
0,0.1,0.2,1.0,0.0
1,0.5,0.1,0.707,-0.707
2,0.2,0.6,0.0,1.0
3,0.8,0.8,-0.707,0.707
4,0.0,0.0,-1.0,0.0
5,1.2,0.5,0.0,-1.0
6,-0.5,0.4,0.866,0.5"""

    arquivo_memoria = io.StringIO(csv_simulado)
    
    # 1. Leitura dos dados
    coords, vectors = carregar_malha(arquivo_memoria)
    
    # 2. Definição do ponto de avaliação P
    Ponto_Avaliacao = [0.3, 0.3]
    Tolerancia = 1e-3
    Tamanho_Vizinhanca = 5
    
    # 3. Execução do algoritmo
    nos_selecionados, determinante, matriz_A = selecionar_nos_vnmm_2d(
        P=Ponto_Avaliacao, 
        nodes_coords=coords, 
        nodes_vectors=vectors, 
        K=Tamanho_Vizinhanca, 
        Tol_det=Tolerancia
    )
    
    # 4. Resultados
    print(f"Ponto de Avaliação: {Ponto_Avaliacao}")
    if nos_selecionados:
        print(f"Nós de suporte selecionados (Índices Globais): {nos_selecionados}")
        print(f"Determinante |det(A)|: {determinante:.6f}")
        print("Matriz A aprovada:")
        print(np.round(matriz_A, 4))
        
        # Apenas para conferência espacial
        print("\nCoordenadas físicas dos nós escolhidos:")
        for idx in nos_selecionados:
            print(f"Nó {idx}: {coords[idx]}")
    else:
        print(f"Falha: Nenhum trio estável encontrado para a tolerância de {Tolerancia} dentro dos {Tamanho_Vizinhanca} vizinhos.")
