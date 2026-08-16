import itertools
import numpy as np

def selecionar_nos_vnmm_2d(P, nodes_coords, nodes_vectors, arvore_busca, K=15, Tol_det=1e-3):
    """
    Algoritmo incremental heurístico para seleção de 3 nós de suporte.
    
    Parâmetros:
    - P: Coordenada do ponto de avaliação (x, y).
    - nodes_coords: Array numpy com as coordenadas de todos os nós globais.
    - nodes_vectors: Array numpy com as componentes vetoriais (tx, ty).
    - arvore_busca: Objeto KDTree já pré-construído com as coordenadas dos nós.
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
    
    # Fase 1: Recuperação de vizinhos mais próximos utilizando a árvore pré-construída
    distancias, indices_vizinhos = arvore_busca.query(P, k=K_busca)
    
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
            
            # Fase 3: Construção da Matriz de Interpolação A via operações por colunas
            coords_locais = nodes_coords[trio_candidato] - P
            vecs = nodes_vectors[trio_candidato]
            
            x_local = coords_locais[:, 0]
            y_local = coords_locais[:, 1]
            tx = vecs[:, 0]
            ty = vecs[:, 1]
            
            A = np.empty((3, 3))
            A[:, 0] = tx
            A[:, 1] = ty
            A[:, 2] = y_local * tx - x_local * ty
                
            # Fase 4: Avaliação Algébrica
            det_A = np.abs(np.linalg.det(A))
            
            # Critério de Parada Antecipada
            if det_A >= Tol_det:
                return trio_candidato, det_A, A
                
    # Caso nenhuma combinação atenda à tolerância dentro da vizinhança K
    return None, 0.0, None
