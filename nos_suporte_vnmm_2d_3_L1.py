import itertools
import numpy as np

def nos_suporte_vnmm_2d_3_L1(
    P, 
    nodes_coords, 
    nodes_vectors, 
    arvore_busca, 
    K=15, 
    Tol_det=1e-3, 
    adaptativo=True, 
    passo_K=5, 
    K_max=None
):
    """
    Algoritmo incremental heurístico para seleção de 3 nós de suporte com suporte adaptativo de K.
    
    Parâmetros:
    - P: Coordenada do ponto de avaliação (x, y).
    - nodes_coords: Array numpy com as coordenadas de todos os nós globais.
    - nodes_vectors: Array numpy com as componentes vetoriais (tx, ty).
    - arvore_busca: Objeto KDTree já pré-construído com as coordenadas dos nós.
    - K: Número inicial de vizinhos mais próximos a serem recuperados pela kd-tree.
    - Tol_det: Tolerância mínima aceitável para o determinante de A.
    - adaptativo: Booleano indicando se o tamanho da vizinhança K deve ser expandido
                  automaticamente caso nenhum trio satisfaça Tol_det na vizinhança inicial.
    - passo_K: Incremento no número de vizinhos K em cada iteração de expansão.
    - K_max: Limite máximo de vizinhos a expandir (padrão: total de nós da malha).
    
    Retorna:
    - trio_indices: Índices originais (no array global) dos 3 nós selecionados.
    - det_A: O determinante absoluto da matriz A selecionada.
    - A_mat: A matriz de momento direcional 3x3 que foi aprovada.
    """
    P = np.asarray(P, dtype=float)
    N_total = len(nodes_coords)
    limite_K = N_total if K_max is None else min(K_max, N_total)
    K_atual = min(K, N_total)
    
    melhor_trio = None
    melhor_det = 0.0
    melhor_A = None
    melhor_k_efetivo = 0
    
    while True:
        # Fase 1: Recuperação de vizinhos mais próximos utilizando a árvore pré-construída
        distancias, indices_vizinhos = arvore_busca.query(P, k=K_atual)
        
        # Tratamento caso retorne apenas 1 vizinho (para evitar erro de iteração)
        if K_atual == 1:
            indices_vizinhos = [indices_vizinhos]
            
        # Fase 2: Busca Incremental com Fallback da Âncora
        for idx_ancora_local in range(K_atual - 2):
            ancora_global = indices_vizinhos[idx_ancora_local]
            indices_restantes = range(idx_ancora_local + 1, K_atual)
            
            # Gerando combinações 2 a 2 dos nós restantes
            for comb in itertools.combinations(indices_restantes, 2):
                idx2_global = indices_vizinhos[comb[0]]
                idx3_global = indices_vizinhos[comb[1]]
                k_efetivo_candidato = comb[1] + 1
                
                trio_candidato = [ancora_global, idx2_global, idx3_global]
                
                # Fase 3: Construção da Matriz de Interpolação A via operações por colunas
                coords_locais = nodes_coords[trio_candidato] - P
                vecs = nodes_vectors[trio_candidato]
                
                x_local = coords_locais[:, 0]
                y_local = coords_locais[:, 1]
                tx = vecs[:, 0]
                ty = vecs[:, 1]
                
                A = np.empty((3, 3), dtype=float)
                A[:, 0] = tx
                A[:, 1] = ty
                A[:, 2] = y_local * tx - x_local * ty
                
                # Fase 4: Avaliação Algébrica
                det_A = np.abs(np.linalg.det(A))
                
                if det_A > melhor_det:
                    melhor_det = det_A
                    melhor_trio = trio_candidato
                    melhor_A = A
                    melhor_k_efetivo = k_efetivo_candidato
                
                # Critério de Parada Antecipada
                if det_A >= Tol_det:
                    return trio_candidato, det_A, A, k_efetivo_candidato
                    
        # Se não for adaptativo ou já atingiu o limite de vizinhos
        if not adaptativo or K_atual >= limite_K:
            break
            
        # Expande K para a próxima iteração
        K_atual = min(K_atual + passo_K, limite_K)
        
    # Caso nenhuma combinação atinja Tol_det, retorna a melhor encontrada se houver, ou None
    if melhor_trio is not None and melhor_det >= Tol_det:
        return melhor_trio, melhor_det, melhor_A, melhor_k_efetivo
    elif melhor_trio is not None and adaptativo:
        # Retorna o melhor trio disponível quando adaptativo está ativo
        return melhor_trio, melhor_det, melhor_A, melhor_k_efetivo
    else:
        return None, 0.0, None, 0
