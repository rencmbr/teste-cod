import itertools
import numpy as np


def nos_suporte_vnmm_2d_6_P1(
    P, 
    nodes_coords, 
    nodes_vectors, 
    arvore_busca, 
    K=12, 
    Tol_det=1e-4, 
    adaptativo=True, 
    passo_K=4, 
    K_max=None
):
    """
    Algoritmo incremental heurístico para seleção de 6 nós de suporte com suporte adaptativo de K,
    utilizando a base linear completa P1 (6 termos).
    
    Fundamentação Matemática:
    - Base polinomial linear completa:
      P1 = <[1,0]^T, [0,1]^T, [x,0]^T, [y,0]^T, [0,x]^T, [0,y]^T>
    - Matriz de colocação A (6x6) para cada sexteto em coordenadas locais (x_k - x_P, y_k - y_P):
      Linha k: [t_kx, t_ky, dx_k * t_kx, dy_k * t_kx, dx_k * t_ky, dy_k * t_ky]
    - Análise dimensional: 2 colunas O(1) e 4 colunas O(h) => det(A) ~ O(h^4)
    
    Parâmetros:
    - P: Coordenada do ponto de avaliação (x, y).
    - nodes_coords: Array numpy (N, 2) com as coordenadas de todos os nós globais.
    - nodes_vectors: Array numpy (N, 2) com as componentes vetoriais (tx, ty).
    - arvore_busca: Objeto KDTree já pré-construído com as coordenadas dos nós.
    - K: Número inicial de vizinhos mais próximos a serem recuperados pela KD-Tree (padrão: 12).
    - Tol_det: Tolerância mínima aceitável para o determinante |det(A)|.
    - adaptativo: Booleano indicando se a vizinhança K deve ser expandida automaticamente.
    - passo_K: Incremento no número de vizinhos K em cada iteração de expansão (padrão: 4).
    - K_max: Limite máximo de vizinhos a expandir (padrão: total de nós da malha).
    
    Retorna:
    - sexteto_indices: Lista com os 6 índices globais dos nós selecionados.
    - det_A: O determinante absoluto da matriz A selecionada.
    - A_mat: A matriz de colocação 6x6 aprovada.
    - k_efetivo: Número efetivo de vizinhos mais próximos utilizados na busca.
    """
    P = np.asarray(P, dtype=float)
    N_total = len(nodes_coords)
    limite_K = N_total if K_max is None else min(K_max, N_total)
    K_atual = min(max(K, 6), N_total)
    
    melhor_sexteto = None
    melhor_det = 0.0
    melhor_A = None
    melhor_k_efetivo = 0
    
    while True:
        # Fase 1: Recuperação dos K vizinhos mais próximos via KD-Tree
        distancias, indices_vizinhos = arvore_busca.query(P, k=K_atual)
        
        if K_atual == 1:
            indices_vizinhos = [indices_vizinhos]
        else:
            indices_vizinhos = list(indices_vizinhos)
            
        # Fase 2: Busca Combinatória com Fixação Progressiva de Âncora
        # Garante prioridade geométrica aos nós mais próximos do ponto P
        for idx_ancora_local in range(K_atual - 5):
            ancora_global = indices_vizinhos[idx_ancora_local]
            indices_restantes = range(idx_ancora_local + 1, K_atual)
            
            # Combinações 5 a 5 dentre os vizinhos subsequentes
            for comb in itertools.combinations(indices_restantes, 5):
                sexteto_candidato = [ancora_global] + [indices_vizinhos[c] for c in comb]
                k_efetivo_candidato = comb[-1] + 1
                
                # Fase 3: Construção Vetorizada da Matriz de Colocação A (6x6)
                coords_locais = nodes_coords[sexteto_candidato] - P
                vecs = nodes_vectors[sexteto_candidato]
                
                dx = coords_locais[:, 0]
                dy = coords_locais[:, 1]
                tx = vecs[:, 0]
                ty = vecs[:, 1]
                
                A = np.empty((6, 6), dtype=float)
                A[:, 0] = tx
                A[:, 1] = ty
                A[:, 2] = dx * tx
                A[:, 3] = dy * tx
                A[:, 4] = dx * ty
                A[:, 5] = dy * ty
                
                # Fase 4: Avaliação do Determinante
                det_A = float(np.abs(np.linalg.det(A)))
                
                if det_A > melhor_det:
                    melhor_det = det_A
                    melhor_sexteto = sexteto_candidato
                    melhor_A = A
                    melhor_k_efetivo = k_efetivo_candidato
                    
                # Critério de Parada Antecipada
                if det_A >= Tol_det:
                    return sexteto_candidato, det_A, A, k_efetivo_candidato
                    
        # Se não for adaptativo ou já atingiu o limite máximo de vizinhos
        if not adaptativo or K_atual >= limite_K:
            break
            
        # Expansão dinâmica da vizinhança
        K_atual = min(K_atual + passo_K, limite_K)
        
    # Retorno após esgotar as tentativas
    if melhor_sexteto is not None and (melhor_det >= Tol_det or adaptativo):
        return melhor_sexteto, melhor_det, melhor_A, melhor_k_efetivo
    else:
        return None, 0.0, None, 0
