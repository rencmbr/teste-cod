import numpy as np


def calcular_funcoes_forma_vnmm_2d(P, nodes_coords, nodes_vectors, nos_selecionados, matriz_a=None):
    """
    Calcula as funções de forma vetoriais N_i e o seu rotacional (curl N_i) 
    no ponto de avaliação P para a formulação VNMM 2D com 3 nós de suporte.
    
    Fundamentação matemática (conforme Readme.md):
    - Base polinomial de ordem 1: L^1 = <[1, 0]^T, [0, 1]^T, [y, -x]^T>
    - Função de forma N_i(x, y) = beta_1i * [1, 0]^T + beta_2i * [0, 1]^T + beta_3i * [y, -x]^T
    - Condição de projeção: N_i(x_k, y_k) . t_k = delta_ik  =>  A * beta_i = L_i
    - Matriz de coeficientes: beta = [beta_1, beta_2, beta_3] = inv(A)
    - No ponto de avaliação P (origem do sistema local x=0, y=0):
        N_i(P) = [beta_1i, beta_2i]^T
        Phi(P) = [N_1(P), N_2(P), N_3(P)] (matriz 2x3)
    - Rotacional de N_i:
        curl(N_i) = [0, 0, -2 * beta_3i]^T  (componente z: -2 * beta_3i)
        curl_Phi = [-2 * beta_31, -2 * beta_32, -2 * beta_33] (vetor 1x3 ou 3)
    
    Parâmetros:
    - P: Coordenadas do ponto de avaliação (x, y).
    - nodes_coords: Array numpy com as coordenadas dos nós globais (N, 2).
    - nodes_vectors: Array numpy com as componentes vetoriais (tx, ty) dos nós globais (N, 2).
    - nos_selecionados: Lista/array com os 3 índices globais dos nós de suporte.
    - matriz_a: (Opcional) Matriz A (3x3) já calculada previamente no passo de seleção.
    
    Retorna:
    - Phi: Matriz (2, 3) das funções de forma no ponto de avaliação P.
           A coluna i corresponde ao vetor N_i(P) = [N_ix(P), N_iy(P)]^T.
    - rot_Phi: Array (3,) contendo a componente z do rotacional de cada função de forma:
               [-2*beta_31, -2*beta_32, -2*beta_33].
    - beta: Matriz (3, 3) dos coeficientes locais da interpolação (inv(A)).
            A coluna i contém [beta_1i, beta_2i, beta_3i]^T.
    """
    P = np.asarray(P, dtype=float)
    
    # Se a matriz A não for fornecida, calcula-a a partir dos nós selecionados
    if matriz_a is None:
        coords_locais = nodes_coords[nos_selecionados] - P
        vecs = nodes_vectors[nos_selecionados]
        
        x_local = coords_locais[:, 0]
        y_local = coords_locais[:, 1]
        tx = vecs[:, 0]
        ty = vecs[:, 1]
        
        A = np.empty((3, 3), dtype=float)
        A[:, 0] = tx
        A[:, 1] = ty
        A[:, 2] = y_local * tx - x_local * ty
    else:
        A = np.asarray(matriz_a, dtype=float)
        
    # Resolução do sistema A * beta_i = L_i para i=1,2,3
    # Como [L_1, L_2, L_3] = I_3, beta = inv(A)
    beta = np.linalg.inv(A)
    
    # Funções de forma no ponto de avaliação P (onde x_local=0, y_local=0)
    # N_ix(P) = beta_1i, N_iy(P) = beta_2i
    Phi = beta[0:2, :]  # shape (2, 3)
    
    # Rotacional das funções de forma: curl(N_i) = -2 * beta_3i na direção z
    rot_Phi = -2.0 * beta[2, :]  # shape (3,)
    
    return Phi, rot_Phi, beta
