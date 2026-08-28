import numpy as np


def funcoes_forma_vnmm_2d_6_P1(P, nodes_coords, nodes_vectors, nos_selecionados, matriz_a=None):
    """
    Calcula as funções de forma vetoriais N_i e o seu rotacional (curl N_i)
    no ponto de avaliação P para a formulação VNMM 2D com a base completa P1 (6 termos).
    
    Fundamentação matemática (conforme Readme.md e plano_implementacao_base_P1.md):
    - Base polinomial linear completa:
        P1 = <[1, 0]^T, [0, 1]^T, [x, 0]^T, [y, 0]^T, [0, x]^T, [0, y]^T>
    - Função de forma em coordenadas locais (dx = x - x_P, dy = y - y_P):
        N_i(dx, dy) = beta_1i [1, 0]^T + beta_2i [0, 1]^T + beta_3i [dx, 0]^T 
                    + beta_4i [dy, 0]^T + beta_5i [0, dx]^T + beta_6i [0, dy]^T
    - Condição de colocação nos nós de suporte: N_i(x_k) . t_k = delta_ik => A * beta = I_6 => beta = inv(A)
    - No ponto de avaliação P (dx = 0, dy = 0):
        N_i(P) = [beta_1i, beta_2i]^T
        Phi(P) = [N_1(P), N_2(P), ..., N_6(P)] = beta[0:2, :] (matriz 2x6)
    - Rotacional das funções de forma:
        curl(N_i) = (beta_5i - beta_4i) z_hat
        rot_Phi = beta[4, :] - beta[3, :] (vetor de dimensão 6)
    
    Parâmetros:
    - P: Coordenadas do ponto de avaliação (x, y).
    - nodes_coords: Array numpy com as coordenadas dos nós globais (N, 2).
    - nodes_vectors: Array numpy com as componentes vetoriais (tx, ty) dos nós globais (N, 2).
    - nos_selecionados: Lista/array com os 6 índices globais dos nós de suporte.
    - matriz_a: (Opcional) Matriz A (6x6) já calculada previamente no passo de seleção.
    
    Retorna:
    - Phi: Matriz (2, 6) das funções de forma no ponto de avaliação P.
           A coluna i corresponde ao vetor N_i(P) = [N_ix(P), N_iy(P)]^T.
    - rot_Phi: Array (6,) contendo a componente z do rotacional de cada função de forma:
               beta[4, :] - beta[3, :].
    - beta: Matriz (6, 6) dos coeficientes locais da interpolação (inv(A)).
            A coluna i contém os 6 coeficientes associados ao nó i.
    """
    P = np.asarray(P, dtype=float)
    
    # Se a matriz A não for fornecida, calcula-a a partir dos nós selecionados
    if matriz_a is None:
        coords_locais = nodes_coords[nos_selecionados] - P
        vecs = nodes_vectors[nos_selecionados]
        
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
    else:
        A = np.asarray(matriz_a, dtype=float)
        
    # Resolução do sistema A * beta = I_6 => beta = inv(A)
    beta = np.linalg.inv(A)
    
    # Funções de forma no ponto de avaliação P (onde dx=0, dy=0)
    # N_ix(P) = beta_1i, N_iy(P) = beta_2i
    Phi = beta[0:2, :]  # shape (2, 6)
    
    # Rotacional das funções de forma: curl(N_i) = (beta_5i - beta_4i) z_hat
    # onde beta_5i corresponde ao termo [0, dx]^T (índice 4 em 0-based) e beta_4i ao termo [dy, 0]^T (índice 3 em 0-based)
    rot_Phi = beta[4, :] - beta[3, :]  # shape (6,)
    
    return Phi, rot_Phi, beta
