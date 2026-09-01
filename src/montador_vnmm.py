import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

# Adiciona o diretório raiz e codigo ao sys.path
DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_SRC)
DIRETORIO_CODIGO = os.path.join(DIRETORIO_RAIZ, "codigo")
for p in [DIRETORIO_RAIZ, DIRETORIO_CODIGO]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nos_suporte_vnmm_2d_3_L1 import nos_suporte_vnmm_2d_3_L1
from funcoes_forma_vnmm_2d_3_L1 import funcoes_forma_vnmm_2d_3_L1
from nos_suporte_vnmm_2d_6_P1 import nos_suporte_vnmm_2d_6_P1
from funcoes_forma_vnmm_2d_6_P1 import funcoes_forma_vnmm_2d_6_P1
from src.quadratura_gauss import obter_pontos_pesos_gauss_1d


def montar_matrizes_vnmm_2d(
    coords, 
    vectors, 
    pontos_gauss=None, 
    pesos_gauss=None, 
    base="P1", 
    tolerancia_det=None, 
    mu_r=1.0, 
    epsilon_r=1.0,
    s_div=6.0,
    Ncx=None,
    Ncy=None,
    Lx=np.pi,
    Ly=np.pi,
    pontos_por_dir=3,
    modo_suporte="ponto_gauss"
):
    """
    Monta as matrizes globais de rigidez K (com regularização div-curl) e de massa M em formato CSR
    para o problema de autovalores eletromagnéticos bidimensionais (TEz):
    
        K = K_curl + s_div * K_div
        K_curl_ij = integral_Omega (1 / mu_r) * (rot N_i)_z * (rot N_j)_z dOmega
        K_div_ij  = integral_Omega (1 / mu_r) * (div N_i) * (div N_j) dOmega
        M_ij      = integral_Omega epsilon_r * (N_i . N_j) dOmega
        
    Parâmetros:
    - coords: Array numpy (N, 2) com as coordenadas dos nós.
    - vectors: Array numpy (N, 2) com as componentes dos vetores diretores unitários.
    - base: 'P1' (6 nós/termos, linear completa) ou 'L1' (3 nós/termos, Nedelec de 1ª ordem).
    - tolerancia_det: Tolerância do determinante |det(A)| para a seleção adaptativa de nós (padrão: adaptativo O(h^4)).
    - mu_r: Permeabilidade magnética relativa (padrão: 1.0).
    - epsilon_r: Permissividade elétrica relativa (padrão: 1.0).
    - s_div: Fator de penalidade de divergência para eliminação de modos espúrios de gradiente (padrão: 6.0).
    - Ncx: Número de células de integração na direção x (padrão: round(0.6 * Nx)).
    - Ncy: Número de células de integração na direção y (padrão: round(0.6 * Ny)).
    - Lx: Dimensão do domínio em x (padrão: pi).
    - Ly: Dimensão do domínio em y (padrão: pi).
    - pontos_por_dir: Número de pontos de Gauss por direção (padrão: 3 para 3x3 = 9 pontos/célula).
    - modo_suporte: 'ponto_gauss' (estilo EFG, nós determinados em cada ponto de integração)
                    ou 'centro_celula' (nós fixados no centro de cada célula).
    
    Retorna:
    - K: Matriz esparsa csr_matrix (N, N) de rigidez regularizada.
    - M: Matriz esparsa csr_matrix (N, N) de massa.
    """
    N_total = len(coords)
    arvore = KDTree(coords)
    
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    tipo_base = base.upper()
    if tipo_base in ["P1", "6_P1", "6NOS"]:
        is_P1 = True
    elif tipo_base in ["L1", "3_L1", "3NOS"]:
        is_P1 = False
    else:
        raise ValueError(f"Base '{base}' não suportada. Opções: 'P1' ou 'L1'.")
        
    # Determina o número de células de integração e espaçamento característico
    N_lado = int(np.round(np.sqrt(N_total)))
    if Ncx is None:
        Ncx = max(4, int(np.round(0.6 * N_lado)))
    if Ncy is None:
        Ncy = max(4, int(np.round(0.6 * N_lado)))
        
    h_char = max(Lx / max(N_lado - 1, 1), Ly / max(N_lado - 1, 1))
    h_ref = np.pi / 20.0
    if tolerancia_det is None:
        tolerancia_det = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)
    
    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []
    
    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)
        
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy
            
            if modo_suporte == "centro_celula":
                Pc = np.array([xc, yc])
                # 1. Seleção dos nós de suporte para o centro da célula
                if is_P1:
                    nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_6_P1(
                        P=Pc,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        arvore_busca=arvore,
                        K=12,
                        Tol_det=tolerancia_det,
                        adaptativo=True,
                        passo_K=4
                    )
                    beta = np.linalg.inv(A_mat)
                    rot_Phi = beta[4, :] - beta[3, :]      # shape (6,)
                    div_Phi = beta[2, :] + beta[5, :]      # shape (6,)
                    n_supp = 6
                else:
                    nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_3_L1(
                        P=Pc,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        arvore_busca=arvore,
                        K=8,
                        Tol_det=tolerancia_det,
                        adaptativo=True,
                        passo_K=2
                    )
                    beta = np.linalg.inv(A_mat)
                    rot_Phi = -2.0 * beta[2, :]             # shape (3,)
                    div_Phi = np.zeros(3, dtype=float)     # Base L1 é identicamente solenoidal
                    n_supp = 3
                
                # 2. Integração numérica sobre os pontos de Gauss da célula
                for wi, xi in zip(w_1d, xi_1d):
                    xg = xc + 0.5 * dx * xi
                    dx_g = xg - xc
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        dy_g = yg - yc
                        
                        wg = wi * wj * det_J
                        
                        if is_P1:
                            Phi_x = beta[0, :] + beta[2, :] * dx_g + beta[3, :] * dy_g
                            Phi_y = beta[1, :] + beta[4, :] * dx_g + beta[5, :] * dy_g
                        else:
                            Phi_x = beta[0, :] + beta[2, :] * dy_g
                            Phi_y = beta[1, :] - beta[2, :] * dx_g
                            
                        Phi_g = np.vstack([Phi_x, Phi_y])  # shape (2, n_supp)
                        
                        rot_outer = np.outer(rot_Phi, rot_Phi) * (inv_mu * wg)
                        div_outer = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                        vec_dot = (Phi_g.T @ Phi_g) * (eps * wg)
                        
                        for a in range(n_supp):
                            Ia = nos[a]
                            for b in range(n_supp):
                                Ib = nos[b]
                                
                                val_kc = rot_outer[a, b]
                                val_kd = div_outer[a, b]
                                val_m = vec_dot[a, b]
                                
                                if abs(val_kc) > 1e-18:
                                    rows_Kc.append(Ia)
                                    cols_Kc.append(Ib)
                                    data_Kc.append(val_kc)
                                    
                                if abs(val_kd) > 1e-18:
                                    rows_Kd.append(Ia)
                                    cols_Kd.append(Ib)
                                    data_Kd.append(val_kd)
                                    
                                if abs(val_m) > 1e-18:
                                    rows_M.append(Ia)
                                    cols_M.append(Ib)
                                    data_M.append(val_m)
                                    
            else:
                # modo_suporte == "ponto_gauss" (Estilo EFG: suporte individual por ponto de integração)
                for wi, xi in zip(w_1d, xi_1d):
                    xg = xc + 0.5 * dx * xi
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        Pg = np.array([xg, yg])
                        wg = wi * wj * det_J
                        
                        if is_P1:
                            nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_6_P1(
                                P=Pg,
                                nodes_coords=coords,
                                nodes_vectors=vectors,
                                arvore_busca=arvore,
                                K=12,
                                Tol_det=tolerancia_det,
                                adaptativo=True,
                                passo_K=4
                            )
                            beta = np.linalg.inv(A_mat)
                            # Como a origem local é o próprio Pg (Delta_x=0, Delta_y=0):
                            Phi_g = beta[0:2, :]                   # shape (2, 6)
                            rot_Phi = beta[4, :] - beta[3, :]      # shape (6,)
                            div_Phi = beta[2, :] + beta[5, :]      # shape (6,)
                            n_supp = 6
                        else:
                            nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_3_L1(
                                P=Pg,
                                nodes_coords=coords,
                                nodes_vectors=vectors,
                                arvore_busca=arvore,
                                K=8,
                                Tol_det=tolerancia_det,
                                adaptativo=True,
                                passo_K=2
                            )
                            beta = np.linalg.inv(A_mat)
                            # Como a origem local é o próprio Pg:
                            Phi_g = beta[0:2, :]                   # shape (2, 3)
                            rot_Phi = -2.0 * beta[2, :]             # shape (3,)
                            div_Phi = np.zeros(3, dtype=float)     # Base L1 é identicamente solenoidal
                            n_supp = 3
                            
                        rot_outer = np.outer(rot_Phi, rot_Phi) * (inv_mu * wg)
                        div_outer = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                        vec_dot = (Phi_g.T @ Phi_g) * (eps * wg)
                        
                        for a in range(n_supp):
                            Ia = nos[a]
                            for b in range(n_supp):
                                Ib = nos[b]
                                
                                val_kc = rot_outer[a, b]
                                val_kd = div_outer[a, b]
                                val_m = vec_dot[a, b]
                                
                                if abs(val_kc) > 1e-18:
                                    rows_Kc.append(Ia)
                                    cols_Kc.append(Ib)
                                    data_Kc.append(val_kc)
                                    
                                if abs(val_kd) > 1e-18:
                                    rows_Kd.append(Ia)
                                    cols_Kd.append(Ib)
                                    data_Kd.append(val_kd)
                                    
                                if abs(val_m) > 1e-18:
                                    rows_M.append(Ia)
                                    cols_M.append(Ib)
                                    data_M.append(val_m)
                                    
    # 3. Montagem e simetrização
    Kc = coo_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(N_total, N_total)).tocsr()
    Kd = coo_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(N_total, N_total)).tocsr()
    M = coo_matrix((data_M, (rows_M, cols_M)), shape=(N_total, N_total)).tocsr()
    
    Kc = 0.5 * (Kc + Kc.T)
    Kd = 0.5 * (Kd + Kd.T)
    M = 0.5 * (M + M.T)
    
    K = Kc + s_div * Kd
    
    return K, M

