import os
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

DIRETORIO_SRC = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_SRC)
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from src.fem_edge_2d import montar_matrizes_fem_aresta_2d
from src.quadratura_gauss import obter_pontos_pesos_gauss_1d
from codigo.nos_suporte_vnmm_2d_6_P1 import nos_suporte_vnmm_2d_6_P1
from src.eigen_solver_cavity import MODOS_ANALITICOS_REF


def gerar_malha_hibrida_cavidade(
    Lx=np.pi, 
    Ly=np.pi, 
    frac_fem=0.5, 
    Nex_fem=8, 
    Ney=12, 
    Nx_vnmm=9, 
    Ny_vnmm=13,
    tipo_interior_vnmm="alternado"
):
    """
    Gera as malhas particionadas para o acoplamento híbrido FEM-VNMM:
    - Subdomínio 1 (FEM): [0, x_int] x [0, Ly] com malha triangular estruturada (Nex_fem x Ney células).
    - Subdomínio 2 (VNMM): [x_int, Lx] x [0, Ly] com nuvem de nós.
      Na interface x = x_int, os nós de contorno do VNMM são posicionados exatamente
      nos pontos médios das Ney arestas verticais da interface FEM, com vetor t = [0, 1]^T.
    
    Retorna:
    - dados_fem: dicionário com nós, elementos, arestas e mapeamento de interface.
    - dados_vnmm: dicionário com coordenadas dos nós, vetores diretores e índices de contorno/interface.
    """
    x_int = frac_fem * Lx
    
    # =========================================================================
    # 1. SUBDOMÍNIO FEM [0, x_int] x [0, Ly]
    # =========================================================================
    x_lin_fem = np.linspace(0.0, x_int, Nex_fem + 1)
    y_lin_fem = np.linspace(0.0, Ly, Ney + 1)
    
    nodes_fem = []
    node_grid_fem = np.zeros((Ney + 1, Nex_fem + 1), dtype=int)
    node_id = 0
    for j in range(Ney + 1):
        for i in range(Nex_fem + 1):
            nodes_fem.append([x_lin_fem[i], y_lin_fem[j]])
            node_grid_fem[j, i] = node_id
            node_id += 1
    nodes_fem = np.array(nodes_fem, dtype=float)
    
    elements_fem = []
    for j in range(Ney):
        for i in range(Nex_fem):
            n_bl = node_grid_fem[j, i]
            n_br = node_grid_fem[j, i + 1]
            n_tl = node_grid_fem[j + 1, i]
            n_tr = node_grid_fem[j + 1, i + 1]
            
            elements_fem.append([n_bl, n_br, n_tr])
            elements_fem.append([n_bl, n_tr, n_tl])
    elements_fem = np.array(elements_fem, dtype=int)
    
    # Identificação de arestas do FEM
    edge_dict_fem = {}
    elem_edges_fem = np.zeros((len(elements_fem), 3), dtype=int)
    elem_edge_signs_fem = np.zeros((len(elements_fem), 3), dtype=float)
    
    edge_id = 0
    for elem_idx, elem in enumerate(elements_fem):
        local_edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for local_idx, (n1, n2) in enumerate(local_edges):
            ordered = (min(n1, n2), max(n1, n2))
            if ordered not in edge_dict_fem:
                edge_dict_fem[ordered] = edge_id
                global_eid = edge_id
                edge_id += 1
            else:
                global_eid = edge_dict_fem[ordered]
                
            elem_edges_fem[elem_idx, local_idx] = global_eid
            elem_edge_signs_fem[elem_idx, local_idx] = 1.0 if n1 < n2 else -1.0
            
    edges_fem = np.zeros((edge_id, 2), dtype=int)
    for (n1, n2), eid in edge_dict_fem.items():
        edges_fem[eid] = [n1, n2]
        
    # Classificação das arestas FEM: PEC externa vs Interface vs Interna
    tol = 1e-7
    is_pec_edge_fem = np.zeros(edge_id, dtype=bool)
    is_interface_edge_fem = np.zeros(edge_id, dtype=bool)
    interface_edge_ids = []
    interface_edge_midpoints = []
    
    for eid in range(edge_id):
        n1, n2 = edges_fem[eid]
        x1, y1 = nodes_fem[n1]
        x2, y2 = nodes_fem[n2]
        
        # Borda externa esquerda (x=0), inferior (y=0) ou superior (y=Ly)
        if (abs(x1) < tol and abs(x2) < tol) or (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - Ly) < tol and abs(y2 - Ly) < tol):
            is_pec_edge_fem[eid] = True
        # Borda de Interface vertical em x = x_int
        elif (abs(x1 - x_int) < tol and abs(x2 - x_int) < tol):
            is_interface_edge_fem[eid] = True
            interface_edge_ids.append(eid)
            pm = 0.5 * (np.array([x1, y1]) + np.array([x2, y2]))
            interface_edge_midpoints.append((pm[1], eid, pm))
            
    # Ordena as arestas de interface por y crescente
    interface_edge_midpoints.sort(key=lambda item: item[0])
    ordered_interface_eids = [item[1] for item in interface_edge_midpoints]
    ordered_interface_pms = np.array([item[2] for item in interface_edge_midpoints])
    
    dados_fem = {
        'nodes': nodes_fem,
        'elements': elements_fem,
        'edges': edges_fem,
        'elem_edges': elem_edges_fem,
        'elem_edge_signs': elem_edge_signs_fem,
        'is_pec_edge': is_pec_edge_fem,
        'is_interface_edge': is_interface_edge_fem,
        'interface_eids': ordered_interface_eids,
        'interface_midpoints': ordered_interface_pms,
        'x_int': x_int,
        'Lx': Lx,
        'Ly': Ly
    }
    
    # =========================================================================
    # 2. SUBDOMÍNIO VNMM [x_int, Lx] x [0, Ly]
    # =========================================================================
    x_lin_vnmm = np.linspace(x_int, Lx, Nx_vnmm)
    y_lin_vnmm = np.linspace(0.0, Ly, Ny_vnmm)
    
    nodes_vnmm = []
    vectors_vnmm = []
    is_pec_vnmm = []
    is_interface_vnmm = []
    interface_node_map = {} # interface_idx -> vnmm_node_id
    
    # 2.1 Nós de interface (x = x_int): posicionados exatamente nos pontos médios das arestas FEM
    # com vetor t = [0, 1] perfeitamente alinhado à direção da aresta vertical orientada de baixo para cima
    for int_idx, pm in enumerate(ordered_interface_pms):
        vnmm_nid = len(nodes_vnmm)
        nodes_vnmm.append(pm)
        vectors_vnmm.append([0.0, 1.0]) # Sentido positivo de y
        is_pec_vnmm.append(False)
        is_interface_vnmm.append(True)
        interface_node_map[int_idx] = vnmm_nid
        
    # 2.2 Nós dos cantos na interface: (x_int, 0) e (x_int, Ly) com vetor horizontal [1, 0] (PEC)
    nodes_vnmm.append([x_int, 0.0])
    vectors_vnmm.append([1.0, 0.0])
    is_pec_vnmm.append(True)
    is_interface_vnmm.append(False)
    
    nodes_vnmm.append([x_int, Ly])
    vectors_vnmm.append([1.0, 0.0])
    is_pec_vnmm.append(True)
    is_interface_vnmm.append(False)
    
    # 2.3 Nós do interior e fronteiras externas PEC do subdomínio VNMM
    for i in range(1, Nx_vnmm): # i=0 é a interface, já criada acima
        x = x_lin_vnmm[i]
        for j in range(Ny_vnmm):
            y = y_lin_vnmm[j]
            vnmm_nid = len(nodes_vnmm)
            nodes_vnmm.append([x, y])
            
            # Fronteira externa PEC direita (x = Lx)
            if abs(x - Lx) < tol:
                vectors_vnmm.append([0.0, 1.0]) # Tangente à parede vertical
                is_pec_vnmm.append(True)
                is_interface_vnmm.append(False)
            # Fronteira externa PEC inferior (y = 0)
            elif abs(y) < tol:
                vectors_vnmm.append([1.0, 0.0]) # Tangente à parede horizontal
                is_pec_vnmm.append(True)
                is_interface_vnmm.append(False)
            # Fronteira externa PEC superior (y = Ly)
            elif abs(y - Ly) < tol:
                vectors_vnmm.append([1.0, 0.0]) # Tangente à parede horizontal
                is_pec_vnmm.append(True)
                is_interface_vnmm.append(False)
            # Nós internos de VNMM
            else:
                is_pec_vnmm.append(False)
                is_interface_vnmm.append(False)
                if tipo_interior_vnmm == "alternado":
                    # Alternância 45° e 135°
                    ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                    vectors_vnmm.append([np.cos(ang), np.sin(ang)])
                else:
                    vectors_vnmm.append([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
                    
    nodes_vnmm = np.array(nodes_vnmm, dtype=float)
    vectors_vnmm = np.array(vectors_vnmm, dtype=float)
    is_pec_vnmm = np.array(is_pec_vnmm, dtype=bool)
    is_interface_vnmm = np.array(is_interface_vnmm, dtype=bool)
    
    dados_vnmm = {
        'coords': nodes_vnmm,
        'vectors': vectors_vnmm,
        'is_pec': is_pec_vnmm,
        'is_interface': is_interface_vnmm,
        'interface_node_map': interface_node_map,
        'x_int': x_int,
        'Lx': Lx,
        'Ly': Ly,
        'Nx': Nx_vnmm,
        'Ny': Ny_vnmm
    }
    
    return dados_fem, dados_vnmm


def montar_matrizes_hibridas_fem_vnmm(
    dados_fem, 
    dados_vnmm, 
    Ncx_vnmm=8, 
    Ncy_vnmm=10, 
    pontos_por_dir=2, 
    s_div_vnmm=6.0, 
    mu_r=1.0, 
    epsilon_r=1.0
):
    """
    Monta as matrizes de rigidez K_hibrido e massa M_hibrido acopladas pelo método direto conforme:
    - Graus de Liberdade Mestres:
      1. e_fem_int: arestas internas do subdomínio FEM.
      2. e_gamma: arestas da interface Gamma_int (circulação [V]).
      3. c_vnmm_int: nós internos do subdomínio VNMM (campo [V/m]).
    - Relação Dimensional de Interface:
      c_gamma_k = (1 / delta_y_k) * e_gamma_k, com vetor unitário t alinhado à aresta orientada.
    """
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    # -------------------------------------------------------------------------
    # 1. MONTAGEM DO SUBDOMÍNIO FEM
    # -------------------------------------------------------------------------
    K_fem_full, M_fem_full = montar_matrizes_fem_aresta_2d(
        dados_fem['nodes'],
        dados_fem['elements'],
        dados_fem['edges'],
        dados_fem['elem_edges'],
        dados_fem['elem_edge_signs'],
        mu_r=mu_r,
        epsilon_r=epsilon_r
    )
    
    # Separação de índices de arestas FEM: Internas, Interface e PEC externa
    idx_fem_int = np.where(~dados_fem['is_pec_edge'] & ~dados_fem['is_interface_edge'])[0]
    idx_fem_gamma = np.array(dados_fem['interface_eids'], dtype=int)
    
    N_fem_int = len(idx_fem_int)
    N_gamma = len(idx_fem_gamma)
    
    # -------------------------------------------------------------------------
    # 2. MONTAGEM DO SUBDOMÍNIO VNMM COM CÉLULAS EM [x_int, Lx] x [0, Ly]
    # -------------------------------------------------------------------------
    coords_vnmm = dados_vnmm['coords']
    vectors_vnmm = dados_vnmm['vectors']
    N_vnmm_total = len(coords_vnmm)
    
    arvore_vnmm = KDTree(coords_vnmm)
    x_int = dados_vnmm['x_int']
    Lx = dados_vnmm['Lx']
    Ly = dados_vnmm['Ly']
    
    x_edges_vnmm = np.linspace(x_int, Lx, Ncx_vnmm + 1)
    y_edges_vnmm = np.linspace(0.0, Ly, Ncy_vnmm + 1)
    
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)
    
    h_ref = np.pi / 20.0
    h_vnmm = max((Lx - x_int) / (dados_vnmm['Nx'] - 1), Ly / (dados_vnmm['Ny'] - 1))
    tol_det_vnmm = 1e-4 * (h_vnmm / h_ref)**4
    
    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_Mv, cols_Mv, data_Mv = [], [], []
    
    for j in range(Ncy_vnmm):
        y0, y1 = y_edges_vnmm[j], y_edges_vnmm[j + 1]
        dy = y1 - y0
        y_mid = 0.5 * (y0 + y1)
        for i in range(Ncx_vnmm):
            x0, x1 = x_edges_vnmm[i], x_edges_vnmm[i + 1]
            dx = x1 - x0
            x_mid = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy
            
            for wi, xi in zip(w_1d, xi_1d):
                xg = x_mid + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = y_mid + 0.5 * dy * eta
                    Pg = np.array([xg, yg])
                    
                    nos, _, A_mat, _ = nos_suporte_vnmm_2d_6_P1(
                        P=Pg, 
                        nodes_coords=coords_vnmm, 
                        nodes_vectors=vectors_vnmm, 
                        arvore_busca=arvore_vnmm, 
                        K=12, 
                        Tol_det=tol_det_vnmm, 
                        adaptativo=True
                    )
                    beta = la.inv(A_mat)
                    
                    Phi_x = beta[0, :]
                    Phi_y = beta[1, :]
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]
                    
                    peso = wi * wj * det_J
                    Kc_elem = inv_mu * peso * np.outer(rot_Phi, rot_Phi)
                    Kd_elem = s_div_vnmm * inv_mu * peso * np.outer(div_Phi, div_Phi)
                    M_elem = eps * peso * (np.outer(Phi_x, Phi_x) + np.outer(Phi_y, Phi_y))
                    
                    for a in range(6):
                        na = nos[a]
                        for b in range(6):
                            nb = nos[b]
                            rows_Kc.append(na); cols_Kc.append(nb); data_Kc.append(Kc_elem[a, b])
                            rows_Kd.append(na); cols_Kd.append(nb); data_Kd.append(Kd_elem[a, b])
                            rows_Mv.append(na); cols_Mv.append(nb); data_Mv.append(M_elem[a, b])
                        
    K_vnmm_full = coo_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
    if s_div_vnmm > 0.0:
        Kd_vnmm = coo_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
        K_vnmm_full = K_vnmm_full + Kd_vnmm
    M_vnmm_full = coo_matrix((data_Mv, (rows_Mv, cols_Mv)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
    
    # Separação de índices de nós VNMM: Internos vs Interface
    idx_vnmm_int = np.where(~dados_vnmm['is_pec'] & ~dados_vnmm['is_interface'])[0]
    idx_vnmm_gamma = np.array([dados_vnmm['interface_node_map'][k] for k in range(N_gamma)], dtype=int)
    
    N_vnmm_int = len(idx_vnmm_int)
    
    # Fator de escala dimensional da interface: delta_y de cada aresta vertical
    delta_y_gamma = Ly / N_gamma
    T_val = 1.0 / delta_y_gamma
    
    # -------------------------------------------------------------------------
    # 3. ACOPLAMENTO GLOBAL DAS MATRIZES EM BLOCOS
    # -------------------------------------------------------------------------
    N_global = N_fem_int + N_gamma + N_vnmm_int
    
    K_glob = np.zeros((N_global, N_global), dtype=float)
    M_glob = np.zeros((N_global, N_global), dtype=float)
    
    s_f_int = slice(0, N_fem_int)
    s_gamma = slice(N_fem_int, N_fem_int + N_gamma)
    s_v_int = slice(N_fem_int + N_gamma, N_global)
    
    # A. Contribuição FEM
    K_f_dense = K_fem_full.toarray()
    M_f_dense = M_fem_full.toarray()
    
    K_glob[s_f_int, s_f_int] += K_f_dense[idx_fem_int, :][:, idx_fem_int]
    K_glob[s_f_int, s_gamma] += K_f_dense[idx_fem_int, :][:, idx_fem_gamma]
    K_glob[s_gamma, s_f_int] += K_f_dense[idx_fem_gamma, :][:, idx_fem_int]
    K_glob[s_gamma, s_gamma] += K_f_dense[idx_fem_gamma, :][:, idx_fem_gamma]
    
    M_glob[s_f_int, s_f_int] += M_f_dense[idx_fem_int, :][:, idx_fem_int]
    M_glob[s_f_int, s_gamma] += M_f_dense[idx_fem_int, :][:, idx_fem_gamma]
    M_glob[s_gamma, s_f_int] += M_f_dense[idx_fem_gamma, :][:, idx_fem_int]
    M_glob[s_gamma, s_gamma] += M_f_dense[idx_fem_gamma, :][:, idx_fem_gamma]
    
    # B. Contribuição VNMM com a transformação T_gamma
    K_v_dense = K_vnmm_full.toarray()
    M_v_dense = M_vnmm_full.toarray()
    
    K_glob[s_gamma, s_gamma] += (T_val**2) * K_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_gamma]
    K_glob[s_gamma, s_v_int] += T_val * K_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_int]
    K_glob[s_v_int, s_gamma] += T_val * K_v_dense[idx_vnmm_int, :][:, idx_vnmm_gamma]
    K_glob[s_v_int, s_v_int] += K_v_dense[idx_vnmm_int, :][:, idx_vnmm_int]
    
    M_glob[s_gamma, s_gamma] += (T_val**2) * M_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_gamma]
    M_glob[s_gamma, s_v_int] += T_val * M_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_int]
    M_glob[s_v_int, s_gamma] += T_val * M_v_dense[idx_vnmm_int, :][:, idx_vnmm_gamma]
    M_glob[s_v_int, s_v_int] += M_v_dense[idx_vnmm_int, :][:, idx_vnmm_int]
    
    # Simetrização numérica
    K_glob = 0.5 * (K_glob + K_glob.T)
    M_glob = 0.5 * (M_glob + M_glob.T)
    
    info_dofs = {
        'N_global': N_global,
        'N_fem_int': N_fem_int,
        'N_gamma': N_gamma,
        'N_vnmm_int': N_vnmm_int,
        'delta_y_gamma': delta_y_gamma
    }
    
    return csr_matrix(K_glob), csr_matrix(M_glob), info_dofs


def resolver_autovalores_hibrido_fem_vnmm(
    Lx=np.pi, 
    Ly=np.pi, 
    frac_fem=0.5, 
    Nex_fem=8, 
    Ney=12, 
    Nx_vnmm=9, 
    Ny_vnmm=13,
    Ncx_vnmm=8, 
    Ncy_vnmm=10, 
    s_div_vnmm=6.0, 
    num_autovalores=10, 
    tol_zero=0.1
):
    """
    Pipeline completo do solver híbrido acoplado FEM de Aresta + VNMM 2D.
    """
    dados_fem, dados_vnmm = gerar_malha_hibrida_cavidade(
        Lx=Lx, Ly=Ly, frac_fem=frac_fem, Nex_fem=Nex_fem, Ney=Ney,
        Nx_vnmm=Nx_vnmm, Ny_vnmm=Ny_vnmm
    )
    
    K_glob, M_glob, info_dofs = montar_matrizes_hibridas_fem_vnmm(
        dados_fem, dados_vnmm, Ncx_vnmm=Ncx_vnmm, Ncy_vnmm=Ncy_vnmm,
        s_div_vnmm=s_div_vnmm
    )
    
    K_dense = K_glob.toarray()
    M_dense = M_glob.toarray()
    
    try:
        vals, vecs = la.eigh(K_dense, M_dense)
    except Exception:
        vals, vecs = la.eig(K_dense, M_dense)
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    mascara_positivos = vals > tol_zero
    vals_positivos = np.sort(vals[mascara_positivos])
    n_nulos = np.sum(~mascara_positivos)
    
    # Identifica os modos correspondentes aos 10 primeiros modos analíticos
    ref_vals = np.array([item[2] for item in MODOS_ANALITICOS_REF[:num_autovalores]])
    ref_kc = np.array([item[3] for item in MODOS_ANALITICOS_REF[:num_autovalores]])
    
    # Seleção por proximidade espectral para cada modo analítico
    autovalores_num = []
    for l_ref in ref_vals:
        # Encontra o autovalor positivo mais próximo de l_ref
        idx_prox = np.argmin(np.abs(vals_positivos - l_ref))
        autovalores_num.append(vals_positivos[idx_prox])
    autovalores_num = np.array(autovalores_num)
    
    kc_num = np.sqrt(np.maximum(autovalores_num, 0.0))
    erros_lambda = np.abs(autovalores_num - ref_vals) / ref_vals * 100.0
    erros_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    return {
        'info_dofs': info_dofs,
        'autovalores_todos': vals,
        'autovalores_numericos': autovalores_num,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': erros_lambda,
        'erros_kc_pct': erros_kc,
        'erro_medio_lambda_pct': float(np.mean(erros_lambda)),
        'erro_medio_kc_pct': float(np.mean(erros_kc)),
        'erro_max_kc_pct': float(np.max(erros_kc)),
        'n_nulos_descartados': int(n_nulos),
        'K_glob': K_glob,
        'M_glob': M_glob
    }
