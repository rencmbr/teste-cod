import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs


def gerar_malha_triangular_cavidade(
    Nex=11, 
    Ney=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    jitter_frac=0.0, 
    seed=42
):
    """
    Gera uma malha triangular 2D cobrindo o domínio [0, Lx] x [0, Ly].
    Cada célula retangular é subdividida em 2 triângulos.
    Suporta perturbação estocástica controlada (jitter) para os nós internos.
    
    Retorna:
    - nodes: Array numpy (N_nodes, 2) com as coordenadas cartesianas (x, y) de cada vértice.
    - elements: Array numpy (N_elem, 3) com os índices dos 3 vértices de cada triângulo (em sentido anti-horário).
    - edges: Array numpy (N_edges, 2) com os pares (n1, n2) de cada aresta orientada (com n1 < n2).
    - elem_edges: Array numpy (N_elem, 3) com os índices globais das 3 arestas de cada elemento.
    - elem_edge_signs: Array numpy (N_elem, 3) com +1 ou -1 indicando a orientação relativa da aresta no elemento.
    - is_boundary_edge: Array booleano (N_edges,) indicando True para arestas na fronteira PEC.
    """
    if seed is not None:
        np.random.seed(seed)
        
    x_lin = np.linspace(0.0, Lx, Nex + 1)
    y_lin = np.linspace(0.0, Ly, Ney + 1)
    dx = Lx / Nex
    dy = Ly / Ney
    
    # 1. Criação dos nós
    nodes_list = []
    node_grid = np.zeros((Ney + 1, Nex + 1), dtype=int)
    node_id = 0
    for j in range(Ney + 1):
        for i in range(Nex + 1):
            x, y = x_lin[i], y_lin[j]
            # Aplica perturbação aleatória estocástica (jitter) apenas aos nós estritamente internos
            if jitter_frac > 0.0 and 0 < i < Nex and 0 < j < Ney:
                x += np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                y += np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
            nodes_list.append([x, y])
            node_grid[j, i] = node_id
            node_id += 1
    nodes = np.array(nodes_list, dtype=float)
    
    # 2. Criação dos elementos triangulares (sentido anti-horário)
    elements_list = []
    for j in range(Ney):
        for i in range(Nex):
            n_bl = node_grid[j, i]       # bottom-left
            n_br = node_grid[j, i + 1]   # bottom-right
            n_tl = node_grid[j + 1, i]   # top-left
            n_tr = node_grid[j + 1, i + 1] # top-right
            
            # Triângulo 1 (inferior direito): n_bl -> n_br -> n_tr
            elements_list.append([n_bl, n_br, n_tr])
            # Triângulo 2 (superior esquerdo): n_bl -> n_tr -> n_tl
            elements_list.append([n_bl, n_tr, n_tl])
            
    elements = np.array(elements_list, dtype=int)
    
    # 3. Mapeamento de arestas globais únicas e orientadas
    edge_dict = {} # (min(n1, n2), max(n1, n2)) -> edge_id
    elem_edges = np.zeros((len(elements), 3), dtype=int)
    elem_edge_signs = np.zeros((len(elements), 3), dtype=float)
    
    edge_id = 0
    for elem_idx, elem in enumerate(elements):
        # 3 arestas locais: (0, 1), (1, 2), (2, 0)
        local_edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        
        for local_idx, (n1, n2) in enumerate(local_edges):
            ordered_edge = (min(n1, n2), max(n1, n2))
            if ordered_edge not in edge_dict:
                edge_dict[ordered_edge] = edge_id
                global_eid = edge_id
                edge_id += 1
            else:
                global_eid = edge_dict[ordered_edge]
                
            elem_edges[elem_idx, local_idx] = global_eid
            # Se a aresta local vai de n1 -> n2 e n1 < n2, o sinal é +1; caso contrário -1
            elem_edge_signs[elem_idx, local_idx] = 1.0 if n1 < n2 else -1.0
            
    # Cria a lista de arestas globais
    edges = np.zeros((edge_id, 2), dtype=int)
    for (n1, n2), eid in edge_dict.items():
        edges[eid] = [n1, n2]
        
    # 4. Identificação de arestas de fronteira PEC
    is_boundary_edge = np.zeros(edge_id, dtype=bool)
    tol = 1e-7
    for eid in range(edge_id):
        n1, n2 = edges[eid]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        
        # Borda inferior (y=0) ou superior (y=Ly)
        if (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - Ly) < tol and abs(y2 - Ly) < tol):
            is_boundary_edge[eid] = True
        # Borda esquerda (x=0) ou direita (x=Lx)
        elif (abs(x1) < tol and abs(x2) < tol) or (abs(x1 - Lx) < tol and abs(x2 - Lx) < tol):
            is_boundary_edge[eid] = True
            
    return nodes, elements, edges, elem_edges, elem_edge_signs, is_boundary_edge


def montar_matrizes_fem_aresta_2d(
    nodes, 
    elements, 
    edges, 
    elem_edges, 
    elem_edge_signs, 
    mu_r=1.0, 
    epsilon_r=1.0
):
    """
    Monta as matrizes esparsas globais de rigidez K_curl e de massa M para elementos de aresta triangulares
    de Nédélec de 1ª ordem (1-formas de Whitney).
    
    K_ij = integral_Omega (1/mu_r) * (rot W_i)_z * (rot W_j)_z dOmega
    M_ij = integral_Omega epsilon_r * (W_i . W_j) dOmega
    """
    N_edges = len(edges)
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    rows_K, cols_K, data_K = [], [], []
    rows_M, cols_M, data_M = [], [], []
    
    for e_idx, elem in enumerate(elements):
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Área do triângulo (2 * Area = (x2-x1)(y3-y1) - (x3-x1)(y2-y1))
        det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * det2
        if area <= 0:
            raise ValueError(f"Triângulo {e_idx} possui área não-positiva: {area}")
            
        # Gradientes das coordenadas baricêntricas L1, L2, L3
        grad_L = np.zeros((3, 2), dtype=float)
        grad_L[0] = [(y2 - y3) / det2, (x3 - x2) / det2]
        grad_L[1] = [(y3 - y1) / det2, (x1 - x3) / det2]
        grad_L[2] = [(y1 - y2) / det2, (x2 - x1) / det2]
        
        # 3 arestas locais com pares de nós: (0, 1), (1, 2), (2, 0)
        local_nodes = [(0, 1), (1, 2), (2, 0)]
        signs = elem_edge_signs[e_idx]
        global_eids = elem_edges[e_idx]
        
        # 1. Matriz de Rigidez Elementar K_e (Rotacional Constante no Triângulo)
        # (rot w_k)_z = sign_k / area
        rot_W = np.zeros(3, dtype=float)
        for k in range(3):
            rot_W[k] = signs[k] / area
            
        K_e = (inv_mu * area) * np.outer(rot_W, rot_W)
        
        # 2. Matriz de Massa Elementar M_e (Integração Exata Baricêntrica)
        # w_k = sign_k * (L_i grad_L_j - L_j grad_L_i)
        # Integral de L_p * L_q em T_e = area/6 se p==q e area/12 se p!=q
        M_e = np.zeros((3, 3), dtype=float)
        for a in range(3):
            ia, ja = local_nodes[a]
            sa = signs[a]
            for b in range(3):
                ib, jb = local_nodes[b]
                sb = signs[b]
                
                # Produto escalar expandido
                # w_a . w_b = sa*sb * [ (L_ia L_ib)(grad_ja . grad_jb) - (L_ia L_jb)(grad_ja . grad_ib)
                #                     - (L_ja L_ib)(grad_ia . grad_jb) + (L_ja L_jb)(grad_ia . grad_ib) ]
                
                def int_LL(p, q):
                    return area / 6.0 if p == q else area / 12.0
                    
                term1 = int_LL(ia, ib) * np.dot(grad_L[ja], grad_L[jb])
                term2 = int_LL(ia, jb) * np.dot(grad_L[ja], grad_L[ib])
                term3 = int_LL(ja, ib) * np.dot(grad_L[ia], grad_L[jb])
                term4 = int_LL(ja, jb) * np.dot(grad_L[ia], grad_L[ib])
                
                val_m = eps * sa * sb * (term1 - term2 - term3 + term4)
                M_e[a, b] = val_m
                
        # Acumula nas matrizes globais
        for a in range(3):
            ea = global_eids[a]
            for b in range(3):
                eb = global_eids[b]
                
                rows_K.append(ea)
                cols_K.append(eb)
                data_K.append(K_e[a, b])
                
                rows_M.append(ea)
                cols_M.append(eb)
                data_M.append(M_e[a, b])
                
    K_glob = coo_matrix((data_K, (rows_K, cols_K)), shape=(N_edges, N_edges)).tocsr()
    M_glob = coo_matrix((data_M, (rows_M, cols_M)), shape=(N_edges, N_edges)).tocsr()
    
    K_glob = 0.5 * (K_glob + K_glob.T)
    M_glob = 0.5 * (M_glob + M_glob.T)
    
    return K_glob, M_glob


def resolver_autovalores_fem_aresta_2d(
    Nex=11, 
    Ney=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    mu_r=1.0, 
    epsilon_r=1.0, 
    num_autovalores=10, 
    tol_zero=1e-3,
    jitter_frac=0.0,
    seed=42
):
    """
    Pipeline completo do solver de autovalores com elementos de aresta triangulares 2D (Nédélec 1ª ordem).
    
    Retorna dicionário completo com os autovalores, erros percentuais e estatísticas da malha.
    """
    nodes, elements, edges, elem_edges, elem_edge_signs, is_boundary_edge = gerar_malha_triangular_cavidade(
        Nex=Nex, Ney=Ney, Lx=Lx, Ly=Ly, jitter_frac=jitter_frac, seed=seed
    )
    
    K_glob, M_glob = montar_matrizes_fem_aresta_2d(
        nodes, elements, edges, elem_edges, elem_edge_signs, mu_r=mu_r, epsilon_r=epsilon_r
    )
    
    # Aplica condição de contorno PEC (Dirichlet homogênea E_t = 0 nas arestas de fronteira)
    idx_internos = np.where(~is_boundary_edge)[0]
    K_red = K_glob[idx_internos, :][:, idx_internos]
    M_red = M_glob[idx_internos, :][:, idx_internos]
    
    # Resolução do problema generalizado de autovalores
    try:
        vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
    except Exception:
        vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    # Filtra e descarta os autovalores nulos de gradiente (espaço nulo exato de rotacional)
    mascara_positivos = vals > tol_zero
    vals_positivos = np.sort(vals[mascara_positivos])
    n_nulos = np.sum(~mascara_positivos)
    
    autovalores_num = vals_positivos[:num_autovalores]
    
    from src.eigen_solver_cavity import MODOS_ANALITICOS_REF
    ref_vals = np.array([item[2] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    ref_kc = np.array([item[3] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    
    kc_num = np.sqrt(np.maximum(autovalores_num, 0.0))
    erros_lambda = np.abs(autovalores_num - ref_vals) / ref_vals * 100.0
    erros_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    h_max = max(Lx / Nex, Ly / Ney)
    
    return {
        'Nex': Nex,
        'Ney': Ney,
        'N_nodes': len(nodes),
        'N_elements': len(elements),
        'N_edges_total': len(edges),
        'N_incognitas': len(idx_internos), # DoFs ativos
        'N_fronteira': int(np.sum(is_boundary_edge)),
        'n_nulos_descartados': int(n_nulos),
        'h_max': h_max,
        'autovalores_numericos': autovalores_num,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': erros_lambda,
        'erros_kc_pct': erros_kc,
        'erro_medio_lambda_pct': float(np.mean(erros_lambda)),
        'erro_medio_kc_pct': float(np.mean(erros_kc)),
        'erro_max_kc_pct': float(np.max(erros_kc)),
        'K_red': K_red,
        'M_red': M_red
    }
