import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs

from src.malha_cavidade import gerar_malha_cavidade
from src.montador_vnmm import montar_matrizes_vnmm_2d


# Autovalores e números de onda de corte analíticos da Tabela 4-1 (Luilly Ortiz, 2023)
# lambda = n^2 + m^2 (TE_nm com n, m >= 0, (n,m) != (0,0)), kc = sqrt(lambda)
MODOS_ANALITICOS_REF = [
    (1, 0, 1.0, 1.0),            # TE10
    (0, 1, 1.0, 1.0),            # TE01
    (1, 1, 2.0, np.sqrt(2.0)),   # TE11
    (2, 0, 4.0, 2.0),            # TE20
    (0, 2, 4.0, 2.0),            # TE02
    (2, 1, 5.0, np.sqrt(5.0)),   # TE21
    (1, 2, 5.0, np.sqrt(5.0)),   # TE12
    (2, 2, 8.0, np.sqrt(8.0)),   # TE22
    (3, 0, 9.0, 3.0),            # TE30
    (0, 3, 9.0, 3.0),            # TE03
    (3, 1, 10.0, np.sqrt(10.0)), # TE31
    (1, 3, 10.0, np.sqrt(10.0))  # TE13
]


def aplicar_condicao_pec(K, M, is_boundary):
    """
    Aplica a condição de Dirichlet homogênea (paredes PEC) eliminando os graus de liberdade
    dos nós situados na fronteira, onde E_t = c_k = 0.
    
    Retorna as submatrizes reduzidas K_red e M_red, e o array com os índices dos nós internos.
    """
    idx_internos = np.where(~is_boundary)[0]
    
    K_red = K[idx_internos, :][:, idx_internos]
    M_red = M[idx_internos, :][:, idx_internos]
    
    return K_red, M_red, idx_internos


def resolver_problema_autovalores(
    K_red, 
    M_red, 
    num_autovalores=10, 
    sigma=0.5, 
    metodo="eigh"
):
    """
    Resolve o problema generalizado de autovalores:
        K_red * c_red = lambda * M_red * c_red
        
    Retorna os autovalores ordenados crescentemente e os autovetores associados.
    """
    N_int = K_red.shape[0]
    
    if metodo == "eigh" or N_int < 300:
        # Solução densa exata via la.eigh (Cholesky), com fallback para QZ (la.eig) se M_red não for estritamente DP
        try:
            vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
        except (la.LinAlgError, np.linalg.LinAlgError):
            vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
            vals = np.real(vals)
            vecs = np.real(vecs)
    else:
        # Solução esparsa via Shift-and-Invert
        k_busca = min(num_autovalores + 6, N_int - 2)
        try:
            vals, vecs = eigs(
                A=K_red, 
                M=M_red, 
                k=k_busca, 
                sigma=sigma, 
                which='LM'
            )
            vals = np.real(vals)
            vecs = np.real(vecs)
        except Exception:
            vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
            vals = np.real(vals)
            vecs = np.real(vecs)
        
    # Filtra valores positivos não nulos
    mascara_positivos = vals > 1e-4
    vals_validos = vals[mascara_positivos]
    vecs_validos = vecs[:, mascara_positivos]
    
    idx_ordem = np.argsort(vals_validos)
    vals_ordenados = vals_validos[idx_ordem][:num_autovalores]
    vecs_ordenados = vecs_validos[:, idx_ordem][:, :num_autovalores]
    
    return vals_ordenados, vecs_ordenados


def resolver_autovalores_cavidade(
    Nx=21, 
    Ny=21, 
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=None, 
    Ncy=None, 
    base="P1", 
    tipo_interior="alternado", 
    jitter_frac=0.0, 
    num_autovalores=10, 
    sigma=0.5,
    s_div=6.0,
    pontos_por_dir=3,
    tolerancia_det=None,
    modo_suporte="ponto_gauss",
    seed=42
):
    """
    Pipeline completo do solver de autovalores VNMM 2D para cavidade PEC retangular (Tese Luilly Ortiz).
    
    Etapas:
    1. Geração da malha nodal com direções tangentes nas fronteiras PEC.
    2. Montagem esparsa das matrizes K (com regularização div-curl) e M via células de fundo.
    3. Redução do sistema eliminando os nós de fronteira (E_tangente = 0).
    4. Solução do problema generalizado de autovalores.
    5. Comparação e cálculo de erros com relação à Tabela 4-1 da tese de Luilly Ortiz.
    """
    if Ncx is None:
        Ncx = max(4, int(np.round(0.6 * Nx)))
    if Ncy is None:
        Ncy = max(4, int(np.round(0.6 * Ny)))
        
    h_char = max(Lx / max(Nx - 1, 1), Ly / max(Ny - 1, 1))
    h_ref = np.pi / 20.0
    if tolerancia_det is None:
        tolerancia_det = 1e-4 * (h_char / h_ref)**4 if base.upper() in ["P1", "6_P1"] else 1e-4 * (h_char / h_ref)
        
    # 1. Discretização nodal
    coords, vectors, is_boundary = gerar_malha_cavidade(
        Nx=Nx, 
        Ny=Ny, 
        Lx=Lx, 
        Ly=Ly, 
        tipo_interior=tipo_interior, 
        jitter_frac=jitter_frac, 
        seed=seed
    )
    
    # 2. Montagem das matrizes globais K e M
    K, M = montar_matrizes_vnmm_2d(
        coords=coords, 
        vectors=vectors, 
        base=base, 
        tolerancia_det=tolerancia_det,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Lx=Lx,
        Ly=Ly,
        pontos_por_dir=pontos_por_dir,
        modo_suporte=modo_suporte
    )
    
    # 3. Imposição de Dirichlet homogênea (Paredes PEC)
    K_red, M_red, idx_internos = aplicar_condicao_pec(K, M, is_boundary)
    
    # 4. Solução espectral
    autovalores_num, autovetores_red = resolver_problema_autovalores(
        K_red=K_red, 
        M_red=M_red, 
        num_autovalores=num_autovalores, 
        sigma=sigma
    )
    
    # 5. Comparação com a Tabela 4-1
    ref_vals = np.array([item[2] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    ref_kc = np.array([item[3] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    
    kc_num = np.sqrt(autovalores_num)
    
    erros_lambda_pct = np.abs(autovalores_num - ref_vals) / ref_vals * 100.0
    erros_kc_pct = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    # Reconstrução dos autovetores globais
    N_total = len(coords)
    autovetores_globais = np.zeros((N_total, len(autovalores_num)), dtype=float)
    autovetores_globais[idx_internos, :] = autovetores_red
    
    h_max = max(Lx / (Nx - 1), Ly / (Ny - 1))
    
    return {
        'Nx': Nx,
        'Ny': Ny,
        'N_total': N_total,
        'N_internos': len(idx_internos),
        'N_fronteira': int(np.sum(is_boundary)),
        'h_max': h_max,
        'base': base,
        'coords': coords,
        'vectors': vectors,
        'is_boundary': is_boundary,
        'autovalores_numericos': autovalores_num,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': erros_lambda_pct,
        'erros_kc_pct': erros_kc_pct,
        'erro_medio_lambda_pct': float(np.mean(erros_lambda_pct)),
        'erro_max_lambda_pct': float(np.max(erros_lambda_pct)),
        'erro_medio_kc_pct': float(np.mean(erros_kc_pct)),
        'erro_max_kc_pct': float(np.max(erros_kc_pct)),
        'autovetores': autovetores_globais,
        'K_red': K_red,
        'M_red': M_red
    }
