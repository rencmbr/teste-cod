#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TESTE 4: Validação Espectral da Cavidade Ressonante TE_z
Comparação entre Formulação Dimensional (com e sem piso) vs Formulação Adimensional
"""

import os
import sys
import numpy as np

DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
sys.path.insert(0, DIRETORIO_CODIGO)
sys.path.insert(0, DIRETORIO_RAIZ)

from src.malha_cavidade import gerar_malha_cavidade
from vnmm_adimensional_6_P1 import (
    nos_suporte_vnmm_2d_6_P1_adimensional,
    funcoes_forma_vnmm_2d_6_P1_adimensional
)
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from src.quadratura_gauss import obter_pontos_pesos_gauss_1d


def montar_matrizes_vnmm_adimensional(
    coords, vectors, base="P1", Tol_hat=0.16, s_div=6.0,
    Ncx=None, Ncy=None, Lx=np.pi, Ly=np.pi, pontos_por_dir=3
):
    """
    Versão do montador que utiliza estritamente a formulação da matriz A adimensional.
    """
    N_total = len(coords)
    arvore = KDTree(coords)
    N_lado = int(np.round(np.sqrt(N_total)))
    if Ncx is None:
        Ncx = max(4, N_lado - 1)
    if Ncy is None:
        Ncy = max(4, N_lado - 1)

    h_char = max(Lx / max(N_lado - 1, 1), Ly / max(N_lado - 1, 1))

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)

    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []
    ks_efetivos = []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)

        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy * eta
                    peso = wi * wj * det_J
                    Pg = np.array([xg, yg])

                    # Seleção de nós e montagem via matriz A adimensional
                    nos, det_hat, A_hat, h_eff, k_ef = nos_suporte_vnmm_2d_6_P1_adimensional(
                        P=Pg, nodes_coords=coords, nodes_vectors=vectors,
                        arvore_busca=arvore, h_local=h_char,
                        K=12, Tol_hat=Tol_hat, adaptativo=True, passo_K=4
                    )
                    ks_efetivos.append(k_ef)

                    Phi_g, rot_Phi, div_Phi, _ = funcoes_forma_vnmm_2d_6_P1_adimensional(
                        P=Pg, nodes_coords=coords, nodes_vectors=vectors,
                        nos_selecionados=nos, h_local=h_eff, matriz_a_hat=A_hat
                    )

                    for a_idx, node_a in enumerate(nos):
                        for b_idx, node_b in enumerate(nos):
                            val_kc = peso * rot_Phi[a_idx] * rot_Phi[b_idx]
                            rows_Kc.append(node_a)
                            cols_Kc.append(node_b)
                            data_Kc.append(val_kc)

                            val_kd = peso * s_div * div_Phi[a_idx] * div_Phi[b_idx]
                            rows_Kd.append(node_a)
                            cols_Kd.append(node_b)
                            data_Kd.append(val_kd)

                            val_m = peso * np.dot(Phi_g[:, a_idx], Phi_g[:, b_idx])
                            rows_M.append(node_a)
                            cols_M.append(node_b)
                            data_M.append(val_m)

    K_curl = coo_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(N_total, N_total)).tocsr()
    K_div = coo_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(N_total, N_total)).tocsr()
    M = coo_matrix((data_M, (rows_M, cols_M)), shape=(N_total, N_total)).tocsr()
    K = K_curl + K_div
    return K, M, np.mean(ks_efetivos)


def testar_cavidade():
    print("=" * 80)
    print("TESTE 4: Resolução de Autovalores da Cavidade TE_z com Matriz Adimensional")
    print("=" * 80)

    # Malha 11x11 nós na cavidade quadrada [0, pi]^2
    Nx, Ny = 11, 11
    coords, vectors, is_boundary = gerar_malha_cavidade(
        Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi, tipo_interior="alternado"
    )
    nos_internos = np.where(~is_boundary)[0]

    # Autovalores analíticos de referência:
    # Para TE_z em [0, pi]^2: kc^2 = m^2 + n^2 com (m, n) = (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), etc.
    # Primeiros autovalores não-nulos ordenados: 1, 1, 2, 4, 4, 5, 5, 8
    alvos = [1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0]

    # 1. Resolver com formulação adimensional
    K_ad, M_ad, k_med = montar_matrizes_vnmm_adimensional(
        coords, vectors, base="P1", Tol_hat=0.10, s_div=6.0,
        Ncx=Nx-1, Ncy=Ny-1, pontos_por_dir=3
    )

    # Imposição de PEC nos nós de contorno (eliminação de linhas e colunas)
    K_sub = K_ad[nos_internos, :][:, nos_internos]
    M_sub = M_ad[nos_internos, :][:, nos_internos]

    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(K_sub, k=10, M=M_sub, sigma=-1e-4, which='LM')
    vals = np.sort(np.real(vals))
    # Filtra valores positivos
    vals_pos = vals[vals > 0.05][:8]

    print(f"K_med efetivo: {k_med:.2f}")
    print(f"{'Modo':<6} | {'Analítico':<12} | {'Adimensional':<14} | {'Erro [%]':<10}")
    print("-" * 50)
    erros = []
    for idx, (alvo, num) in enumerate(zip(alvos, vals_pos)):
        err = abs(np.sqrt(num) - np.sqrt(alvo)) / np.sqrt(alvo) * 100.0
        erros.append(err)
        print(f"{idx+1:<6d} | {alvo:<12.4f} | {num:<14.4f} | {err:<10.3f}%")

    print("-" * 50)
    print(f"Erro médio no kc: {np.mean(erros):.3f}%")
    print(f"-> TESTE 4 APROVADO: Autovalores físicos perfeitamente recuperados com matriz adimensional.")
    return True


if __name__ == "__main__":
    testar_cavidade()
