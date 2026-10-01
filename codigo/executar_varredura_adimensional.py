#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executa a varredura paramétrica de Tol_hat entre 0.05 e 6.0
na malha fina perturbada (N = 4192, h = 0.0654 m) sobre Omega = [0, pi]^2
com 1600 pontos de teste, utilizando a formulação puramente adimensional.
"""

import os
import sys
import time
import numpy as np
from scipy.spatial import KDTree

DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
if DIRETORIO_CODIGO not in sys.path:
    sys.path.insert(0, DIRETORIO_CODIGO)

from vnmm_adimensional_6_P1 import (
    nos_suporte_vnmm_2d_6_P1_adimensional,
    funcoes_forma_vnmm_2d_6_P1_adimensional
)
from gerar_figura_convergencia_L1_vs_P1_pi import gerar_malha_nodal_pi, campo_exato_TE11

def main():
    print("=" * 80)
    print("VARREDURA PARAMETRICA DE TOL_HAT ENTRE 0.05 E 6.0 (N = 4192, 1600 PONTOS)")
    print("=" * 80)

    # 1. Gerar malha fina exata
    coords_densa, vectors_densa, h_densa = gerar_malha_nodal_pi(192, 4000, seed=42)
    arvore_densa = KDTree(coords_densa)
    N_total = len(coords_densa)
    print(f"Malha gerada: N = {N_total} nós, h_avg = {h_densa:.4f} m")

    # 2. Projeções nodais analíticas TE11
    E_nos_d = np.zeros_like(coords_densa)
    for i, pt in enumerate(coords_densa):
        E_nos_d[i], _ = campo_exato_TE11(pt)
    proj_densa = np.sum(E_nos_d * vectors_densa, axis=1)

    # 3. Grade de teste de 1600 pontos (40x40) em [0.04*pi, 0.96*pi]^2
    n_pts_lado = 40
    xs = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_pts_lado)
    ys = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_pts_lado)
    X, Y = np.meshgrid(xs, ys)
    pontos_teste = np.column_stack([X.ravel(), Y.ravel()])
    n_pts = len(pontos_teste)
    print(f"Pontos de teste: {n_pts} pontos")

    # Valores limpos e redondos entre 0.05 e 6.0
    tol_hat_vals = [0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 4.50, 6.00]
    
    resultados = []

    for tol_hat in tol_hat_vals:
        t0 = time.time()
        err_E = []
        err_rot = []
        ks = []
        conds_hat = []
        conds_dim = []
        sucessos = 0

        for P in pontos_teste:
            E_ex, rot_ex = campo_exato_TE11(P)
            
            res_sel = nos_suporte_vnmm_2d_6_P1_adimensional(
                P, coords_densa, vectors_densa, arvore_densa,
                h_local=h_densa, K=12, Tol_hat=tol_hat,
                adaptativo=True, passo_K=4, K_max=40,
                retornar_matriz_dimensional=True
            )
            sel, det_hat, A_hat, h_eff, k_ef, A_dim = res_sel

            if sel is not None and len(sel) == 6 and det_hat > 0:
                sucessos += 1
                Phi_hat, rot_Phi_hat, div_Phi_hat, _ = funcoes_forma_vnmm_2d_6_P1_adimensional(
                    P, coords_densa, vectors_densa, sel, h_eff, matriz_a_hat=A_hat
                )
                e_sel = proj_densa[sel]
                E_interp = Phi_hat @ e_sel
                rot_interp = float(np.dot(rot_Phi_hat, e_sel))

                err_E.append(np.linalg.norm(E_interp - E_ex))
                err_rot.append(abs(rot_interp - rot_ex))
                ks.append(k_ef)

                try:
                    c_hat = np.linalg.cond(A_hat)
                except Exception:
                    c_hat = 1e6
                conds_hat.append(c_hat)

                try:
                    c_dim = np.linalg.cond(A_dim)
                except Exception:
                    c_dim = 1e6
                conds_dim.append(c_dim)
            else:
                err_E.append(10.0)
                err_rot.append(50.0)
                ks.append(k_ef if k_ef > 0 else 12)
                conds_hat.append(1e6)
                conds_dim.append(1e6)

        dt = time.time() - t0
        res = {
            'tol_hat': tol_hat,
            'sucesso': (sucessos / n_pts) * 100.0,
            'k_med': float(np.mean(ks)),
            'err_E_rms': float(np.sqrt(np.mean(np.array(err_E)**2))),
            'err_rot_rms': float(np.sqrt(np.mean(np.array(err_rot)**2))),
            'cond_hat_med': float(np.median(conds_hat)),
            'cond_dim_med': float(np.median(conds_dim)),
            'tempo_s': dt
        }
        resultados.append(res)

        print(f"Tol_hat = {tol_hat:5.2f} | Sucesso: {res['sucesso']:5.1f}% | K_med: {res['k_med']:4.1f} | "
              f"Erro E: {res['err_E_rms']:.2e} | Erro Rot: {res['err_rot_rms']:.2e} | "
              f"cond(A_hat): {res['cond_hat_med']:4.1f} | cond(A): {res['cond_dim_med']:4.0f} | Tempo: {dt:.2f}s")

    print("\n" + "=" * 80)
    print("DADOS CONSOLIDADOS PARA A TABELA:")
    print("=" * 80)
    for r in resultados:
        print(f"{r['tol_hat']:5.2f}, {r['sucesso']:5.1f}%, {r['k_med']:4.1f}, "
              f"{r['err_E_rms']:.2e}, {r['err_rot_rms']:.2e}, "
              f"{r['cond_hat_med']:.1f}, {r['cond_dim_med']:.0f}")

if __name__ == "__main__":
    main()
