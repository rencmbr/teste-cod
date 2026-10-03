#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para Formulação Adimensional do VNMM 2D com Base Completa P1 (6 nós).

Elimina o dilema do piso dimensional de tolerância (Tol_floor) e o descondicionamento
artificial por escala através da normalização das coordenadas locais pelo espaçamento
característico local h_local:
    A = A_hat @ diag(1, 1, h, h, h, h)
    |det(A_hat)| = |det(A)| / h^4 >= Tol_hat > 0

Propriedades:
1. Escala-invariante: Tol_hat é um número adimensional O(1) puro.
2. Não requer calibração ad-hoc com o tamanho do domínio L_domain ou piso dimensional Tol_floor.
3. Previne dilatação espúria do suporte (K_avg >> 6) quando h -> 0.
4. Melhora a estabilidade numérica na inversão matricial, pois cond(A_hat) depende
   estritamente da qualidade geométrica e diversidade angular do estêncil.
"""

import itertools
import numpy as np


def nos_suporte_vnmm_2d_6_P1_adimensional(
    P,
    nodes_coords,
    nodes_vectors,
    arvore_busca,
    h_local=None,
    K=12,
    Tol_hat=1e-2,
    adaptativo=True,
    passo_K=4,
    K_max=None,
    retornar_matriz_dimensional=False
):
    """
    Seleção heurística incremental de 6 nós de suporte baseada na matriz de colocação
    adimensional A_hat.

    Parâmetros:
    - P: Coordenada do ponto de avaliação (x, y).
    - nodes_coords: Array (N, 2) das coordenadas globais dos nós.
    - nodes_vectors: Array (N, 2) dos vetores diretores unitários (tx, ty).
    - arvore_busca: Objeto KDTree com as coordenadas dos nós.
    - h_local: Espaçamento local característico (float). Se None, é calculado
               automaticamente como a distância média aos 6 vizinhos mais próximos.
    - K: Número inicial de vizinhos na busca KDTree (padrão: 12).
    - Tol_hat: Limiar adimensional para o determinante |det(A_hat)| (padrão: 1e-2).
    - adaptativo: Se True, expande K caso nenhum sexteto atinja Tol_hat.
    - passo_K: Incremento de vizinhos K em cada passo de expansão (padrão: 4).
    - K_max: Limite máximo de vizinhos a expandir.
    - retornar_matriz_dimensional: Se True, retorna também a matriz A dimensional.

    Retorna:
    - melhor_sexteto: Lista com 6 índices globais dos nós selecionados.
    - det_A_hat: Determinante da matriz adimensional |det(A_hat)|.
    - A_hat_mat: Matriz de colocação adimensional 6x6.
    - h_eff: Espaçamento h_local efetivo utilizado.
    - k_efetivo: Número efetivo de vizinhos examinados.
    - (Opcional) A_dimensional: Matriz A dimensional se retornar_matriz_dimensional=True.
    """
    P = np.asarray(P, dtype=float)
    N_total = len(nodes_coords)
    limite_K = N_total if K_max is None else min(K_max, N_total)
    K_atual = min(max(K, 6), N_total)

    melhor_sexteto = None
    melhor_det_hat = 0.0
    melhor_A_hat = None
    melhor_k_efetivo = 0

    while True:
        # Fase 1: Recuperação dos K vizinhos mais próximos
        distancias, indices_vizinhos = arvore_busca.query(P, k=K_atual)

        if K_atual == 1:
            indices_vizinhos = [indices_vizinhos]
            distancias = [distancias]
        else:
            indices_vizinhos = list(indices_vizinhos)

        # Definição do h_local se não fornecido externamente
        if h_local is not None and h_local > 0:
            h_eff = float(h_local)
        else:
            # Distância média aos 6 primeiros vizinhos (ou raio do cluster)
            d6 = distancias[:min(6, len(distancias))]
            h_eff = float(np.mean(d6)) if len(d6) > 0 and np.mean(d6) > 1e-14 else 1.0

        inv_h = 1.0 / h_eff

        # Fase 2: Busca combinatória com fixação progressiva de âncora
        for idx_ancora_local in range(K_atual - 5):
            ancora_global = indices_vizinhos[idx_ancora_local]
            indices_restantes = range(idx_ancora_local + 1, K_atual)

            for comb in itertools.combinations(indices_restantes, 5):
                sexteto_candidato = [ancora_global] + [indices_vizinhos[c] for c in comb]
                k_efetivo_candidato = comb[-1] + 1

                # Fase 3: Construção da Matriz de Colocação Adimensional A_hat (6x6)
                coords_locais = nodes_coords[sexteto_candidato] - P
                vecs = nodes_vectors[sexteto_candidato]

                # Coordenadas locais adimensionais: hat_x = dx / h_eff, hat_y = dy / h_eff
                hat_dx = coords_locais[:, 0] * inv_h
                hat_dy = coords_locais[:, 1] * inv_h
                tx = vecs[:, 0]
                ty = vecs[:, 1]

                A_hat = np.empty((6, 6), dtype=float)
                A_hat[:, 0] = tx
                A_hat[:, 1] = ty
                A_hat[:, 2] = hat_dx * tx
                A_hat[:, 3] = hat_dy * tx
                A_hat[:, 4] = hat_dx * ty
                A_hat[:, 5] = hat_dy * ty

                # Fase 4: Avaliação do Determinante Adimensional
                det_hat = float(np.abs(np.linalg.det(A_hat)))

                if det_hat > melhor_det_hat:
                    melhor_det_hat = det_hat
                    melhor_sexteto = sexteto_candidato
                    melhor_A_hat = A_hat
                    melhor_k_efetivo = k_efetivo_candidato

                # Critério de Parada Antecipada Adimensional
                if det_hat >= Tol_hat:
                    if retornar_matriz_dimensional:
                        D_h = np.diag([1.0, 1.0, h_eff, h_eff, h_eff, h_eff])
                        A_dim = A_hat @ D_h
                        return sexteto_candidato, det_hat, A_hat, h_eff, k_efetivo_candidato, A_dim
                    return sexteto_candidato, det_hat, A_hat, h_eff, k_efetivo_candidato

        if not adaptativo or K_atual >= limite_K:
            break

        K_atual = min(K_atual + passo_K, limite_K)

    # Retorno após varredura
    if melhor_sexteto is not None and (melhor_det_hat >= Tol_hat or adaptativo):
        if retornar_matriz_dimensional:
            D_h = np.diag([1.0, 1.0, h_eff, h_eff, h_eff, h_eff])
            A_dim = melhor_A_hat @ D_h if melhor_A_hat is not None else None
            return melhor_sexteto, melhor_det_hat, melhor_A_hat, h_eff, melhor_k_efetivo, A_dim
        return melhor_sexteto, melhor_det_hat, melhor_A_hat, h_eff, melhor_k_efetivo
    else:
        if retornar_matriz_dimensional:
            return None, 0.0, None, h_eff, 0, None
        return None, 0.0, None, h_eff, 0


def funcoes_forma_vnmm_2d_6_P1_adimensional(
    P,
    nodes_coords,
    nodes_vectors,
    nos_selecionados,
    h_local,
    matriz_a_hat=None
):
    """
    Calcula as funções de forma vetoriais N_i, o rotacional (curl N_i)_z e o divergente (div N_i)
    no ponto de avaliação P utilizando a matriz adimensional A_hat.

    Relação matemática:
    - A = A_hat @ diag(1, 1, h, h, h, h)
    - beta = inv(A) = diag(1, 1, 1/h, 1/h, 1/h, 1/h) @ inv(A_hat)
    - No ponto P (dx=0, dy=0):
        N_i(P) = [beta_1i, beta_2i]^T = [hat_beta_1i, hat_beta_2i]^T
    - Rotacional:
        curl(N_i) = (beta_5i - beta_4i) = (1/h) * (hat_beta_5i - hat_beta_4i)
    - Divergente:
        div(N_i) = (beta_3i + beta_6i) = (1/h) * (hat_beta_3i + hat_beta_6i)
    """
    P = np.asarray(P, dtype=float)
    h_eff = float(h_local)
    inv_h = 1.0 / h_eff

    if matriz_a_hat is None:
        coords_locais = nodes_coords[nos_selecionados] - P
        vecs = nodes_vectors[nos_selecionados]

        hat_dx = coords_locais[:, 0] * inv_h
        hat_dy = coords_locais[:, 1] * inv_h
        tx = vecs[:, 0]
        ty = vecs[:, 1]

        A_hat = np.empty((6, 6), dtype=float)
        A_hat[:, 0] = tx
        A_hat[:, 1] = ty
        A_hat[:, 2] = hat_dx * tx
        A_hat[:, 3] = hat_dy * tx
        A_hat[:, 4] = hat_dx * ty
        A_hat[:, 5] = hat_dy * ty
    else:
        A_hat = np.asarray(matriz_a_hat, dtype=float)

    # Inversão numericamente estável da matriz adimensional
    beta_hat = np.linalg.inv(A_hat)

    # Funções de forma no ponto de avaliação P: exatamente beta_hat[0:2, :]
    Phi = beta_hat[0:2, :]  # shape (2, 6)

    # Derivadas espaciais escaladas por 1 / h_local:
    rot_Phi = (beta_hat[4, :] - beta_hat[3, :]) * inv_h  # shape (6,)
    div_Phi = (beta_hat[2, :] + beta_hat[5, :]) * inv_h  # shape (6,)

    return Phi, rot_Phi, div_Phi, beta_hat
