import numpy as np

from nos_suporte_vnmm_2d_6_P1 import nos_suporte_vnmm_2d_6_P1
from funcoes_forma_vnmm_2d_6_P1 import funcoes_forma_vnmm_2d_6_P1
from func_vet import func_vet


def avaliar_grade_pontos_6_P1(
    coords, 
    vectors, 
    arvore, 
    pontos_avaliacao, 
    tolerancia=1e-3, 
    tamanho_vizinhanca=12, 
    adaptativo=True,
    passo_K=4,
    K_max=None
):
    """
    Avalia a interpolação VNMM 2D e o rotacional em uma grade de pontos utilizando
    a base linear completa P1 com 6 nós de suporte.
    
    Parâmetros:
    - coords: Array numpy (N, 2) com as coordenadas dos nós da malha.
    - vectors: Array numpy (N, 2) com as direções vetoriais dos nós.
    - arvore: Objeto KDTree construído com as coordenadas dos nós.
    - pontos_avaliacao: Array numpy (M, 2) com as coordenadas dos pontos de avaliação.
    - tolerancia: Tolerância mínima para o determinante |det(A)| (escala O(h^4)).
    - tamanho_vizinhanca: Número inicial de vizinhos mais próximos K (padrão: 12).
    - adaptativo: Se True, expande dinamicamente a vizinhança K até satisfazer a tolerância.
    - passo_K: Incremento de vizinhos K em cada expansão adaptativa.
    - K_max: Limite superior para a expansão de K.
    
    Retorna:
    - dict contendo:
        - estatísticas consolidadas (erros mín/máx/méd/RMS de E e rot(E), det(A) mín/máx/méd, taxa de sucesso, etc.)
        - 'resultados': lista de dicionários com todos os dados individuais calculados em cada ponto.
        - 'projecoes_nos': projeções do campo vetorial exato nas direções dos nós.
    """
    E_nos, rot_E_nos = func_vet(coords)
    projecoes_nos = np.sum(E_nos * vectors, axis=1)
    
    total_pontos = len(pontos_avaliacao)
    sucessos = 0
    falhas = 0
    
    dets = []
    erros_vet = []
    erros_rot = []
    ks_efetivos = []
    resultados = []
    
    for idx_ponto, P in enumerate(pontos_avaliacao):
        nos_selecionados, determinante, matriz_a, k_efetivo = nos_suporte_vnmm_2d_6_P1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            arvore_busca=arvore,
            K=tamanho_vizinhanca,
            Tol_det=tolerancia,
            adaptativo=adaptativo,
            passo_K=passo_K,
            K_max=K_max
        )
        
        E_exato, rot_E_exato = func_vet(P)
        
        if nos_selecionados is not None:
            Phi, rot_Phi, beta = funcoes_forma_vnmm_2d_6_P1(
                P=P,
                nodes_coords=coords,
                nodes_vectors=vectors,
                nos_selecionados=nos_selecionados,
                matriz_a=matriz_a
            )
            
            e_s = projecoes_nos[nos_selecionados]
            E_interpolado = Phi @ e_s
            rot_E_interpolado = float(np.dot(rot_Phi, e_s))
            
            erro_vetorial = float(np.linalg.norm(E_interpolado - E_exato))
            erro_rotacional = float(abs(rot_E_interpolado - rot_E_exato))
            
            sucessos += 1
            dets.append(determinante)
            erros_vet.append(erro_vetorial)
            erros_rot.append(erro_rotacional)
            ks_efetivos.append(k_efetivo)
            
            resultados.append({
                'id': idx_ponto,
                'P': P,
                'nos': nos_selecionados,
                'det_A': determinante,
                'k_efetivo': k_efetivo,
                'e_s': e_s,
                'E_interpolado': E_interpolado,
                'rot_E_interpolado': rot_E_interpolado,
                'E_exato': E_exato,
                'rot_E_exato': rot_E_exato,
                'erro_vetorial': erro_vetorial,
                'erro_rotacional': erro_rotacional,
                'Phi': Phi,
                'rot_Phi': rot_Phi,
                'beta': beta
            })
        else:
            falhas += 1
            resultados.append({
                'id': idx_ponto,
                'P': P,
                'nos': None,
                'det_A': 0.0,
                'k_efetivo': 0,
                'e_s': None,
                'E_interpolado': None,
                'rot_E_interpolado': None,
                'E_exato': E_exato,
                'rot_E_exato': rot_E_exato,
                'erro_vetorial': None,
                'erro_rotacional': None,
                'Phi': None,
                'rot_Phi': None,
                'beta': None
            })
            
    erros_vet_arr = np.array(erros_vet) if erros_vet else np.array([0.0])
    erros_rot_arr = np.array(erros_rot) if erros_rot else np.array([0.0])
    dets_arr = np.array(dets) if dets else np.array([0.0])
    ks_arr = np.array(ks_efetivos) if ks_efetivos else np.array([0])
    
    return {
        'total_pontos': total_pontos,
        'sucessos': sucessos,
        'falhas': falhas,
        'taxa_sucesso': (sucessos / total_pontos) * 100.0 if total_pontos > 0 else 0.0,
        'det_min': float(np.min(dets_arr)),
        'det_max': float(np.max(dets_arr)),
        'det_medio': float(np.mean(dets_arr)),
        'k_medio': float(np.mean(ks_arr)),
        'k_max': int(np.max(ks_arr)),
        'erro_vet_min': float(np.min(erros_vet_arr)),
        'erro_vet_max': float(np.max(erros_vet_arr)),
        'erro_vet_medio': float(np.mean(erros_vet_arr)),
        'erro_vet_rms': float(np.sqrt(np.mean(erros_vet_arr**2))),
        'erro_rot_min': float(np.min(erros_rot_arr)),
        'erro_rot_max': float(np.max(erros_rot_arr)),
        'erro_rot_medio': float(np.mean(erros_rot_arr)),
        'erro_rot_rms': float(np.sqrt(np.mean(erros_rot_arr**2))),
        'resultados': resultados,
        'projecoes_nos': projecoes_nos
    }
