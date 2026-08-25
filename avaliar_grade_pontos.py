import numpy as np

from selecionar_nos_vnmm_2d import selecionar_nos_vnmm_2d
from calcular_funcoes_forma_vnmm_2d import calcular_funcoes_forma_vnmm_2d
from func_vet import func_vet


def avaliar_grade_pontos(
    coords, 
    vectors, 
    arvore, 
    pontos_avaliacao, 
    tolerancia=1.0, 
    tamanho_vizinhanca=8, 
    adaptativo=True
):
    """
    Avalia a interpolação VNMM 2D e o rotacional em uma grade de pontos.
    
    Parâmetros:
    - coords: Array numpy (N, 2) com as coordenadas dos nós da malha.
    - vectors: Array numpy (N, 2) com as direções vetoriais dos nós.
    - arvore: Objeto KDTree construído com as coordenadas dos nós.
    - pontos_avaliacao: Array numpy (M, 2) com as coordenadas dos pontos de avaliação.
    - tolerancia: Tolerância mínima para o determinante |det(A)|.
    - tamanho_vizinhanca: Número inicial de vizinhos mais próximos K.
    - adaptativo: Se True, expande dinamicamente a vizinhança K até satisfazer a tolerância.
    
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
    resultados = []
    
    for idx_ponto, P in enumerate(pontos_avaliacao):
        nos_selecionados, determinante, matriz_a = selecionar_nos_vnmm_2d(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            arvore_busca=arvore,
            K=tamanho_vizinhanca,
            Tol_det=tolerancia,
            adaptativo=adaptativo
        )
        
        E_exato, rot_E_exato = func_vet(P)
        
        if nos_selecionados is not None:
            Phi, rot_Phi, beta = calcular_funcoes_forma_vnmm_2d(
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
            
            resultados.append({
                'id': idx_ponto,
                'P': P,
                'nos': nos_selecionados,
                'det_A': determinante,
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
    
    return {
        'total_pontos': total_pontos,
        'sucessos': sucessos,
        'falhas': falhas,
        'taxa_sucesso': (sucessos / total_pontos) * 100.0 if total_pontos > 0 else 0.0,
        'det_min': float(np.min(dets_arr)),
        'det_max': float(np.max(dets_arr)),
        'det_medio': float(np.mean(dets_arr)),
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
