import os
import numpy as np

from carregar_malha import carregar_malha
from plotar_malha import plotar_malha
from construir_arvore_busca import construir_arvore_busca
from selecionar_nos_vnmm_2d import selecionar_nos_vnmm_2d
from calcular_funcoes_forma_vnmm_2d import calcular_funcoes_forma_vnmm_2d
from func_vet import func_vet
from imprimir_detalhes_ponto import imprimir_detalhes_ponto


def main():
    nome_arquivo = os.path.join("malhas", "malha_densa.csv")
    arquivo_imagem = os.path.splitext(nome_arquivo)[0] + ".png"
    
    # 1. Leitura dos dados a partir do arquivo em disco
    coords, vectors = carregar_malha(nome_arquivo)
    
    # 2. Geração e exibição da imagem dos nós e direções vetoriais
    plotar_malha(coords, vectors, caminho_saida=arquivo_imagem, exibir=False)
    
    # 3. Construção prévia da árvore de busca espacial (fora do loop de avaliação)
    arvore = construir_arvore_busca(coords)
    
    # 4. Cálculo das projeções da função vetorial (func_vet) nas direções de cada nó
    # e_k = E(x_k, y_k) . t_k
    E_nos, rot_E_nos = func_vet(coords)  # Campo vetorial e rotacional exatos nos nós
    projecoes_nos = np.sum(E_nos * vectors, axis=1)  # Produto escalar linha a linha (N,)
    
    print(f"--- Projeções Nodais do Campo Vetorial ---")
    print(f"Total de nós projetados: {len(projecoes_nos)}")
    print(f"Amostra das projeções nodais (primeiros 5 nós): {np.round(projecoes_nos[:5], 4)}\n")
    
    # 5. Definição da distribuição regular de pontos de avaliação P estritamente internos ao domínio (-10, 10) x (-10, 10)
    # Parâmetros de densidade e limites do domínio (parametrizáveis)
    nx = 10  # Número de pontos na direção x
    ny = 10  # Número de pontos na direção y
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    
    tolerancia = 1.0
    tamanho_vizinhanca = 8
    
    # Geração da grade regular de pontos estritamente internos (centros de células)
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_vals = np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx)
    y_vals = np.linspace(y_min + dy / 2.0, y_max - dy / 2.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    pontos_avaliacao = np.column_stack([X.ravel(), Y.ravel()])
    total_pontos = len(pontos_avaliacao)
    
    print(f"--- Avaliação e Interpolação VNMM 2D em Grade Regular (Pontos Internos) ---")
    print(f"Domínio: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
    print(f"Espaçamento da grade: dx = {dx:.2f}, dy = {dy:.2f}")
    print(f"Intervalo dos pontos internos: x in [{x_vals[0]:.2f}, {x_vals[-1]:.2f}], y in [{y_vals[0]:.2f}, {y_vals[-1]:.2f}]")
    print(f"Densidade da grade: {nx} x {ny} = {total_pontos} pontos de avaliação")
    print(f"Tamanho da vizinhança de busca K: {tamanho_vizinhanca}, Tolerância |det(A)|: {tolerancia}\n")
    
    # 6. Loop de seleção de nós, cálculo de funções de forma e interpolação vetorial
    resultados = []
    sucessos = 0
    falhas = 0
    
    for idx_ponto, P in enumerate(pontos_avaliacao):
        nos_selecionados, determinante, matriz_a = selecionar_nos_vnmm_2d(
            P=P, 
            nodes_coords=coords, 
            nodes_vectors=vectors, 
            arvore_busca=arvore,
            K=tamanho_vizinhanca, 
            Tol_det=tolerancia
        )
        
        # Campo e rotacional analíticos exatos no ponto P para comparação
        E_exato, rot_E_exato = func_vet(P)
        
        if nos_selecionados:
            Phi, rot_Phi, beta = calcular_funcoes_forma_vnmm_2d(
                P=P,
                nodes_coords=coords,
                nodes_vectors=vectors,
                nos_selecionados=nos_selecionados,
                matriz_a=matriz_a
            )
            
            # Vetor de projeções dos 3 nós de suporte selecionados: e_s = [e_1, e_2, e_3]^T
            e_s = projecoes_nos[nos_selecionados]
            
            # Interpolação vetorial no ponto P: E^h(P) = Phi(P) * e_s
            E_interpolado = Phi @ e_s
            
            # Rotacional aproximado no ponto P: curl(E^h)(P) = rot_Phi * e_s
            rot_E_interpolado = float(np.dot(rot_Phi, e_s))
            
            erro_vetorial = np.linalg.norm(E_interpolado - E_exato)
            erro_rotacional = abs(rot_E_interpolado - rot_E_exato)
            
            sucessos += 1
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
            
    # 7. Exibição dos resultados e estatísticas da avaliação e interpolação
    print(f"--- Resumo da Execução e Interpolação ---")
    print(f"Total de pontos avaliados : {total_pontos}")
    print(f"Pontos com sucesso        : {sucessos} ({sucessos / total_pontos * 100:.1f}%)")
    print(f"Falhas                    : {falhas}")
    
    if sucessos > 0:
        dets = [r['det_A'] for r in resultados if r['nos'] is not None]
        erros = [r['erro_vetorial'] for r in resultados if r['nos'] is not None]
        erros_rot = [r['erro_rotacional'] for r in resultados if r['nos'] is not None]
        
        print(f"\n--- Estatísticas do Determinante |det(A)| ---")
        print(f"|det(A)| Mínimo : {min(dets):.6f}")
        print(f"|det(A)| Máximo : {max(dets):.6f}")
        print(f"|det(A)| Médio  : {np.mean(dets):.6f}")
        
        print(f"\n--- Estatísticas do Erro de Interpolação ||E^h - E_exato|| ---")
        print(f"Erro Mínimo : {min(erros):.6e}")
        print(f"Erro Máximo : {max(erros):.6e}")
        print(f"Erro Médio  : {np.mean(erros):.6e}")
        print(f"Erro RMS    : {np.sqrt(np.mean(np.array(erros)**2)):.6e}")
        
        print(f"\n--- Estatísticas do Erro do Rotacional |curl(E^h) - curl(E)_exato| ---")
        print(f"Erro Mínimo : {min(erros_rot):.6e}")
        print(f"Erro Máximo : {max(erros_rot):.6e}")
        print(f"Erro Médio  : {np.mean(erros_rot):.6e}")
        print(f"Erro RMS    : {np.sqrt(np.mean(np.array(erros_rot)**2)):.6e}")
        
        # 8. Identificação e exibição dos pontos de erro máximo
        resultados_validos = [r for r in resultados if r['nos'] is not None]
        ponto_max_erro_vet = max(resultados_validos, key=lambda r: r['erro_vetorial'])
        ponto_max_erro_rot = max(resultados_validos, key=lambda r: r['erro_rotacional'])
        
        imprimir_detalhes_ponto(
            ponto_max_erro_vet, 
            f"PONTO COM MÁXIMO ERRO DE INTERPOLAÇÃO DA FUNÇÃO (Erro = {ponto_max_erro_vet['erro_vetorial']:.6e})",
            coords=coords,
            vectors=vectors
        )
        
        imprimir_detalhes_ponto(
            ponto_max_erro_rot, 
            f"PONTO COM MÁXIMO ERRO DE APROXIMAÇÃO DO ROTACIONAL (Erro = {ponto_max_erro_rot['erro_rotacional']:.6e})",
            coords=coords,
            vectors=vectors
        )


if __name__ == "__main__":
    main()
