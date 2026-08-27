import os
import numpy as np

from carregar_malha import carregar_malha
from gerar_malha_densa import gerar_malha_densa
from plotar_malha import plotar_malha
from construir_arvore_busca import construir_arvore_busca
from avaliar_grade_pontos_6_P1 import avaliar_grade_pontos_6_P1
from imprimir_detalhes_ponto import imprimir_detalhes_ponto


DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_MALHAS = os.path.join(DIRETORIO_RAIZ, "malhas")


def main():
    print("=================================================================")
    print("      VALIDAÇÃO EM MALHA DENSA: FORMULAÇÃO VNMM 2D (BASE P1 - 6 NÓS)")
    print("=================================================================")
    
    # 1. Configuração e carregamento da malha densa (N = 1928 nós)
    caminho_malha = os.path.join(DIRETORIO_MALHAS, "malha_densa_1928.csv")
    
    if os.path.exists(caminho_malha):
        coords, vectors = carregar_malha(caminho_malha)
    else:
        print("Gerando malha densa (128 contorno + 1800 interior = 1928 nós)...")
        coords, vectors = gerar_malha_densa(
            nome_arquivo=caminho_malha,
            num_nos_fronteira=128,
            num_nos_interior=1800,
            limite=10.0,
            seed=42
        )
        
    arquivo_imagem = os.path.splitext(caminho_malha)[0] + ".png"
    if not os.path.exists(arquivo_imagem):
        plotar_malha(coords, vectors, caminho_saida=arquivo_imagem, exibir=False)
        
    arvore = construir_arvore_busca(coords)
    n_total = len(coords)
    
    # Espaçamento característico h
    h_caracteristico = (2.0 * 10.0) / (128 // 4)  # h = 20 / 32 = 0.625
    
    # 2. Definição da tolerância adaptativa calibrada com a lei quártica Tol_det(h) ~ O(h^4)
    # Referência: Tol_ref = 1.0 para h_ref = 2.0 => Tol_det(h) = 1.0 * (0.625 / 2.0)^4 = 0.0095367
    tol_ref = 1.0
    h_ref = 2.0
    tol_det = tol_ref * (h_caracteristico / h_ref)**4
    tamanho_vizinhanca = 12
    
    # 3. Grade regular de pontos de avaliação (15 x 15 = 225 pontos internos)
    nx, ny = 15, 15
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_vals = np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx)
    y_vals = np.linspace(y_min + dy / 2.0, y_max - dy / 2.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    pontos_avaliacao = np.column_stack([X.ravel(), Y.ravel()])
    total_pontos = len(pontos_avaliacao)
    
    print(f"\n--- Parâmetros do Teste ---")
    print(f"Malha avaliada            : {caminho_malha}")
    print(f"Total de nós da malha (N) : {n_total}")
    print(f"Espaçamento característico: h = {h_caracteristico:.4f}")
    print(f"Grade de avaliação        : {nx} x {ny} = {total_pontos} pontos internos")
    print(f"Tolerância Tol_det(h)     : {tol_det:.6e} (calibrada por Tol_ref * (h/h_ref)^4)")
    print(f"Vizinhança inicial K      : {tamanho_vizinhanca}\n")
    
    # 4. Avaliação na grade de pontos com a base P1
    res = avaliar_grade_pontos_6_P1(
        coords=coords,
        vectors=vectors,
        arvore=arvore,
        pontos_avaliacao=pontos_avaliacao,
        tolerancia=tol_det,
        tamanho_vizinhanca=tamanho_vizinhanca,
        adaptativo=True,
        passo_K=4
    )
    
    # 5. Apresentação dos resultados e métricas
    print(f"=================================================================")
    print(f"               RESULTADOS DA INTERPOLAÇÃO (BASE P1 - 6 NÓS)")
    print(f"=================================================================")
    print(f"Total de pontos avaliados : {res['total_pontos']}")
    print(f"Pontos com sucesso        : {res['sucessos']} ({res['taxa_sucesso']:.1f}%)")
    print(f"Falhas                    : {res['falhas']}")
    
    if res['sucessos'] > 0:
        print(f"\n--- Estatísticas do Determinante |det(A)| e Vizinhança K ---")
        print(f"|det(A)| Mínimo : {res['det_min']:.6e}")
        print(f"|det(A)| Máximo : {res['det_max']:.6e}")
        print(f"|det(A)| Médio  : {res['det_medio']:.6e}")
        print(f"K vizinhos Médio: {res['k_medio']:.2f}")
        print(f"K vizinhos Máx  : {res['k_max']}")
        
        print(f"\n--- Estatísticas do Erro de Interpolação do Campo ||E^h - E_exato|| ---")
        print(f"Erro Mínimo : {res['erro_vet_min']:.6e}")
        print(f"Erro Máximo : {res['erro_vet_max']:.6e}")
        print(f"Erro Médio  : {res['erro_vet_medio']:.6e}")
        print(f"Erro RMS    : {res['erro_vet_rms']:.6e}")
        
        print(f"\n--- Estatísticas do Erro do Rotacional |curl(E^h) - curl(E)_exato| ---")
        print(f"Erro Mínimo : {res['erro_rot_min']:.6e}")
        print(f"Erro Máximo : {res['erro_rot_max']:.6e}")
        print(f"Erro Médio  : {res['erro_rot_medio']:.6e}")
        print(f"Erro RMS    : {res['erro_rot_rms']:.6e}")
        
        # 6. Identificação dos pontos de máximo erro
        resultados_validos = [r for r in res['resultados'] if r['nos'] is not None]
        ponto_max_erro_vet = max(resultados_validos, key=lambda r: r['erro_vetorial'])
        ponto_max_erro_rot = max(resultados_validos, key=lambda r: r['erro_rotacional'])
        
        imprimir_detalhes_ponto(
            ponto_max_erro_vet, 
            f"PONTO COM MÁXIMO ERRO DE INTERPOLAÇÃO DO CAMPO (Erro = {ponto_max_erro_vet['erro_vetorial']:.6e})",
            coords=coords,
            vectors=vectors
        )
        
        imprimir_detalhes_ponto(
            ponto_max_erro_rot, 
            f"PONTO COM MÁXIMO ERRO DE APROXIMAÇÃO DO ROTACIONAL (Erro = {ponto_max_erro_rot['erro_rotacional']:.6e})",
            coords=coords,
            vectors=vectors
        )
        
    print("=================================================================")
    print("Validação da etapa 3 concluída com sucesso!")
    print("=================================================================")
    return res


if __name__ == "__main__":
    main()
