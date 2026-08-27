import os
import numpy as np

from carregar_malha import carregar_malha
from plotar_malha import plotar_malha
from construir_arvore_busca import construir_arvore_busca
from avaliar_grade_pontos import avaliar_grade_pontos
from imprimir_detalhes_ponto import imprimir_detalhes_ponto


DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)


def main():
    nome_arquivo = os.path.join(DIRETORIO_RAIZ, "malhas", "malha_densa.csv")
    arquivo_imagem = os.path.splitext(nome_arquivo)[0] + ".png"
    
    # 1. Leitura dos dados a partir do arquivo em disco
    coords, vectors = carregar_malha(nome_arquivo)
    
    # 2. Geração e exibição da imagem dos nós e direções vetoriais
    plotar_malha(coords, vectors, caminho_saida=arquivo_imagem, exibir=False)
    
    # 3. Construção prévia da árvore de busca espacial (fora do loop de avaliação)
    arvore = construir_arvore_busca(coords)
    
    # 4. Definição da distribuição regular de pontos de avaliação P estritamente internos ao domínio (-10, 10) x (-10, 10)
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
    print(f"Arquivo da malha avaliada: {nome_arquivo}")
    print(f"Total de nós da malha: {len(coords)}")
    print(f"Domínio: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
    print(f"Espaçamento da grade: dx = {dx:.2f}, dy = {dy:.2f}")
    print(f"Intervalo dos pontos internos: x in [{x_vals[0]:.2f}, {x_vals[-1]:.2f}], y in [{y_vals[0]:.2f}, {y_vals[-1]:.2f}]")
    print(f"Densidade da grade: {nx} x {ny} = {total_pontos} pontos de avaliação")
    print(f"Tamanho da vizinhança de busca K: {tamanho_vizinhanca}, Tolerância |det(A)|: {tolerancia}\n")
    
    # 5. Avaliação da grade de pontos 
    res = avaliar_grade_pontos(
        coords=coords,
        vectors=vectors,
        arvore=arvore,
        pontos_avaliacao=pontos_avaliacao,
        tolerancia=tolerancia,
        tamanho_vizinhanca=tamanho_vizinhanca,
        adaptativo=True
    )
    
    # 6. Exibição das estatísticas e resultados
    print(f"--- Projeções Nodais do Campo Vetorial ---")
    print(f"Amostra das projeções nodais (primeiros 5 nós): {np.round(res['projecoes_nos'][:5], 4)}\n")
    
    print(f"--- Resumo da Execução e Interpolação ---")
    print(f"Total de pontos avaliados : {res['total_pontos']}")
    print(f"Pontos com sucesso        : {res['sucessos']} ({res['taxa_sucesso']:.1f}%)")
    print(f"Falhas                    : {res['falhas']}")
    
    if res['sucessos'] > 0:
        print(f"\n--- Estatísticas do Determinante |det(A)| e Vizinhança K ---")
        print(f"|det(A)| Mínimo : {res['det_min']:.6f}")
        print(f"|det(A)| Máximo : {res['det_max']:.6f}")
        print(f"|det(A)| Médio  : {res['det_medio']:.6f}")
        print(f"K vizinhos Médio: {res['k_medio']:.2f}")
        print(f"K vizinhos Máx  : {res['k_max']}")
        
        print(f"\n--- Estatísticas do Erro de Interpolação ||E^h - E_exato|| ---")
        print(f"Erro Mínimo : {res['erro_vet_min']:.6e}")
        print(f"Erro Máximo : {res['erro_vet_max']:.6e}")
        print(f"Erro Médio  : {res['erro_vet_medio']:.6e}")
        print(f"Erro RMS    : {res['erro_vet_rms']:.6e}")
        
        print(f"\n--- Estatísticas do Erro do Rotacional |curl(E^h) - curl(E)_exato| ---")
        print(f"Erro Mínimo : {res['erro_rot_min']:.6e}")
        print(f"Erro Máximo : {res['erro_rot_max']:.6e}")
        print(f"Erro Médio  : {res['erro_rot_medio']:.6e}")
        print(f"Erro RMS    : {res['erro_rot_rms']:.6e}")
        
        # 7. Identificação e exibição dos pontos de erro máximo
        resultados_validos = [r for r in res['resultados'] if r['nos'] is not None]
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
