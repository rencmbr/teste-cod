import io
import numpy as np

from carregar_malha import carregar_malha
from construir_arvore_busca import construir_arvore_busca
from selecionar_nos_vnmm_2d import selecionar_nos_vnmm_2d


def main():
    # Simulando a leitura de um arquivo CSV (dados fictícios)
    csv_simulado = """id,x,y,tx,ty
0,0.1,0.2,1.0,0.0
1,0.5,0.1,0.707,-0.707
2,0.2,0.6,0.0,1.0
3,0.8,0.8,-0.707,0.707
4,0.0,0.0,-1.0,0.0
5,1.2,0.5,0.0,-1.0
6,-0.5,0.4,0.866,0.5"""

    arquivo_memoria = io.StringIO(csv_simulado)
    
    # 1. Leitura dos dados
    coords, vectors = carregar_malha(arquivo_memoria)
    
    # 2. Construção prévia da árvore de busca espacial (fora do loop de avaliação)
    arvore = construir_arvore_busca(coords)
    
    # 3. Definição do ponto de avaliação P
    ponto_avaliacao = [0.3, 0.3]
    tolerancia = 1e-3
    tamanho_vizinhanca = 5
    
    # 4. Execução do algoritmo passando a árvore pré-construída
    nos_selecionados, determinante, matriz_a = selecionar_nos_vnmm_2d(
        P=ponto_avaliacao, 
        nodes_coords=coords, 
        nodes_vectors=vectors, 
        arvore_busca=arvore,
        K=tamanho_vizinhanca, 
        Tol_det=tolerancia
    )
    
    # 5. Resultados
    print(f"Ponto de Avaliação: {ponto_avaliacao}")
    if nos_selecionados:
        print(f"Nós de suporte selecionados (Índices Globais): {nos_selecionados}")
        print(f"Determinante |det(A)|: {determinante:.6f}")
        print("Matriz A aprovada:")
        print(np.round(matriz_a, 4))
        
        # Apenas para conferência espacial
        print("\nCoordenadas físicas dos nós escolhidos:")
        for idx in nos_selecionados:
            print(f"Nó {idx}: {coords[idx]}")
    else:
        print(f"Falha: Nenhum trio estável encontrado para a tolerância de {tolerancia} dentro dos {tamanho_vizinhanca} vizinhos.")


if __name__ == "__main__":
    main()
