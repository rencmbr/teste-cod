import os
import numpy as np

from carregar_malha import carregar_malha
from construir_arvore_busca import construir_arvore_busca
from selecionar_nos_vnmm_2d import selecionar_nos_vnmm_2d


def main():
    nome_arquivo = os.path.join("malhas", "malha.csv")
    
    # 1. Leitura dos dados a partir do arquivo em disco
    coords, vectors = carregar_malha(nome_arquivo)
    
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
