import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree


def gerar_malha_densa(
    nome_arquivo="malhas/malha_densa.csv",
    num_nos_fronteira=40,
    num_nos_interior=150,
    limite=10.0,
    dist_min_fronteira=1.0,
    dist_min_nos=0.9,
    seed=42,
    max_tentativas=100000
):
    """
    Gera uma malha no domínio [-limite, limite] x [-limite, limite] com nós na fronteira
    e nós distribuídos aleatoriamente no interior.
    
    Utiliza uma estrutura de busca espacial (KD-Tree) para garantir de forma eficiente:
    1. Distância mínima de cada nó interior em relação aos nós da fronteira (dist_min_fronteira).
    2. Distância euclidiana mínima entre quaisquer nós da malha (dist_min_nos).
    """
    np.random.seed(seed)
     
    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
     
    coords = []
    vectors = []
    
    # 1. Nós da Fronteira (distribuídos uniformemente sem tocar nos 4 cantos/vértices)
    nos_por_borda = num_nos_fronteira // 4
    passo = (2 * limite) / nos_por_borda
    
    # Borda Inferior: y = -limite, x varia no interior do intervalo (tangente: +x)
    for i in range(nos_por_borda):
        x = -limite + (i + 0.5) * passo
        y = -limite
        coords.append([x, y])
        vectors.append([1.0, 0.0])
        
    # Borda Direita: x = +limite, y varia no interior do intervalo (tangente: +y)
    for i in range(nos_por_borda):
        x = limite
        y = -limite + (i + 0.5) * passo
        coords.append([x, y])
        vectors.append([0.0, 1.0])
        
    # Borda Superior: y = +limite, x varia no interior do intervalo (tangente: -x)
    for i in range(nos_por_borda):
        x = limite - (i + 0.5) * passo
        y = limite
        coords.append([x, y])
        vectors.append([-1.0, 0.0])
        
    # Borda Esquerda: x = -limite, y varia no interior do intervalo (tangente: -y)
    for i in range(nos_por_borda):
        x = -limite
        y = limite - (i + 0.5) * passo
        coords.append([x, y])
        vectors.append([0.0, -1.0])
        
    coords_fronteira = np.array(coords)
    arvore_fronteira = KDTree(coords_fronteira)
    
    # Inicializa a KD-Tree global contendo todos os nós já aprovados
    arvore_total = KDTree(np.array(coords))
    
    # 2. Nós Interiores com busca espacial eficiente
    lim_int = limite - dist_min_fronteira
    nos_gerados = 0
    tentativas = 0
    
    while nos_gerados < num_nos_interior and tentativas < max_tentativas:
        tentativas += 1
        
        # Gera ponto candidato no interior
        pt = np.random.uniform(-lim_int, lim_int, size=2)
        
        # 2.1 Verifica distância mínima aos nós da fronteira via KD-Tree
        dist_fronteira, _ = arvore_fronteira.query(pt, k=1)
        if dist_fronteira < dist_min_fronteira:
            continue
            
        # 2.2 Verifica distância mínima aos nós existentes (fronteira e interiores já aceitos)
        dist_vizinho_mais_proximo, _ = arvore_total.query(pt, k=1)
        if dist_vizinho_mais_proximo < dist_min_nos:
            continue
            
        # Ponto aceito
        coords.append(pt.tolist())
        
        # Direção vetorial unitária aleatória
        theta = np.random.uniform(0, 2 * np.pi)
        vectors.append([np.cos(theta), np.sin(theta)])
        nos_gerados += 1
        
        # Atualiza a KD-Tree com o novo conjunto de nós aceitos
        arvore_total = KDTree(np.array(coords))
        
    if nos_gerados < num_nos_interior:
        print(f"Aviso: Atingido o limite de tentativas. {nos_gerados}/{num_nos_interior} nós interiores foram gerados.")
        
    # 3. Criação do DataFrame e exportação CSV
    coords = np.array(coords)
    vectors = np.array(vectors)
    
    df = pd.DataFrame({
        'id': np.arange(len(coords)),
        'x': np.round(coords[:, 0], 4),
        'y': np.round(coords[:, 1], 4),
        'tx': np.round(vectors[:, 0], 4),
        'ty': np.round(vectors[:, 1], 4)
    })
    
    df.to_csv(nome_arquivo, index=False)
    print(f"Malha gerada com sucesso com {len(df)} nós ({num_nos_fronteira} fronteira + {nos_gerados} interior).")
    print(f"Distância mínima entre nós respeitada: >= {dist_min_nos}")
    print(f"Salvo em: {nome_arquivo}")
    return nome_arquivo

if __name__ == "__main__":
    gerar_malha_densa()
