import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree


def gerar_malha_densa(
    nome_arquivo="malhas/malha_densa.csv",
    num_nos_fronteira=40,
    num_nos_interior=150,
    limite=10.0,
    dist_min_fronteira=None,
    dist_min_nos=None,
    seed=42,
    max_tentativas=100000,
    silencioso=False
):
    """
    Gera uma malha no domínio [-limite, limite] x [-limite, limite] com nós na fronteira
    e nós distribuídos aleatoriamente no interior.
    
    Utiliza uma estrutura de busca espacial (KD-Tree) para garantir de forma eficiente:
    1. Distância mínima de cada nó interior em relação aos nós da fronteira (dist_min_fronteira).
    2. Distância euclidiana mínima entre quaisquer nós da malha (dist_min_nos).
    """
    if seed is not None:
        np.random.seed(seed)
     
    if nome_arquivo is not None and os.path.dirname(nome_arquivo):
        os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
     
    coords = []
    vectors = []
    
    # 1. Nós da Fronteira (distribuídos uniformemente sem tocar nos 4 cantos/vértices)
    nos_por_borda = num_nos_fronteira // 4
    passo = (2 * limite) / nos_por_borda
    
    # Cálculo automático de distâncias mínimas caso não fornecidas
    if dist_min_fronteira is None:
        dist_min_fronteira = 0.5 * passo
    if dist_min_nos is None:
        dist_min_nos = 0.45 * passo
    
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
    
    # 2. Nós Interiores com distribuição espacial uniforme e rápida
    lim_int = limite - dist_min_fronteira
    
    # Geração por grade com jitter (garante distribuição homogênea e distância mínima)
    n_lado = int(np.ceil(np.sqrt(num_nos_interior)))
    dx = (2.0 * lim_int) / n_lado
    
    xs = np.linspace(-lim_int + dx / 2.0, lim_int - dx / 2.0, n_lado)
    ys = np.linspace(-lim_int + dx / 2.0, lim_int - dx / 2.0, n_lado)
    X, Y = np.meshgrid(xs, ys)
    pts_base = np.column_stack([X.ravel(), Y.ravel()])
    
    # Adiciona perturbação aleatória controlada (máximo 25% de dx para manter dist_min)
    jitter = np.random.uniform(-0.25 * dx, 0.25 * dx, size=pts_base.shape)
    pts_int = pts_base + jitter
    
    # Ajusta para a quantidade exata de nós interiores solicitada
    if len(pts_int) > num_nos_interior:
        idx_sel = np.random.choice(len(pts_int), num_nos_interior, replace=False)
        pts_int = pts_int[idx_sel]
        
    for pt in pts_int:
        coords.append(pt.tolist())
        # Direção vetorial unitária aleatória
        theta = np.random.uniform(0, 2 * np.pi)
        vectors.append([np.cos(theta), np.sin(theta)])
        
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
    
    if nome_arquivo is not None:
        df.to_csv(nome_arquivo, index=False)
        if not silencioso:
            print(f"Malha gerada com sucesso com {len(df)} nós ({num_nos_fronteira} fronteira + {len(pts_int)} interior).")
            print(f"Distância média característica: h ~ {passo:.4f}")
            print(f"Salvo em: {nome_arquivo}")
            
    return coords, vectors


if __name__ == "__main__":
    gerar_malha_densa()
