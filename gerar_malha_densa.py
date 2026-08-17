import numpy as np
import pandas as pd
import os

def gerar_malha_densa(
    nome_arquivo="malhas/malha_densa.csv",
    num_nos_fronteira=40,
    num_nos_interior=120,
    limite=10.0,
    dist_min_fronteira=1.0,
    seed=42
):
    """
    Gera uma malha no domínio [-limite, limite] x [-limite, limite] com nós na fronteira
    e nós distribuídos aleatoriamente no interior respeitando a distância mínima da fronteira.
    """
    np.random.seed(seed)
    
    os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
    
    coords = []
    vectors = []
    
    # 1. Nós da Fronteira (40 nós igualmente espaçados no perímetro 4 * 2 * limite = 80)
    # Perímetro total = 80 -> espaçamento = 80 / 40 = 2.0 (10 nós por borda)
    nos_por_borda = num_nos_fronteira // 4
    passo = (2 * limite) / nos_por_borda
    
    # Borda Inferior: y = -limite, x varia de -limite a limite (tangente: +x)
    for i in range(nos_por_borda):
        x = -limite + i * passo
        y = -limite
        coords.append([x, y])
        vectors.append([1.0, 0.0])
        
    # Borda Direita: x = +limite, y varia de -limite a limite (tangente: +y)
    for i in range(nos_por_borda):
        x = limite
        y = -limite + i * passo
        coords.append([x, y])
        vectors.append([0.0, 1.0])
        
    # Borda Superior: y = +limite, x varia de +limite a -limite (tangente: -x)
    for i in range(nos_por_borda):
        x = limite - i * passo
        y = limite
        coords.append([x, y])
        vectors.append([-1.0, 0.0])
        
    # Borda Esquerda: x = -limite, y varia de +limite a -limite (tangente: -y)
    for i in range(nos_por_borda):
        x = -limite
        y = limite - i * passo
        coords.append([x, y])
        vectors.append([0.0, -1.0])
        
    coords_fronteira = np.array(coords)
    
    # 2. Nós Interiores: coordenadas aleatórias com distância >= dist_min_fronteira dos nós da fronteira
    lim_int = limite - dist_min_fronteira
    nos_gerados = 0
    while nos_gerados < num_nos_interior:
        # Gera ponto candidato no interior
        pt = np.random.uniform(-lim_int, lim_int, size=2)
        
        # Verifica a distância euclidiana mínima aos nós da fronteira
        dists = np.linalg.norm(coords_fronteira - pt, axis=1)
        if np.all(dists >= dist_min_fronteira):
            coords.append(pt.tolist())
            
            # Direção vetorial unitária aleatória
            theta = np.random.uniform(0, 2 * np.pi)
            vectors.append([np.cos(theta), np.sin(theta)])
            nos_gerados += 1
            
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
    print(f"Malha gerada com sucesso com {len(df)} nós ({num_nos_fronteira} fronteira + {num_nos_interior} interior).")
    print(f"Salvo em: {nome_arquivo}")
    return nome_arquivo

if __name__ == "__main__":
    gerar_malha_densa()
