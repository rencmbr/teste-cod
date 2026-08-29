import numpy as np


def gerar_malha_cavidade(
    Nx=11, 
    Ny=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    tipo_interior="alternado", 
    jitter_frac=0.0, 
    seed=42
):
    """
    Gera uma distribuição nodal regular ou quasi-regular para uma cavidade retangular [0, Lx] x [0, Ly]
    com imposição estrita de direcionalidade tangente nas fronteiras PEC.
    
    Regras de Direcionalidade de Fronteira (Paredes PEC):
    - Arestas horizontais (y=0 e y=Ly): vetor unitário t = [1, 0]^T (tangente ao eixo x).
    - Arestas verticais (x=0 e x=Lx): vetor unitário t = [0, 1]^T (tangente ao eixo y).
    - Cantos: associados à direção tangente horizontal ou vertical.
    
    Parâmetros:
    - Nx: Número de divisões nodais ao longo do eixo x (mínimo 3).
    - Ny: Número de divisões nodais ao longo do eixo y (mínimo 3).
    - Lx: Comprimento da cavidade na direção x (padrão: pi).
    - Ly: Comprimento da cavidade na direção y (padrão: pi).
    - tipo_interior: Tipo de orientação para os nós internos:
        * 'alternado': Alterna direções a 45° e 135° (ou eixos x e y).
        * 'diagonal': Direção fixa a 45° ([1/sqrt(2), 1/sqrt(2)]).
        * 'aleatorio': Ângulo theta uniforme em [0, 2*pi).
        * 'cruzado': Alterna entre [1,0], [0,1], [1,1]/sqrt(2), [-1,1]/sqrt(2).
    - jitter_frac: Fração máxima de deslocamento aleatório para nós internos (0.0 = perfeitamente regular).
    - seed: Semente para reprodutibilidade aleatória.
    
    Retorna:
    - coords: Array numpy (N, 2) com as coordenadas cartesianas (x, y) dos nós.
    - vectors: Array numpy (N, 2) com os vetores unitários de direção associados (tx, ty).
    - is_boundary: Array booleano (N,) indicando True para nós situados na fronteira PEC.
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx e Ny devem ser maiores ou iguais a 3.")
        
    np.random.seed(seed)
    
    x_lin = np.linspace(0.0, Lx, Nx)
    y_lin = np.linspace(0.0, Ly, Ny)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    tol_borda = 1e-7 * min(dx, dy)
    
    coords_list = []
    vectors_list = []
    is_boundary_list = []
    
    for j, y in enumerate(y_lin):
        for i, x in enumerate(x_lin):
            eh_borda_esq = (i == 0)
            eh_borda_dir = (i == Nx - 1)
            eh_borda_inf = (j == 0)
            eh_borda_sup = (j == Ny - 1)
            
            eh_fronteira = eh_borda_esq or eh_borda_dir or eh_borda_inf or eh_borda_sup
            
            # Posição do nó
            if eh_fronteira or jitter_frac == 0.0:
                pos = np.array([x, y], dtype=float)
            else:
                jx = np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                jy = np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
                pos = np.array([x + jx, y + jy], dtype=float)
                
            # Vetor diretor unitário
            if eh_fronteira:
                if eh_borda_inf or eh_borda_sup:
                    # Parede horizontal: tangente é [1, 0]
                    vec = np.array([1.0, 0.0], dtype=float)
                else:
                    # Parede vertical: tangente é [0, 1]
                    vec = np.array([0.0, 1.0], dtype=float)
            else:
                # Nós interiores
                if tipo_interior == "diagonal":
                    vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                elif tipo_interior == "alternado":
                    # Alterna entre 45° e 135°
                    if (i + j) % 2 == 0:
                        vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                    else:
                        vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                elif tipo_interior == "cruzado":
                    idx_mod = (i + j) % 4
                    if idx_mod == 0:
                        vec = np.array([1.0, 0.0], dtype=float)
                    elif idx_mod == 1:
                        vec = np.array([0.0, 1.0], dtype=float)
                    elif idx_mod == 2:
                        vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                    else:
                        vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                elif tipo_interior == "aleatorio":
                    theta = np.random.uniform(0.0, 2.0 * np.pi)
                    vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                else:
                    raise ValueError(f"tipo_interior desconhecido: '{tipo_interior}'")
                    
            coords_list.append(pos)
            vectors_list.append(vec / np.linalg.norm(vec))
            is_boundary_list.append(eh_fronteira)
            
    return np.array(coords_list), np.array(vectors_list), np.array(is_boundary_list, dtype=bool)
