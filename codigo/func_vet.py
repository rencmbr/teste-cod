import numpy as np


def func_vet(P, A=1.0):
    """
    Calcula o campo elétrico E(x, y) e o seu rotacional curl(E) para o modo TE11 (m=1, n=1)
    em uma cavidade PEC com dimensões Lx = 20 e Ly = 20 deslocada para o intervalo [-10, 10] x [-10, 10].
    
    Equações do modo TE11:
        E_x(x, y) =  A * np.cos(pi * (x + 10.0) / 20.0) * np.sin(pi * (y + 10.0) / 20.0)
        E_y(x, y) = -A * np.sin(pi * (x + 10.0) / 20.0) * np.cos(pi * (y + 10.0) / 20.0)
        
    Rotacional analítico (componente z):
        curl(E)_z = dE_y/dx - dE_x/dy
                  = -2 * A * (pi / 20.0) * np.cos(pi * (x + 10.0) / 20.0) * np.cos(pi * (y + 10.0) / 20.0)
                  = -(A * pi / 10.0) * np.cos(pi * (x + 10.0) / 20.0) * np.cos(pi * (y + 10.0) / 20.0)
        
    Parâmetros:
    - P: Coordenadas do ponto 2D [x, y], tupla (x, y) ou array numpy (2,).
         Também suporta array de coordenadas com formato (N, 2).
    - A: Amplitude do campo (padrão: 1.0).
         
    Retorna:
    - E: Array numpy contendo o vetor [Ex, Ey] no ponto P (formato (2,) para ponto único,
         ou (N, 2) caso P seja uma lista/array de pontos).
    - rot_E: Componente z do rotacional analítico no ponto P (float para ponto único,
             ou array de formato (N,) para múltiplos pontos).
    """
    P = np.asarray(P, dtype=float)
    
    if P.ndim == 1:
        x, y = P[0], P[1]
        u = np.pi * (x + 10.0) / 20.0
        v = np.pi * (y + 10.0) / 20.0
        
        Ex =  A * np.cos(u) * np.sin(v)
        Ey = -A * np.sin(u) * np.cos(v)
        rot_E = float(-(A * np.pi / 10.0) * np.cos(u) * np.cos(v))
        
        return np.array([Ex, Ey], dtype=float), rot_E
    elif P.ndim == 2:
        x = P[:, 0]
        y = P[:, 1]
        u = np.pi * (x + 10.0) / 20.0
        v = np.pi * (y + 10.0) / 20.0
        
        Ex =  A * np.cos(u) * np.sin(v)
        Ey = -A * np.sin(u) * np.cos(v)
        rot_E = -(A * np.pi / 10.0) * np.cos(u) * np.cos(v)
        
        return np.column_stack([Ex, Ey]), rot_E
    else:
        raise ValueError(f"Formato de entrada inválido: {P.shape}. Esperado (2,) ou (N, 2).")
