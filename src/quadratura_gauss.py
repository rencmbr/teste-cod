import numpy as np


def obter_pontos_pesos_gauss_1d(n_pontos=2):
    """
    Retorna os pontos e pesos de quadratura de Gauss-Legendre 1D no intervalo canônico [-1, 1].
    """
    if n_pontos == 1:
        return np.array([0.0]), np.array([2.0])
    elif n_pontos == 2:
        val = 1.0 / np.sqrt(3.0)
        return np.array([-val, val]), np.array([1.0, 1.0])
    elif n_pontos == 3:
        val = np.sqrt(3.0 / 5.0)
        return np.array([-val, 0.0, val]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif n_pontos == 4:
        x1 = np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        x2 = np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-x2, -x1, x1, x2]), np.array([w2, w1, w1, w2])
    else:
        # Usa numpy para ordens arbitrárias
        xi, w = np.polynomial.legendre.leggauss(n_pontos)
        return xi, w


def gerar_celulas_quadratura(
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=10, 
    Ncy=10, 
    pontos_por_dir=2
):
    """
    Constrói a grade de células de integração de fundo quadriláteras regulares cobrindo [0, Lx] x [0, Ly]
    e mapeia os pontos e pesos de quadratura de Gauss 2D.
    
    Parâmetros:
    - Lx: Dimensão do domínio em x.
    - Ly: Dimensão do domínio em y.
    - Ncx: Número de células de integração na direção x.
    - Ncy: Número de células de integração na direção y.
    - pontos_por_dir: Número de pontos de Gauss por direção em cada célula (ex: 2 para 2x2 = 4 pontos/célula).
    
    Retorna:
    - pontos_gauss: Array numpy (M, 2) contendo as coordenadas cartesianas (xg, yg) de todos os pontos de Gauss.
    - pesos_gauss: Array numpy (M,) contendo os pesos efetivos de quadratura (w_g * det(J_e)).
    - info_celulas: Lista de dicionários com os limites de cada célula de integração.
    """
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)
    
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    
    pontos_lista = []
    pesos_lista = []
    info_celulas = []
    
    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        y_mid = 0.5 * (y0 + y1)
        
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            x_mid = 0.5 * (x0 + x1)
            
            # Jacobiano da transformação canônica [-1, 1]^2 -> [x0, x1] x [y0, y1]
            # dx_dxi = dx / 2, dy_deta = dy / 2 => det(J) = dx * dy / 4
            det_J = 0.25 * dx * dy
            
            info_celulas.append({
                'cell_id': j * Ncx + i,
                'x_range': (x0, x1),
                'y_range': (y0, y1),
                'area': dx * dy
            })
            
            for wi, xi in zip(w_1d, xi_1d):
                xg = x_mid + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = y_mid + 0.5 * dy * eta
                    
                    peso_efetivo = wi * wj * det_J
                    
                    pontos_lista.append([xg, yg])
                    pesos_lista.append(peso_efetivo)
                    
    return np.array(pontos_lista, dtype=float), np.array(pesos_lista, dtype=float), info_celulas
