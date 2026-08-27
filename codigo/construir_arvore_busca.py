from scipy.spatial import KDTree

def construir_arvore_busca(nodes_coords):
    """
    Constrói a estrutura espacial de busca (KD-Tree) a partir das coordenadas dos nós.
    
    Parâmetros:
    - nodes_coords: Array numpy com as coordenadas dos nós globais.
    
    Retorna:
    - arvore: Objeto KDTree pronto para consultas espaciais.
    """
    return KDTree(nodes_coords)
