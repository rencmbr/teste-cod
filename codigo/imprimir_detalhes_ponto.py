import numpy as np


def imprimir_detalhes_ponto(r, titulo, coords=None, vectors=None):
    """
    Exibe no terminal os detalhes de avaliação e interpolação VNMM 2D de um ponto.
    
    Parâmetros:
    - r: Dicionário contendo os dados do ponto avaliado (id, P, nos, det_A, e_s,
         Phi, E_exato, E_interpolado, erro_vetorial, rot_E_exato, rot_E_interpolado, erro_rotacional).
    - titulo: Título da seção descritiva a ser exibida no cabeçalho.
    - coords: (Opcional) Array com as coordenadas dos nós globais para detalhar os nós de suporte.
    - vectors: (Opcional) Array com as direções vetoriais dos nós globais.
    """
    p_str = f"[{r['P'][0]:.2f}, {r['P'][1]:.2f}]"
    print(f"\n=======================================================")
    print(f"{titulo}")
    print(f"=======================================================")
    print(f"Ponto ID                     : {r['id']}")
    print(f"Coordenadas P                : {p_str}")
    print(f"Nós de suporte (índices)     : {r['nos']}")
    if coords is not None and vectors is not None and r['nos'] is not None:
        print(f"Coordenadas dos nós de suporte:")
        for idx_no in r['nos']:
            print(f"  Nó {idx_no:>3d}: Coords={coords[idx_no]}, Vector={vectors[idx_no]}")
    print(f"Determinante |det(A)|        : {r['det_A']:.6f}")
    if r.get('k_efetivo') is not None:
        print(f"K vizinhos efetivos usados   : {r['k_efetivo']}")
    if r.get('e_s') is not None:
        print(f"Projeções nodais e_s         : {np.round(r['e_s'], 4)}")
    if r.get('Phi') is not None:
        Phi = np.asarray(r['Phi'])
        print(f"Matriz Phi(P) ({Phi.shape[0]}x{Phi.shape[1]}):\n{np.round(Phi, 4)}")
    if r.get('E_exato') is not None:
        print(f"Vetor Exato E(P)             : [{r['E_exato'][0]:.6f}, {r['E_exato'][1]:.6f}]^T")
    if r.get('E_interpolado') is not None:
        print(f"Vetor Interpolado E^h(P)     : [{r['E_interpolado'][0]:.6f}, {r['E_interpolado'][1]:.6f}]^T")
    if r.get('erro_vetorial') is not None:
        print(f"Erro do Campo ||E^h - E||    : {r['erro_vetorial']:.6e}")
    if r.get('rot_E_exato') is not None:
        print(f"Rotacional Exato curl(E)     : {r['rot_E_exato']:.6f} z_hat")
    if r.get('rot_E_interpolado') is not None:
        print(f"Rotacional Aprox curl(E^h)   : {r['rot_E_interpolado']:.6f} z_hat")
    if r.get('erro_rotacional') is not None:
        print(f"Erro do Rotacional           : {r['erro_rotacional']:.6e}")
