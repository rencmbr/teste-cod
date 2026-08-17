import matplotlib.pyplot as plt


def plotar_malha(nodes_coords, nodes_vectors, caminho_saida=None, exibir=True):
    """
    Gera e exibe uma imagem contendo as posições dos nós e suas respectivas direções vetoriais.
    
    Parâmetros:
    - nodes_coords: Array numpy (N, 2) com as coordenadas dos nós (x, y).
    - nodes_vectors: Array numpy (N, 2) com as componentes vetoriais (tx, ty).
    - caminho_saida: Caminho do arquivo para salvar a imagem gerada (ex: 'malhas/malha.png').
    - exibir: Booleano indicando se a visualização deve ser apresentada na tela (padrão: True).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = nodes_coords[:, 0]
    y = nodes_coords[:, 1]
    tx = nodes_vectors[:, 0]
    ty = nodes_vectors[:, 1]
    
    # Plot dos nós
    ax.scatter(x, y, color="blue", s=60, zorder=3, label="Nós")
    
    # Anotação com o índice de cada nó
    for idx, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(
            f"Nó {idx}", 
            (xi, yi), 
            textcoords="offset points", 
            xytext=(7, 7), 
            fontsize=9, 
            fontweight="bold",
            color="black"
        )
    
    # Plot dos vetores de direção
    ax.quiver(
        x, y, tx, ty, 
        angles="xy", 
        scale_units="xy", 
        scale=1.0, 
        color="red", 
        width=0.004, 
        label="Direções Vetoriais (t)", 
        zorder=4
    )
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Visualização da Malha e Direções Vetoriais")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    
    fig.tight_layout()
    
    if caminho_saida:
        plt.savefig(caminho_saida, dpi=300)
        print(f"Imagem da malha salva com sucesso em: {caminho_saida}")
        
    if exibir:
        plt.show()
        
    plt.close(fig)
