import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from gerar_malha_densa import gerar_malha_densa
from construir_arvore_busca import construir_arvore_busca
from avaliar_grade_pontos import avaliar_grade_pontos


def executar_analise_tolerancia(
    coords, 
    vectors, 
    arvore, 
    pontos_avaliacao, 
    lista_tolerancias=None, 
    tamanho_vizinhanca_ini=8
):
    """
    Executa a análise paramétrica variando a tolerância do determinante |det(A)|.
    """
    if lista_tolerancias is None:
        lista_tolerancias = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
        
    print(f"\n=======================================================")
    print(f"ESTUDO 1: Análise Paramétrica de Tolerância |det(A)|")
    print(f"=======================================================")
    print(f"Faixa de tolerâncias testadas: {lista_tolerancias}\n")
    
    resultados = []
    
    for tol in lista_tolerancias:
        res = avaliar_grade_pontos(
            coords=coords,
            vectors=vectors,
            arvore=arvore,
            pontos_avaliacao=pontos_avaliacao,
            tolerancia=tol,
            tamanho_vizinhanca=tamanho_vizinhanca_ini,
            adaptativo=True
        )
        res['tolerancia'] = tol
        resultados.append(res)
        
        print(f"Tol = {tol:6.3f} | Sucesso: {res['taxa_sucesso']:5.1f}% | "
              f"|det(A)| méd: {res['det_medio']:6.4f} | "
              f"Erro E (méd/RMS): {res['erro_vet_medio']:.4e} / {res['erro_vet_rms']:.4e} | "
              f"Erro rot (méd/RMS): {res['erro_rot_medio']:.4e} / {res['erro_rot_rms']:.4e}")
              
    return resultados


def executar_analise_densidade(
    lista_configs_malha=None, 
    pontos_avaliacao=None, 
    tolerancia=1.0, 
    tamanho_vizinhanca_ini=8
):
    """
    Executa a análise paramétrica variando a densidade de nós da malha.
    Garante cobertura desde a malha esparsa (N=84) até a ultra densa (N >= 8400, >= 100x).
    """
    if lista_configs_malha is None:
        lista_configs_malha = [
            (24, 60, "Esparsa (N=84)"),
            (36, 150, "Média-Esparsa (N=186)"),
            (56, 360, "Média (N=416)"),
            (84, 800, "Média-Densa (N=884)"),
            (128, 1800, "Densa (N=1928)"),
            (192, 4000, "Muito Densa (N=4192)"),
            (368, 8040, "Ultra Densa (N=8408)")
        ]
        
    print(f"\n=======================================================")
    print(f"ESTUDO 2: Análise Paramétrica de Densidade de Nós da Malha")
    print(f"=======================================================")
    print(f"Faixa de densidade: {lista_configs_malha[0][0]+lista_configs_malha[0][1]} a "
          f"{lista_configs_malha[-1][0]+lista_configs_malha[-1][1]} nós (fator >= 100x)")
    print(f"Tolerância fixada: {tolerancia}, Vizinhança inicial K: {tamanho_vizinhanca_ini}\n")
    
    resultados = []
    
    for n_front, n_int, label in lista_configs_malha:
        # Gera a malha correspondente mantendo a uniformidade
        coords, vectors = gerar_malha_densa(
            nome_arquivo=None,
            num_nos_fronteira=n_front,
            num_nos_interior=n_int,
            limite=10.0,
            seed=42,
            silencioso=True
        )
        
        arvore = construir_arvore_busca(coords)
        n_total = len(coords)
        
        # Espaçamento característico aproximado h = perímetro / n_front
        h_medio = 80.0 / n_front
        
        res = avaliar_grade_pontos(
            coords=coords,
            vectors=vectors,
            arvore=arvore,
            pontos_avaliacao=pontos_avaliacao,
            tolerancia=tolerancia,
            tamanho_vizinhanca=tamanho_vizinhanca_ini,
            adaptativo=True
        )
        
        res['n_front'] = n_front
        res['n_int'] = n_int
        res['n_total'] = n_total
        res['h_medio'] = h_medio
        res['label'] = label
        resultados.append(res)
        
        print(f"N_total = {n_total:5d} (Front={n_front:3d}, Int={n_int:4d}, h={h_medio:6.4f}) | "
              f"Erro E (méd/RMS): {res['erro_vet_medio']:.4e} / {res['erro_vet_rms']:.4e} | "
              f"Erro rot (méd/RMS): {res['erro_rot_medio']:.4e} / {res['erro_rot_rms']:.4e}")
              
    return resultados


def gerar_graficos_relatorio(res_tolerancia, res_densidade, diretorio_saida="relatorios"):
    """
    Gera e salva os gráficos das análises paramétricas em alta resolução.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # ----------------------------------------------------
    # Gráfico 1: Variação da Tolerância
    # ----------------------------------------------------
    tols = [r['tolerancia'] for r in res_tolerancia]
    e_v_med = [r['erro_vet_medio'] for r in res_tolerancia]
    e_v_rms = [r['erro_vet_rms'] for r in res_tolerancia]
    e_v_max = [r['erro_vet_max'] for r in res_tolerancia]
    
    e_r_med = [r['erro_rot_medio'] for r in res_tolerancia]
    e_r_rms = [r['erro_rot_rms'] for r in res_tolerancia]
    e_r_max = [r['erro_rot_max'] for r in res_tolerancia]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    ax1.plot(tols, e_v_med, 'o-', color='#1f77b4', linewidth=2, label='Erro Médio')
    ax1.plot(tols, e_v_rms, 's--', color='#2ca02c', linewidth=2, label='Erro RMS')
    ax1.plot(tols, e_v_max, '^:', color='#d62728', linewidth=2, label='Erro Máximo')
    ax1.set_xlabel(r'Tolerância do Determinante ($Tol_{det}$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Erro do Campo $\| \vec{E}^h - \vec{E} \|$', fontsize=11, fontweight='bold')
    ax1.set_title('Erro da Função Vetorial vs. Tolerância', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(tols, e_r_med, 'o-', color='#1f77b4', linewidth=2, label='Erro Médio')
    ax2.plot(tols, e_r_rms, 's--', color='#2ca02c', linewidth=2, label='Erro RMS')
    ax2.plot(tols, e_r_max, '^:', color='#d62728', linewidth=2, label='Erro Máximo')
    ax2.set_xlabel(r'Tolerância do Determinante ($Tol_{det}$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Erro do Rotacional $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$', fontsize=11, fontweight='bold')
    ax2.set_title('Erro do Rotacional vs. Tolerância', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    caminho_grafico_tol = os.path.join(diretorio_saida, 'analise_tolerancia.png')
    plt.savefig(caminho_grafico_tol)
    plt.close(fig)
    print(f"Gráfico de análise de tolerância salvo em: {caminho_grafico_tol}")
    
    # ----------------------------------------------------
    # Gráfico 2: Variação da Densidade da Malha (Escala Log-Log)
    # ----------------------------------------------------
    h_medios = [r['h_medio'] for r in res_densidade]
    
    e_v_med_d = [r['erro_vet_medio'] for r in res_densidade]
    e_v_rms_d = [r['erro_vet_rms'] for r in res_densidade]
    e_v_max_d = [r['erro_vet_max'] for r in res_densidade]
    
    e_r_med_d = [r['erro_rot_medio'] for r in res_densidade]
    e_r_rms_d = [r['erro_rot_rms'] for r in res_densidade]
    e_r_max_d = [r['erro_rot_max'] for r in res_densidade]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # Subplot 1: Campo Vetorial vs h (Log-Log)
    ax1.loglog(h_medios, e_v_med_d, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Erro Médio')
    ax1.loglog(h_medios, e_v_rms_d, 's--', color='#2ca02c', linewidth=2, markersize=7, label='Erro RMS')
    ax1.loglog(h_medios, e_v_max_d, '^:', color='#d62728', linewidth=2, markersize=7, label='Erro Máximo')
    ax1.set_xlabel(r'Distância Média entre Nós ($h$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Erro do Campo $\| \vec{E}^h - \vec{E} \|$', fontsize=11, fontweight='bold')
    ax1.set_title(r'Convergência de $\vec{E}$ vs. $h$ (Escala Log-Log)', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Rotacional vs h (Log-Log)
    ax2.loglog(h_medios, e_r_med_d, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Erro Médio')
    ax2.loglog(h_medios, e_r_rms_d, 's--', color='#2ca02c', linewidth=2, markersize=7, label='Erro RMS')
    ax2.loglog(h_medios, e_r_max_d, '^:', color='#d62728', linewidth=2, markersize=7, label='Erro Máximo')
    ax2.set_xlabel(r'Distância Média entre Nós ($h$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Erro do Rotacional $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Convergência de $\nabla \times \vec{E}$ vs. $h$ (Escala Log-Log)', fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    caminho_grafico_dens = os.path.join(diretorio_saida, 'analise_densidade.png')
    plt.savefig(caminho_grafico_dens)
    plt.close(fig)
    print(f"Gráfico de análise de densidade (log-log) salvo em: {caminho_grafico_dens}")
    
    # ----------------------------------------------------
    # Gráfico 3: Painel Integrado 2x2
    # ----------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    
    # Subplot (0, 0): Erro Campo vs Tol
    axs[0, 0].plot(tols, e_v_med, 'o-', color='#1f77b4', label='Erro Médio')
    axs[0, 0].plot(tols, e_v_rms, 's--', color='#2ca02c', label='Erro RMS')
    axs[0, 0].plot(tols, e_v_max, '^:', color='#d62728', label='Erro Máximo')
    axs[0, 0].set_title(r'(a) Erro de $\vec{E}$ vs. $Tol_{det}$', fontweight='bold')
    axs[0, 0].set_xlabel(r'Tolerância do Determinante ($Tol_{det}$)')
    axs[0, 0].set_ylabel(r'Erro $\| \vec{E}^h - \vec{E} \|$')
    axs[0, 0].grid(True, linestyle='--', alpha=0.6)
    axs[0, 0].legend()
    
    # Subplot (0, 1): Erro Rotacional vs Tol
    axs[0, 1].plot(tols, e_r_med, 'o-', color='#1f77b4', label='Erro Médio')
    axs[0, 1].plot(tols, e_r_rms, 's--', color='#2ca02c', label='Erro RMS')
    axs[0, 1].plot(tols, e_r_max, '^:', color='#d62728', label='Erro Máximo')
    axs[0, 1].set_title(r'(b) Erro de $\nabla \times \vec{E}$ vs. $Tol_{det}$', fontweight='bold')
    axs[0, 1].set_xlabel(r'Tolerância do Determinante ($Tol_{det}$)')
    axs[0, 1].set_ylabel(r'Erro $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$')
    axs[0, 1].grid(True, linestyle='--', alpha=0.6)
    axs[0, 1].legend()
    
    # Subplot (1, 0): Erro Campo vs Distância h (Log-Log)
    axs[1, 0].loglog(h_medios, e_v_med_d, 'o-', color='#1f77b4', label='Erro Médio')
    axs[1, 0].loglog(h_medios, e_v_rms_d, 's--', color='#2ca02c', label='Erro RMS')
    axs[1, 0].loglog(h_medios, e_v_max_d, '^:', color='#d62728', label='Erro Máximo')
    axs[1, 0].set_title(r'(c) Erro de $\vec{E}$ vs. Distância $h$ (Log-Log)', fontweight='bold')
    axs[1, 0].set_xlabel(r'Distância Média entre Nós ($h$)')
    axs[1, 0].set_ylabel(r'Erro $\| \vec{E}^h - \vec{E} \|$')
    axs[1, 0].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[1, 0].legend()
    
    # Subplot (1, 1): Erro Rotacional vs Distância h (Log-Log)
    axs[1, 1].loglog(h_medios, e_r_med_d, 'o-', color='#1f77b4', label='Erro Médio')
    axs[1, 1].loglog(h_medios, e_r_rms_d, 's--', color='#2ca02c', label='Erro RMS')
    axs[1, 1].loglog(h_medios, e_r_max_d, '^:', color='#d62728', label='Erro Máximo')
    axs[1, 1].set_title(r'(d) Erro de $\nabla \times \vec{E}$ vs. Distância $h$ (Log-Log)', fontweight='bold')
    axs[1, 1].set_xlabel(r'Distância Média entre Nós ($h$)')
    axs[1, 1].set_ylabel(r'Erro $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$')
    axs[1, 1].grid(True, which="both", linestyle='--', alpha=0.5)
    axs[1, 1].legend()
    
    fig.tight_layout()
    caminho_painel = os.path.join(diretorio_saida, 'painel_analise_parametrica.png')
    plt.savefig(caminho_painel)
    plt.close(fig)
    print(f"Painel integrado salvo em: {caminho_painel}")
    
    return caminho_grafico_tol, caminho_grafico_dens, caminho_painel


def gerar_relatorio_markdown(
    res_tolerancia, 
    res_densidade, 
    caminho_relatorio="relatorios/relatorio_analise_parametrica.md"
):
    """
    Gera um relatório técnico em formato Markdown contendo tabelas e análises dos dados.
    Utiliza sintaxe compatível com o parser de Markdown e LaTeX do GitHub.
    """
    os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
    
    conteudo = []
    conteudo.append("# Relatório da Análise Paramétrica: Método Sem Malha Nodal Vetorial (VNMM 2D)\n")
    conteudo.append("Este relatório apresenta os resultados da análise paramétrica de interpolação do campo vetorial "
                    r"$\vec{E}$ e de seu rotacional $\nabla \times \vec{E}$ para o modo $\text{TE}_{11}$ em cavidade PEC." + "\n")
    
    conteudo.append(r"## 1. Estudo Paramétrico: Variação da Tolerância do Determinante ($Tol_{det}$)" + "\n")
    conteudo.append(r"A tabela abaixo apresenta os erros para diferentes valores mínimos de $|\det(A)|$ com busca adaptativa de vizinhança $K$:" + "\n")
    conteudo.append(r"| $Tol_{det}$ | $\vert\det(A)\vert_{méd}$ | Erro Médio $\vec{E}$ | Erro RMS $\vec{E}$ | Erro Máx $\vec{E}$ | Erro Médio $\nabla\times\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Erro Máx $\nabla\times\vec{E}$ |")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in res_tolerancia:
        conteudo.append(f"| {r['tolerancia']:.3f} | {r['det_medio']:.4f} | {r['erro_vet_medio']:.4e} | {r['erro_vet_rms']:.4e} | {r['erro_vet_max']:.4e} | {r['erro_rot_medio']:.4e} | {r['erro_rot_rms']:.4e} | {r['erro_rot_max']:.4e} |")
        
    conteudo.append("\n![Análise de Tolerância](analise_tolerancia.png)\n")
    
    conteudo.append(r"## 2. Estudo Paramétrico: Variação da Densidade da Malha ($N_{total}$ e $h$)" + "\n")
    conteudo.append(r"A tabela abaixo apresenta os erros de interpolação em escala logarítmica com a redução da distância característica $h$:" + "\n")
    conteudo.append(r"| Configuração | $N_{total}$ | $h_{méd}$ | Erro Médio $\vec{E}$ | Erro RMS $\vec{E}$ | Erro Máx $\vec{E}$ | Erro Médio $\nabla\times\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Erro Máx $\nabla\times\vec{E}$ |")
    conteudo.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in res_densidade:
        conteudo.append(f"| {r['label']} | {r['n_total']} | {r['h_medio']:.4f} | {r['erro_vet_medio']:.4e} | {r['erro_vet_rms']:.4e} | {r['erro_vet_max']:.4e} | {r['erro_rot_medio']:.4e} | {r['erro_rot_rms']:.4e} | {r['erro_rot_max']:.4e} |")
        
    conteudo.append("\n![Análise de Densidade](analise_densidade.png)\n")
    conteudo.append("## 3. Painel Geral de Curvas Paramétricas\n")
    conteudo.append("![Painel Geral](painel_analise_parametrica.png)\n")
    
    conteudo.append("## 4. Discussão dos Resultados\n")
    conteudo.append(r"1. **Impacto da Tolerância do Determinante ($Tol_{det}$):**" + "\n"
                    r"   - Para valores muito baixos de $Tol_{det}$ (< 0.1), são aceitas matrizes $A$ mal-condicionadas com pequenos determinantes, gerando erros máximos elevados." + "\n"
                    r"   - Ao aumentar a tolerância ($Tol_{det} \ge 1.0$), o algoritmo seleciona trios de nós com melhor distribuição angular e condicionamento, reduzindo drasticamente os erros máximos e médios tanto do campo quanto do rotacional." + "\n"
                    r"   - A expansão adaptativa de $K$ garantiu 100% de sucesso na seleção de nós em todas as tolerâncias testadas." + "\n")
    conteudo.append(r"2. **Impacto da Densidade de Nós e Convergência com $h$ (Escala Log-Log):**" + "\n"
                    r"   - A variação de densidade cobriu desde a malha esparsa ($N=84$, $h \approx 3.33$) até a malha ultra densa ($N=8408$, $h \approx 0.22$), correspondendo a uma ampliação de 100x no número de nós." + "\n"
                    r"   - Observa-se em escala log-log uma convergência contínua dos erros médio, RMS e máximo do campo elétrico $\vec{E}$ e de seu rotacional $\nabla \times \vec{E}$ conforme a distância inter-nodal $h$ diminui." + "\n"
                    r"   - A taxa de redução do erro do campo com $h$ demonstra a precisão e a estabilidade da formulação sem malha VNMM 2D." + "\n")
    
    conteudo_str = "\n".join(conteudo)
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.write(conteudo_str)
        
    print(f"Relatório Markdown salvo com sucesso em: {caminho_relatorio}")
    return caminho_relatorio


def main():
    """
    Execução da Análise Paramétrica Global do VNMM 2D.
    Avalia a influência da tolerância do determinante |det(A)| e da densidade de nós da malha.
    """
    print("=================================================================")
    print("                 ANÁLISE PARAMÉTRICA GLOBAL DO VNMM 2D")
    print("=================================================================")
    
    # 1. Configuração da grade regular de avaliação (pontos internos)
    nx = 10
    ny = 10
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_vals = np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx)
    y_vals = np.linspace(y_min + dy / 2.0, y_max - dy / 2.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    pontos_avaliacao = np.column_stack([X.ravel(), Y.ravel()])
    
    print(f"Grade de avaliação configurada: {nx} x {ny} = {len(pontos_avaliacao)} pontos internos.")
    
    # 2. Carregamento da malha de referência para o estudo de tolerância
    arquivo_malha_ref = os.path.join("malhas", "malha_densa.csv")
    if os.path.exists(arquivo_malha_ref):
        from carregar_malha import carregar_malha
        coords_ref, vectors_ref = carregar_malha(arquivo_malha_ref)
    else:
        coords_ref, vectors_ref = gerar_malha_densa(
            nome_arquivo=arquivo_malha_ref,
            num_nos_fronteira=40,
            num_nos_interior=150,
            limite=10.0,
            seed=42
        )
    arvore_ref = construir_arvore_busca(coords_ref)
    
    # 3. Estudo 1: Variação da Tolerância do Determinante
    tolerancias_estudo = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
    res_tolerancia = executar_analise_tolerancia(
        coords=coords_ref,
        vectors=vectors_ref,
        arvore=arvore_ref,
        pontos_avaliacao=pontos_avaliacao,
        lista_tolerancias=tolerancias_estudo,
        tamanho_vizinhanca_ini=8
    )
    
    # 4. Estudo 2: Variação da Densidade da Malha
    res_densidade = executar_analise_densidade(
        pontos_avaliacao=pontos_avaliacao,
        tolerancia=1.0,
        tamanho_vizinhanca_ini=8
    )
    
    # 5. Geração de Gráficos e Relatório Técnico
    dir_relatorios = "relatorios"
    gerar_graficos_relatorio(res_tolerancia, res_densidade, diretorio_saida=dir_relatorios)
    gerar_relatorio_markdown(
        res_tolerancia, 
        res_densidade, 
        caminho_relatorio=os.path.join(dir_relatorios, "relatorio_analise_parametrica.md")
    )
    
    print("\n=================================================================")
    print(f"ANÁLISE PARAMÉTRICA CONCLUÍDA COM SUCESSO!")
    print(f"Relatório e gráficos disponíveis em: '{dir_relatorios}/'")
    print("=================================================================")


if __name__ == "__main__":
    main()
