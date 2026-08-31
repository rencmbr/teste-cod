import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from src.eigen_solver_cavity import resolver_autovalores_cavidade, MODOS_ANALITICOS_REF


def executar_estudo_integracao_caso_base(
    Nx=21, 
    Ny=21, 
    lista_Nc=[10, 15, 20, 30, 40], 
    lista_p=[2, 3, 4, 5],
    s_div=6.0,
    base="P1"
):
    """
    Executa a varredura bidimensional de integração no caso base (Nx=21, Ny=21, 441 nós):
    - Variação do número de células de integração Ncx = Ncy in lista_Nc
    - Variação da ordem de quadratura p in lista_p (p x p pontos de Gauss por célula)
    - Comparação entre modo_suporte='ponto_gauss' e 'centro_celula'
    """
    print("==================================================================================")
    print(f"  ESTUDO DE INTEGRAÇÃO NUMÉRICA NO CASO BASE: CAVIDADE PEC 2D ({Nx}x{Ny} = {Nx*Ny} NÓS)")
    print(f"  Base: {base} | s_div = {s_div:.1f} | Modos: Ponto de Gauss (EFG) vs Centro de Célula")
    print("==================================================================================\n")
    
    resultados_ponto_gauss = []
    resultados_centro_celula = []
    
    # 1. Varredura com modo_suporte = 'ponto_gauss'
    print(">>> 1. Executando varredura com SUPORTE POR PONTO DE GAUSS (Estilo EFG)...")
    for Nc in lista_Nc:
        for p in lista_p:
            t0 = time.time()
            res = resolver_autovalores_cavidade(
                Nx=Nx,
                Ny=Ny,
                Lx=np.pi,
                Ly=np.pi,
                Ncx=Nc,
                Ncy=Nc,
                base=base,
                tipo_interior="alternado",
                num_autovalores=10,
                s_div=s_div,
                pontos_por_dir=p,
                modo_suporte="ponto_gauss"
            )
            t_exec = time.time() - t0
            
            res['Nc'] = Nc
            res['p'] = p
            res['n_celulas'] = Nc * Nc
            res['total_pontos_gauss'] = Nc * Nc * p * p
            res['tempo_seg'] = t_exec
            res['modo_suporte'] = 'ponto_gauss'
            resultados_ponto_gauss.append(res)
            
            print(f"  [Ponto Gauss] Células: {Nc:2d}x{Nc:2d} ({Nc*Nc:4d} cel) | Gauss: {p}x{p} ({p*p:2d} pts/cel) | "
                  f"Total Pts: {Nc*Nc*p*p:5d} | Erro Méd λ: {res['erro_medio_lambda_pct']:5.2f}% | "
                  f"Erro Méd kc: {res['erro_medio_kc_pct']:5.2f}% | Erro Máx kc: {res['erro_max_kc_pct']:5.2f}% | "
                  f"Tempo: {t_exec:.2f}s")
                  
    # 2. Varredura com modo_suporte = 'centro_celula' para referência comparativa
    print("\n>>> 2. Executando varredura com SUPORTE POR CENTRO DE CÉLULA (Referência)...")
    for Nc in lista_Nc:
        for p in lista_p:
            t0 = time.time()
            res = resolver_autovalores_cavidade(
                Nx=Nx,
                Ny=Ny,
                Lx=np.pi,
                Ly=np.pi,
                Ncx=Nc,
                Ncy=Nc,
                base=base,
                tipo_interior="alternado",
                num_autovalores=10,
                s_div=s_div,
                pontos_por_dir=p,
                modo_suporte="centro_celula"
            )
            t_exec = time.time() - t0
            
            res['Nc'] = Nc
            res['p'] = p
            res['n_celulas'] = Nc * Nc
            res['total_pontos_gauss'] = Nc * Nc * p * p
            res['tempo_seg'] = t_exec
            res['modo_suporte'] = 'centro_celula'
            resultados_centro_celula.append(res)
            
            print(f"  [Centro Célula] Células: {Nc:2d}x{Nc:2d} ({Nc*Nc:4d} cel) | Gauss: {p}x{p} ({p*p:2d} pts/cel) | "
                  f"Total Pts: {Nc*Nc*p*p:5d} | Erro Méd λ: {res['erro_medio_lambda_pct']:5.2f}% | "
                  f"Erro Méd kc: {res['erro_medio_kc_pct']:5.2f}% | Erro Máx kc: {res['erro_max_kc_pct']:5.2f}% | "
                  f"Tempo: {t_exec:.2f}s")
                  
    return {
        'Nx': Nx,
        'Ny': Ny,
        'lista_Nc': lista_Nc,
        'lista_p': lista_p,
        'ponto_gauss': resultados_ponto_gauss,
        'centro_celula': resultados_centro_celula
    }


def gerar_graficos_estudo_integracao(dados_estudo, diretorio_saida=DIRETORIO_RELATORIOS):
    """
    Gera gráficos comparativos do estudo de integração:
    1. Mapa de calor / Curvas de Erro Médio de kc vs (Nc, p) para Ponto de Gauss
    2. Comparativo de Erro Médio e Máximo: Ponto de Gauss vs Centro de Célula
    3. Trade-off Erro vs Total de Pontos de Gauss e Tempo Computacional
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    
    lista_Nc = dados_estudo['lista_Nc']
    lista_p = dados_estudo['lista_p']
    res_pg = dados_estudo['ponto_gauss']
    res_cc = dados_estudo['centro_celula']
    
    # ----------------------------------------------------
    # Gráfico 1: Erro Médio de kc vs Número de Células para diferentes ordens p
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    cores = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    marcadores = ['o-', 's--', '^-.', 'd:']
    
    for idx_p, p in enumerate(lista_p):
        sub_pg = [r for r in res_pg if r['p'] == p]
        ncs_pg = [r['Nc'] for r in sub_pg]
        erros_pg = [r['erro_medio_kc_pct'] for r in sub_pg]
        ax1.plot(ncs_pg, erros_pg, marcadores[idx_p], color=cores[idx_p], linewidth=2.0, 
                 label=f"Gauss {p}x{p} ({p*p} pts/célula)")
                 
        sub_cc = [r for r in res_cc if r['p'] == p]
        ncs_cc = [r['Nc'] for r in sub_cc]
        erros_cc = [r['erro_medio_kc_pct'] for r in sub_cc]
        ax2.plot(ncs_cc, erros_cc, marcadores[idx_p], color=cores[idx_p], linewidth=2.0, 
                 label=f"Gauss {p}x{p} ({p*p} pts/célula)")
                 
    ax1.set_xlabel(r"Número de Células por Direção $N_{cx} = N_{cy}$", fontsize=11)
    ax1.set_ylabel(r"Erro Médio do Número de Onda $k_c$ [%]", fontsize=11)
    ax1.set_title("Suporte por PONTO DE GAUSS (Estilo EFG)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)
    
    ax2.set_xlabel(r"Número de Células por Direção $N_{cx} = N_{cy}$", fontsize=11)
    ax2.set_ylabel(r"Erro Médio do Número de Onda $k_c$ [%]", fontsize=11)
    ax2.set_title("Suporte por CENTRO DE CÉLULA (Anterior)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)
    
    fig.suptitle("Impacto da Discretização de Integração no Caso Base (Cavidade 21x21)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    caminho_fig1 = os.path.join(diretorio_saida, "estudo_integracao_erro_kc.png")
    fig.savefig(caminho_fig1, dpi=300)
    plt.close(fig)
    print(f"Gráfico de erro de integração salvo em: {caminho_fig1}")
    
    # ----------------------------------------------------
    # Gráfico 2: Trade-off Erro vs Total de Pontos de Gauss
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    pts_pg = [r['total_pontos_gauss'] for r in res_pg]
    erros_pg = [r['erro_medio_kc_pct'] for r in res_pg]
    
    pts_cc = [r['total_pontos_gauss'] for r in res_cc]
    erros_cc = [r['erro_medio_kc_pct'] for r in res_cc]
    
    ax.scatter(pts_pg, erros_pg, color='#1f77b4', s=70, alpha=0.85, label='Suporte por Ponto de Gauss (EFG)', edgecolors='k')
    ax.scatter(pts_cc, erros_cc, color='#d62728', s=70, alpha=0.85, marker='s', label='Suporte por Centro de Célula', edgecolors='k')
    
    # Destaca os melhores pontos
    idx_min_pg = np.argmin(erros_pg)
    ax.annotate(f"Mínimo Ponto Gauss ({erros_pg[idx_min_pg]:.2f}%)\n{res_pg[idx_min_pg]['Nc']}x{res_pg[idx_min_pg]['Nc']} cel, p={res_pg[idx_min_pg]['p']}",
                xy=(pts_pg[idx_min_pg], erros_pg[idx_min_pg]),
                xytext=(pts_pg[idx_min_pg]*1.2, erros_pg[idx_min_pg]+0.5),
                arrowprops=dict(facecolor='blue', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='blue')
                
    ax.set_xscale('log')
    ax.set_xlabel("Número Total de Pontos de Gauss no Domínio (Escala Log)", fontsize=11)
    ax.set_ylabel(r"Erro Médio do Número de Onda $k_c$ [%]", fontsize=11)
    ax.set_title(r"Trade-off: Precisão Espectral vs Resolução de Quadratura", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    caminho_fig2 = os.path.join(diretorio_saida, "estudo_integracao_tradeoff_pontos.png")
    fig.savefig(caminho_fig2, dpi=300)
    plt.close(fig)
    print(f"Gráfico de trade-off salvo em: {caminho_fig2}")


def gerar_relatorio_markdown_estudo(dados_estudo, diretorio_saida=DIRETORIO_RELATORIOS):
    """
    Gera o relatório técnico detalhado em Markdown com tabelas de todos os testes.
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_relatorio = os.path.join(diretorio_saida, "relatorio_estudo_integracao_caso_base.md")
    
    res_pg = dados_estudo['ponto_gauss']
    res_cc = dados_estudo['centro_celula']
    Nx = dados_estudo['Nx']
    Ny = dados_estudo['Ny']
    
    # Encontra as melhores configurações
    melhor_pg = min(res_pg, key=lambda x: x['erro_medio_kc_pct'])
    melhor_cc = min(res_cc, key=lambda x: x['erro_medio_kc_pct'])
    
    conteudo = []
    conteudo.append(f"# Relatório de Estudo de Integração Numérica no Caso Base ({Nx}x{Ny} = {Nx*Ny} Nós)\n\n")
    conteudo.append("Este documento apresenta a análise comparativa sistemática de esquemas de integração numérica para o "
                    "**Método Sem Malha Nodal Vetorial (VNMM 2D)** na cavidade ressonante PEC bidimensional $[0, \\pi]^2$, "
                    "avaliando o impacto do **suporte individual por ponto de Gauss (estilo EFG)** versus o **suporte por centro de célula**, "
                    "bem como o número de células de integração e a ordem da quadratura de Gauss.\n\n")
                    
    conteudo.append("## 1. Destaques e Melhores Configurações\n\n")
    conteudo.append(f"- **Melhor Configuração [Ponto de Gauss]:** Células ${melhor_pg['Nc']} \\times {melhor_pg['Nc']}$ "
                    f"($N_c = {melhor_pg['n_celulas']}$), Quadratura ${melhor_pg['p']} \\times {melhor_pg['p']}$ "
                    f"({melhor_pg['p']*melhor_pg['p']} pts/célula, total {melhor_pg['total_pontos_gauss']} pts) "
                    f"$\\implies$ **Erro Médio $k_c = {melhor_pg['erro_medio_kc_pct']:.2f}\\%$** (Erro Máx: {melhor_pg['erro_max_kc_pct']:.2f}%, Tempo: {melhor_pg['tempo_seg']:.2f}s)\n")
    conteudo.append(f"- **Melhor Configuração [Centro de Célula]:** Células ${melhor_cc['Nc']} \\times {melhor_cc['Nc']}$ "
                    f"($N_c = {melhor_cc['n_celulas']}$), Quadratura ${melhor_cc['p']} \\times {melhor_cc['p']}$ "
                    f"$\\implies$ **Erro Médio $k_c = {melhor_cc['erro_medio_kc_pct']:.2f}\\%$** (Erro Máx: {melhor_cc['erro_max_kc_pct']:.2f}%, Tempo: {melhor_cc['tempo_seg']:.2f}s)\n\n")
                    
    conteudo.append("![Impacto da Integração no Erro](estudo_integracao_erro_kc.png)\n\n")
    conteudo.append("![Trade-off de Pontos de Gauss](estudo_integracao_tradeoff_pontos.png)\n\n")
    
    conteudo.append("## 2. Tabela de Resultados: Suporte por Ponto de Gauss (Estilo EFG)\n\n")
    conteudo.append("| Células ($N_{cx} \\times N_{cy}$) | Total Células | Gauss ($p \\times p$) | Total Pontos Gauss | Erro Médio $\\lambda$ (%) | Erro Médio $k_c$ (%) | Erro Máx $k_c$ (%) | Tempo (s) |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in res_pg:
        conteudo.append(f"| ${r['Nc']} \\times {r['Nc']}$ | {r['n_celulas']} | ${r['p']} \\times {r['p']}$ ({r['p']*r['p']} pts) | {r['total_pontos_gauss']} | {r['erro_medio_lambda_pct']:5.2f}% | **{r['erro_medio_kc_pct']:5.2f}%** | {r['erro_max_kc_pct']:5.2f}% | {r['tempo_seg']:5.2f}s |\n")
        
    conteudo.append("\n## 3. Tabela de Resultados: Suporte por Centro de Célula (Referência Anterior)\n\n")
    conteudo.append("| Células ($N_{cx} \\times N_{cy}$) | Total Células | Gauss ($p \\times p$) | Total Pontos Gauss | Erro Médio $\\lambda$ (%) | Erro Médio $k_c$ (%) | Erro Máx $k_c$ (%) | Tempo (s) |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in res_cc:
        conteudo.append(f"| ${r['Nc']} \\times {r['Nc']}$ | {r['n_celulas']} | ${r['p']} \\times {r['p']}$ ({r['p']*r['p']} pts) | {r['total_pontos_gauss']} | {r['erro_medio_lambda_pct']:5.2f}% | **{r['erro_medio_kc_pct']:5.2f}%** | {r['erro_max_kc_pct']:5.2f}% | {r['tempo_seg']:5.2f}s |\n")
        
    conteudo.append("\n## 4. Comparação Modal Detalhada na Configuração Ótima de Ponto de Gauss\n\n")
    conteudo.append(f"Configuração: Células ${melhor_pg['Nc']} \\times {melhor_pg['Nc']}$, Gauss ${melhor_pg['p']} \\times {melhor_pg['p']}$:\n\n")
    conteudo.append("| Modo ($TE_{nm}$) | $\\lambda_{analítico}$ | $\\lambda_{VNMM}$ | Erro $\\lambda$ (%) | $k_{c, analítico}$ | $k_{c, VNMM}$ | Erro $k_c$ (%) |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    nomes_modos = ["TE_{10}", "TE_{01}", "TE_{11}", "TE_{20}", "TE_{02}", "TE_{21}", "TE_{12}", "TE_{22}", "TE_{30}", "TE_{03}"]
    for i in range(10):
        m_nome = nomes_modos[i]
        l_ref = melhor_pg['autovalores_analiticos'][i]
        l_num = melhor_pg['autovalores_numericos'][i]
        e_l = melhor_pg['erros_lambda_pct'][i]
        kc_r = melhor_pg['kc_analitico'][i]
        kc_n = melhor_pg['kc_numerico'][i]
        e_kc = melhor_pg['erros_kc_pct'][i]
        conteudo.append(f"| ${m_nome}$ | {l_ref:6.2f} | {l_num:7.4f} | **{e_l:5.2f}%** | {kc_r:6.3f} | {kc_n:6.3f} | **{e_kc:5.2f}%** |\n")
        
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.writelines(conteudo)
        
    print(f"Relatório do estudo salvo em: {caminho_relatorio}")


def main():
    dados = executar_estudo_integracao_caso_base()
    gerar_graficos_estudo_integracao(dados)
    gerar_relatorio_markdown_estudo(dados)
    print("\n>>> Estudo de integração concluído com sucesso!")


if __name__ == "__main__":
    main()
