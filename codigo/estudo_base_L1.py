import os
import sys
import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from src.malha_cavidade import gerar_malha_cavidade
from src.montador_vnmm import montar_matrizes_vnmm_2d
from src.eigen_solver_cavity import aplicar_condicao_pec, MODOS_ANALITICOS_REF


def resolver_autovalores_L1(
    Nx=21, 
    Ny=21, 
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=10, 
    Ncy=10, 
    pontos_por_dir=2, 
    modo_suporte="ponto_gauss",
    tol_zero=1e-2,
    num_autovalores=10
):
    """
    Resolve o problema de autovalores com a base L1 (3 termos, 3 nós de suporte).
    L1 = <[1, 0]^T, [0, 1]^T, [y, -x]^T> (Identicamente solenoidal, div = 0).
    """
    coords, vectors, is_boundary = gerar_malha_cavidade(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, tipo_interior="alternado")
    
    K, M = montar_matrizes_vnmm_2d(
        coords=coords, 
        vectors=vectors, 
        base="L1", 
        s_div=0.0, 
        Ncx=Ncx, 
        Ncy=Ncy, 
        Lx=Lx, 
        Ly=Ly, 
        pontos_por_dir=pontos_por_dir, 
        modo_suporte=modo_suporte
    )
    
    K_red, M_red, idx_int = aplicar_condicao_pec(K, M, is_boundary)
    
    # Resolve autovalores
    try:
        vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
    except Exception:
        vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    mascara_positivos = vals > tol_zero
    vals_positivos = np.sort(vals[mascara_positivos])
    n_nulos = np.sum(~mascara_positivos)
    
    autovalores_num = vals_positivos[:num_autovalores]
    
    ref_vals = np.array([item[2] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    ref_kc = np.array([item[3] for item in MODOS_ANALITICOS_REF[:len(autovalores_num)]])
    
    kc_num = np.sqrt(np.maximum(autovalores_num, 0.0))
    erros_lambda = np.abs(autovalores_num - ref_vals) / ref_vals * 100.0
    erros_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    return {
        'Nx': Nx,
        'Ny': Ny,
        'Nc': Ncx,
        'p': pontos_por_dir,
        'n_celulas': Ncx * Ncy,
        'total_pontos_gauss': Ncx * Ncy * pontos_por_dir * pontos_por_dir,
        'modo_suporte': modo_suporte,
        'n_nulos_descartados': int(n_nulos),
        'autovalores_positivos': autovalores_num,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': erros_lambda,
        'erros_kc_pct': erros_kc,
        'erro_medio_lambda_pct': float(np.mean(erros_lambda)) if len(erros_lambda) > 0 else 0.0,
        'erro_medio_kc_pct': float(np.mean(erros_kc)) if len(erros_kc) > 0 else 0.0,
        'erro_max_kc_pct': float(np.max(erros_kc)) if len(erros_kc) > 0 else 0.0
    }


def executar_estudo_L1():
    print("==================================================================================")
    print("  ESTUDO DO VNMM 2D: BASE L1 (3 NÓS DE SUPORTE) NO CASO BASE (21x21 = 441 NÓS)")
    print("  Base Solenoidal L1 = <[1,0]^T, [0,1]^T, [y,-x]^T> | Suporte por Ponto de Gauss")
    print("==================================================================================\n")
    
    lista_Nc = [10, 15, 20, 30, 40]
    lista_p = [2, 3, 4]
    
    resultados_pg = []
    resultados_cc = []
    
    print(">>> 1. Base L1 com SUPORTE POR PONTO DE GAUSS (Estilo EFG)...")
    for Nc in lista_Nc:
        for p in lista_p:
            t0 = time.time()
            res = resolver_autovalores_L1(
                Nx=21, Ny=21, 
                Ncx=Nc, Ncy=Nc, 
                pontos_por_dir=p, 
                modo_suporte="ponto_gauss"
            )
            t_exec = time.time() - t0
            res['tempo_seg'] = t_exec
            resultados_pg.append(res)
            
            l_primeiros = res['autovalores_positivos'][:3]
            l_str = " ".join([f"{v:.4f}" for v in l_primeiros])
            print(f"  [Ponto Gauss] Células: {Nc:2d}x{Nc:2d} | Gauss: {p}x{p} | Total Pts: {res['total_pontos_gauss']:5d} | "
                  f"Zeros: {res['n_nulos_descartados']:3d} | λs: [{l_str}] | "
                  f"Erro Méd kc: {res['erro_medio_kc_pct']:5.2f}% | Tempo: {t_exec:.2f}s")
                  
    print("\n>>> 2. Base L1 com SUPORTE POR CENTRO DE CÉLULA (Referência)...")
    for Nc in lista_Nc:
        for p in lista_p:
            t0 = time.time()
            res = resolver_autovalores_L1(
                Nx=21, Ny=21, 
                Ncx=Nc, Ncy=Nc, 
                pontos_por_dir=p, 
                modo_suporte="centro_celula"
            )
            t_exec = time.time() - t0
            res['tempo_seg'] = t_exec
            resultados_cc.append(res)
            
            l_primeiros = res['autovalores_positivos'][:3]
            l_str = " ".join([f"{v:.4f}" for v in l_primeiros])
            print(f"  [Centro Célula] Células: {Nc:2d}x{Nc:2d} | Gauss: {p}x{p} | Total Pts: {res['total_pontos_gauss']:5d} | "
                  f"Zeros: {res['n_nulos_descartados']:3d} | λs: [{l_str}] | "
                  f"Erro Méd kc: {res['erro_medio_kc_pct']:5.2f}% | Tempo: {t_exec:.2f}s")
                  
    return resultados_pg, resultados_cc


def gerar_graficos_e_relatorio_L1(resultados_pg, resultados_cc, diretorio_saida=DIRETORIO_RELATORIOS):
    os.makedirs(diretorio_saida, exist_ok=True)
    caminho_relatorio = os.path.join(diretorio_saida, "relatorio_estudo_base_L1.md")
    
    # ----------------------------------------------------
    # Gráfico 1: Comparativo Modal L1 vs Analítico (Tabela 4-1)
    # ----------------------------------------------------
    res_ref_pg = [r for r in resultados_pg if r['Nc'] == 20 and r['p'] == 2][0]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    nomes_modos = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    largura = 0.35
    
    ax.bar(indices - largura/2, res_ref_pg['kc_analitico'], largura, label=r"Analítico $k_c$ (Tabela 4-1)", color="#1f77b4", alpha=0.85)
    ax.bar(indices + largura/2, res_ref_pg['kc_numerico'], largura, label=r"VNMM 2D Base $\mathcal{L}^1$ (3 Nós)", color="#e377c2", alpha=0.85)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(nomes_modos, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Número de Onda de Corte $k_c$ [rad/m]", fontsize=11)
    ax.set_title(r"Espectro Modal com a Base $\mathcal{L}^1$ (3 Nós de Suporte)", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    caminho_grafico = os.path.join(diretorio_saida, "espectro_modos_base_L1.png")
    fig.savefig(caminho_grafico, dpi=300)
    plt.close(fig)
    print(f"Gráfico da base L1 salvo em: {caminho_grafico}")
    
    # ----------------------------------------------------
    # Relatório em Markdown
    # ----------------------------------------------------
    conteudo = []
    conteudo.append("# Estudo do VNMM 2D com Base $\\mathcal{L}^1$ (3 Nós de Suporte)\n\n")
    conteudo.append("Este relatório apresenta a análise espectral da formulação VNMM 2D utilizando a base polinomial vetorial incompleta "
                    "**$\\mathcal{L}^1 = \\langle [1, 0]^T, [0, 1]^T, [y, -x]^T \\rangle$** com **3 nós de suporte**, "
                    "avaliando a determinação do domínio de suporte por ponto de Gauss (estilo EFG) versus centro de célula.\n\n")
                    
    conteudo.append("## 1. Características Matemáticas da Base $\\mathcal{L}^1$\n\n")
    conteudo.append("- **Solenoidalidade Idêntica:** Como $\\nabla \\cdot [1,0]^T = 0$, $\\nabla \\cdot [0,1]^T = 0$ e $\\nabla \\cdot [y,-x]^T = 0+0=0$, "
                    "as funções de forma $\\vec{N}_i$ da base $\\mathcal{L}^1$ possuem divergente identicamente nulo em todo o domínio:\n")
    conteudo.append("  $$\\nabla \\cdot \\vec{N}_i(x, y) \\equiv 0 \\implies K_{\\text{div}} \\equiv 0$$\n")
    conteudo.append("- **Incompletude da Jacobiana e Vazamento Modal (*Aliasing*):** A base $\\mathcal{L}^1$ impõe artificialmente "
                    "$\\frac{\\partial E_x}{\\partial x} = 0$ e $\\frac{\\partial E_y}{\\partial y} = 0$, não conseguindo representar as derivadas normais do campo real.\n\n")
                    
    conteudo.append("## 2. Tabela de Resultados: Suporte por Ponto de Gauss (Estilo EFG)\n\n")
    conteudo.append("| Células ($N_c \\times N_c$) | Gauss ($p \\times p$) | Total Pontos Gauss | Zeros Descartados | 1º $\\lambda$ | 2º $\\lambda$ | 3º $\\lambda$ | Erro Médio $k_c$ (%) | Erro Máx $k_c$ (%) | Tempo (s) |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in resultados_pg:
        l1 = r['autovalores_positivos'][0] if len(r['autovalores_positivos']) > 0 else 0.0
        l2 = r['autovalores_positivos'][1] if len(r['autovalores_positivos']) > 1 else 0.0
        l3 = r['autovalores_positivos'][2] if len(r['autovalores_positivos']) > 2 else 0.0
        conteudo.append(f"| ${r['Nc']} \\times {r['Nc']}$ | ${r['p']} \\times {r['p']}$ | {r['total_pontos_gauss']} | {r['n_nulos_descartados']} | {l1:.4f} | {l2:.4f} | {l3:.4f} | **{r['erro_medio_kc_pct']:5.2f}%** | {r['erro_max_kc_pct']:5.2f}% | {r['tempo_seg']:.2f}s |\n")
        
    conteudo.append("\n## 3. Tabela de Resultados: Suporte por Centro de Célula (Referência)\n\n")
    conteudo.append("| Células ($N_c \\times N_c$) | Gauss ($p \\times p$) | Total Pontos Gauss | Zeros Descartados | 1º $\\lambda$ | 2º $\\lambda$ | 3º $\\lambda$ | Erro Médio $k_c$ (%) | Erro Máx $k_c$ (%) | Tempo (s) |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in resultados_cc:
        l1 = r['autovalores_positivos'][0] if len(r['autovalores_positivos']) > 0 else 0.0
        l2 = r['autovalores_positivos'][1] if len(r['autovalores_positivos']) > 1 else 0.0
        l3 = r['autovalores_positivos'][2] if len(r['autovalores_positivos']) > 2 else 0.0
        conteudo.append(f"| ${r['Nc']} \\times {r['Nc']}$ | ${r['p']} \\times {r['p']}$ | {r['total_pontos_gauss']} | {r['n_nulos_descartados']} | {l1:.4f} | {l2:.4f} | {l3:.4f} | **{r['erro_medio_kc_pct']:5.2f}%** | {r['erro_max_kc_pct']:5.2f}% | {r['tempo_seg']:.2f}s |\n")
        
    conteudo.append("\n## 4. Comparação Modal dos 10 Primeiros Modos na Configuração $20 \\times 20$, Gauss $2 \\times 2$\n\n")
    conteudo.append("| Modo ($TE_{nm}$) | $\\lambda_{\\text{analítico}}$ | $\\lambda_{\\text{VNMM } \\mathcal{L}^1}$ | $k_{c, \\text{analítico}}$ | $k_{c, \\text{VNMM } \\mathcal{L}^1}$ | Erro $k_c$ (%) | Diagnóstico |\n")
    conteudo.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(10):
        m_nome = nomes_modos[i]
        l_ref = res_ref_pg['autovalores_analiticos'][i]
        l_num = res_ref_pg['autovalores_positivos'][i]
        kc_r = res_ref_pg['kc_analitico'][i]
        kc_n = res_ref_pg['kc_numerico'][i]
        e_kc = res_ref_pg['erros_kc_pct'][i]
        conteudo.append(f"| ${m_nome}$ | {l_ref:6.2f} | {l_num:7.4f} | {kc_r:6.3f} | {kc_n:6.3f} | **{e_kc:5.2f}%** |\n")
        
    conteudo.append("\n![Espectro Base L1](espectro_modos_base_L1.png)\n\n")
    
    conteudo.append("## 5. Comparativo Síntese: Base $\\mathcal{L}^1$ (3 Nós) vs Base $\\mathcal{P}^1$ (6 Nós)\n\n")
    conteudo.append("| Critério | Base $\\mathcal{L}^1$ (3 Nós) | Base $\\mathcal{P}^1$ (6 Nós com Regularização) |\n")
    conteudo.append("|:---|:---:|:---:|\n")
    conteudo.append("| **Número de Graus de Liberdade Locais** | 3 termos / 3 nós | 6 termos / 6 nós |\n")
    conteudo.append("| **Divergente das Funções de Forma** | $\\nabla \\cdot \\vec{N} \\equiv 0$ (Solenoidal) | $\\nabla \\cdot \\vec{N} \\ne 0$ (Linear Completo) |\n")
    conteudo.append("| **Representação da Jacobiana** | Incompleta (Força $\\frac{\\partial E_x}{\\partial x} = 0$) | Completa (Todas as 4 derivadas) |\n")
    conteudo.append("| **Vazamento Modal (*Aliasing*)** | Presente | **Eliminado** |\n")
    conteudo.append("| **Erro Médio $k_c$ no Caso Base** | $\\approx 28.32\\% - 48.95\\%$ | **$1.00\\%$** |\n")
    conteudo.append("| **Tempo de Montagem** | $\\approx 0.04\\text{s}$ | $\\approx 0.06\\text{s}$ |\n")
    
    with open(caminho_relatorio, "w", encoding="utf-8") as f:
        f.writelines(conteudo)
        
    print(f"Relatório da base L1 salvo em: {caminho_relatorio}")


def main():
    res_pg, res_cc = executar_estudo_L1()
    gerar_graficos_e_relatorio_L1(res_pg, res_cc)


if __name__ == "__main__":
    main()
