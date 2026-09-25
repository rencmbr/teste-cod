#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate publication-quality figures for the VNMM 2D paper.
All figures are produced in vector PDF format (with 300 DPI PNG previews)
with all labels, titles, legends, and annotations strictly in English.

Target Figures:
1. figura_convergencia_interpolacao_L1_vs_P1.pdf (Section 6.1)
2. figura_espectro_cavidade_com_sem_regularizacao.pdf (Section 6.2)
3. figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf (Section 6.3)
4. figura_robustez_perturbacao_estocastica.pdf (Section 6.4)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Global Publication Styling (IEEE Trans. / Elsevier standard)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.2,
    "figure.titlesize": 12.5,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "lines.markersize": 6.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Color palette: curated, high-contrast, publication-grade
COLOR_NAVY = "#0f4c81"       # Deep Navy (Primary / P1)
COLOR_CRIMSON = "#c0392b"    # Crimson / Red (L1 / Spurious / Without Floor)
COLOR_TEAL = "#16a085"       # Teal / Emerald (Curl / Regularized)
COLOR_ORANGE = "#d35400"     # Amber / Orange (Hybrid)
COLOR_GREEN = "#27ae60"      # Green (FEM)
COLOR_PURPLE = "#8e44ad"     # Violet / Purple
COLOR_GRAY = "#555555"       # Neutral gray for reference lines
COLOR_LIGHT_BG = "#f8f9fa"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "relatorios")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# FIGURE 1: Interpolation Convergence L1 vs P1 (Section 6.1)
# =============================================================================
def generate_figure_1():
    print(">>> Generating Figure 1: L1 vs P1 Convergence (Section 6.1)...")
    
    # Numerical data from [0, pi]^2 benchmark (gerar_figura_convergencia_L1_vs_P1_pi.py)
    h_vals = np.array([0.5236, 0.3491, 0.2244, 0.1496, 0.0982, 0.0654])
    N_nodes = np.array([84, 186, 416, 884, 1928, 4192])
    
    # L1 (3-node incomplete formulation)
    E_rms_L1 = np.array([1.9136e-1, 1.1462e-1, 7.4377e-2, 5.0783e-2, 3.5623e-2, 2.4643e-2])
    rot_rms_L1 = np.array([8.6074e-1, 8.3762e-1, 8.3360e-1, 8.4263e-1, 8.8781e-1, 8.6317e-1])
    
    # P1 (6-node complete formulation with tolerance floor)
    E_rms_P1 = np.array([1.2430e-1, 3.9749e-2, 1.4656e-2, 6.3940e-3, 2.8591e-3, 1.2416e-3])
    rot_rms_P1 = np.array([2.5542e-1, 1.6616e-1, 1.0686e-1, 7.2390e-2, 4.8940e-2, 3.3319e-2])
    
    # Fitted slopes
    p_E_L1 = np.polyfit(np.log(h_vals), np.log(E_rms_L1), 1)[0]
    p_E_P1 = np.polyfit(np.log(h_vals), np.log(E_rms_P1), 1)[0]
    p_rot_L1 = np.polyfit(np.log(h_vals), np.log(rot_rms_L1), 1)[0]
    p_rot_P1 = np.polyfit(np.log(h_vals), np.log(rot_rms_P1), 1)[0]
    
    fig, (ax_E, ax_rot) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.92, wspace=0.24)
    
    # --- Subplot (a): Electric Field RMS Error ---
    ax_E.loglog(h_vals, E_rms_L1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                markeredgewidth=1.8, label=f"Incomplete Base $\\mathcal{{L}}^1$ (3 nodes): $\\mathcal{{O}}(h^{{{p_E_L1:.2f}}})$")
    ax_E.loglog(h_vals, E_rms_P1, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                label=f"Complete Base $\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_E_P1:.2f}}})$")
    
    # Reference lines
    h_ref = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = E_rms_P1[2] / (h_vals[2]**2)
    c_h1 = E_rms_L1[1] / (h_vals[1]**1)
    ax_E.loglog(h_ref, c_h2 * (h_ref**2), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^2)$ slope")
    ax_E.loglog(h_ref, c_h1 * (h_ref**1), 'gray', linestyle='-.', alpha=0.45, label=r"Theoretical $\mathcal{O}(h^1)$ slope")
    
    ax_E.set_ylim(5.0e-4, 0.35)
    ax_E.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{avg}}$ [m]")
    ax_E.set_ylabel(r"Electric Field RMS Error $\|\mathbf{E} - \mathbf{E}^h\|_{\mathrm{RMS}}$")
    ax_E.set_title(r"(a) Electric Field Convergence ($\mathbf{E}$)")
    ax_E.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_E.legend(loc="lower right", framealpha=0.92, fontsize=9.0)
    
    # --- Subplot (b): Curl RMS Error ---
    ax_rot.loglog(h_vals, rot_rms_L1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                  markeredgewidth=1.8, label=r"Incomplete Base $\mathcal{L}^1$ (3 nodes): $\mathcal{O}(1)$ Stagnation")
    ax_rot.loglog(h_vals, rot_rms_P1, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                  label=f"Complete Base $\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_rot_P1:.2f}}})$")
    
    c_curl1 = rot_rms_P1[2] / (h_vals[2]**1)
    ax_rot.loglog(h_ref, c_curl1 * (h_ref**1), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^1)$ slope")
    
    ax_rot.set_ylim(1.5e-2, 1.8)
    ax_rot.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{avg}}$ [m]")
    ax_rot.set_ylabel(r"Curl RMS Error $\|(\nabla \times \mathbf{E})_z - (\nabla \times \mathbf{E}^h)_z\|_{\mathrm{RMS}}$")
    ax_rot.set_title(r"(b) Curl Operator Convergence ($(\nabla \times \mathbf{E})_z$)")
    ax_rot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_rot.legend(loc="lower right", framealpha=0.92, fontsize=9.0)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_convergencia_interpolacao_L1_vs_P1.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_convergencia_interpolacao_L1_vs_P1.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# FIGURE 2: Cavity Discrete Spectrum With/Without Regularization (Section 6.2)
# =============================================================================
def generate_figure_2():
    print(">>> Generating Figure 2: Cavity Spectrum Regularization (Section 6.2)...")
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    
    # Analytical eigenvalues and cutoffs
    lambda_exact = np.array([1.00, 1.00, 2.00, 4.00, 4.00, 5.00, 5.00, 8.00, 9.00, 9.00])
    kc_exact = np.sqrt(lambda_exact)
    
    # Computed eigenvalues without regularization (s_div = 0.0)
    # Filtered lowest 10 positive eigenvalues from Table in relatorio_sem_regularizacao_divergente.md
    lambda_s0 = np.array([0.8659, 0.9921, 1.0608, 1.6836, 1.9761, 3.5889, 4.2328, 4.5317, 4.8789, 5.2358])
    is_spurious_s0 = [False, False, True, True, False, False, False, True, False, False]
    
    # Computed eigenvalues with div-curl regularization (s_div = 6.0)
    # From Table 4-1 benchmark (21x21 nodes)
    lambda_s6 = np.array([0.9679, 0.9836, 1.9409, 3.9275, 4.0724, 4.8925, 5.0287, 8.0743, 9.1089, 9.3409])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    
    width = 0.38
    
    # --- Subplot (a): Unregularized Spectrum (s_div = 0.0) ---
    bars_exact1 = ax1.bar(indices - width/2, lambda_exact, width, color="#7f8c8d", alpha=0.55,
                          edgecolor="#2c3e50", linewidth=1.0, label="Analytical Reference $\\lambda_{\\mathrm{exact}}$")
    
    # Bar colors for computed s0: Navy for physical, Crimson with hatching for spurious
    for i, idx in enumerate(indices):
        if is_spurious_s0[i]:
            ax1.bar(idx + width/2, lambda_s0[i], width, color="#fadbd8", edgecolor=COLOR_CRIMSON,
                    hatch="//", linewidth=1.4)
            ax1.text(idx + width/2, lambda_s0[i] + 0.18, "Spurious", ha="center", va="bottom",
                     fontsize=7.8, rotation=90, color=COLOR_CRIMSON, fontweight="bold")
        else:
            ax1.bar(idx + width/2, lambda_s0[i], width, color="#34495e", alpha=0.85,
                    edgecolor="#2c3e50", linewidth=1.0)
    
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f"Mode {i}" for i in indices], fontsize=9.0)
    ax1.set_ylabel(r"Discrete Eigenvalue $\lambda = k_c^2$")
    ax1.set_title(r"(a) Unregularized Formulation ($s_{\mathrm{div}} = 0.0$)" "\n"
                  r"Proliferation of Spurious Gradient Modes ($\mathbf{E} = \nabla \phi$)")
    ax1.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 10.5)
    
    # Custom legend for ax1
    legend_elements1 = [
        Patch(facecolor="#7f8c8d", alpha=0.55, edgecolor="#2c3e50", label="Analytical Reference"),
        Patch(facecolor="#34495e", alpha=0.85, edgecolor="#2c3e50", label="Physical Transverse Mode"),
        Patch(facecolor="#fadbd8", edgecolor=COLOR_CRIMSON, hatch="//", label="Spurious Gradient Mode")
    ]
    ax1.legend(handles=legend_elements1, loc="upper left", framealpha=0.92)
    
    ax1.annotate(r"Severe spectral corruption:" "\n" r"Gradient nullspace leakage" "\n" r"$\nabla \times (\nabla \phi) \neq 0$ discretely",
                 xy=(3.5, 1.8), xytext=(2.2, 6.6),
                 arrowprops=dict(arrowstyle="->", color=COLOR_CRIMSON, lw=1.3),
                 fontsize=8.8, fontweight="bold", color=COLOR_CRIMSON,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#fde8e8", edgecolor=COLOR_CRIMSON, alpha=0.95))
    
    # --- Subplot (b): Div-Curl Regularized Spectrum (s_div = 6.0) ---
    bars_exact2 = ax2.bar(indices - width/2, lambda_exact, width, color="#7f8c8d", alpha=0.55,
                          edgecolor="#2c3e50", linewidth=1.0, label="Analytical Reference $\\lambda_{\\mathrm{exact}}$")
    bars_s6 = ax2.bar(indices + width/2, lambda_s6, width, color=COLOR_TEAL, alpha=0.88,
                      edgecolor="#0e6251", linewidth=1.0, label=r"VNMM 2D $\mathcal{P}^1$ ($s_{\mathrm{div}} = 6.0$)")
    
    ax2.set_xticks(indices)
    ax2.set_xticklabels(mode_names, fontsize=9.2, fontweight="bold")
    ax2.set_title(r"(b) Div-Curl Regularized Formulation ($s_{\mathrm{div}} = 6.0$)" "\n"
                  r"Total Spectral Purification of Useful Bandwidth")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.5)
    
    # Add mode error labels on top of bars
    for i, idx in enumerate(indices):
        err_pct = abs(np.sqrt(lambda_s6[i]) - kc_exact[i]) / kc_exact[i] * 100.0
        ax2.text(idx + width/2, lambda_s6[i] + 0.15, f"{err_pct:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color="#0e6251", fontweight="bold")
        
    ax2.annotate("100% Spurious-Free Bandwidth\nAll gradient modes shifted to $\\lambda > 50$\nMean $k_c$ error = 1.00%",
                 xy=(5.0, 5.0), xytext=(1.8, 7.3),
                 arrowprops=dict(arrowstyle="->", color=COLOR_TEAL, lw=1.3),
                 fontsize=8.8, fontweight="bold", color="#0e6251",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e6f7f2", edgecolor=COLOR_TEAL, alpha=0.95))
    
    ax2.legend(loc="upper left", framealpha=0.92)
    
    fig.tight_layout()
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_espectro_cavidade_com_sem_regularizacao.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_espectro_cavidade_com_sem_regularizacao.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# FIGURE 3: Convergence Hybrid vs Pure VNMM vs Edge FEM (Section 6.3)
# =============================================================================
def generate_figure_3():
    print(">>> Generating Figure 3: Convergence Hybrid vs VNMM vs FEM (Section 6.3 - 6 meshes)...")
    
    # -------------------------------------------------------------------------
    # 1. Edge FEM (Nédélec 1st order): 1 DoF per internal edge (~3 DoFs / vertex)
    #    6 mesh levels: N in [9, 13, 17, 21, 25, 29]
    # -------------------------------------------------------------------------
    dofs_fem = np.array([176, 408, 736, 1160, 1680, 2296])
    h_fem = np.sqrt(np.pi**2 / dofs_fem)
    err_fem = np.array([0.810, 0.367, 0.208, 0.133, 0.093, 0.068])
    
    # -------------------------------------------------------------------------
    # 2. Balanced Hybrid (FEM-VNMM): Equivalence rho_edge,FEM approx rho_node,VNMM
    #    Balanced subdomains: FEM edges ~ VNMM nodes (~50% DoFs each)
    # -------------------------------------------------------------------------
    dofs_hybrid = np.array([170, 390, 727, 1134, 1672, 2243])
    h_hybrid = np.sqrt(np.pi**2 / dofs_hybrid)
    err_hybrid = np.array([3.144, 0.888, 0.636, 0.359, 0.173, 0.101])
    
    # -------------------------------------------------------------------------
    # 3. Pure VNMM 2D (P1): Edge-equivalent nodal density (Nv ~ sqrt(3)*(N-1)+1)
    #    Matching the active DoF resolution of Edge FEM across all 6 grid levels
    # -------------------------------------------------------------------------
    dofs_vnmm = np.array([169, 400, 729, 1156, 1681, 2209])
    h_vnmm = np.sqrt(np.pi**2 / dofs_vnmm)
    err_vnmm = np.array([10.233, 3.667, 1.645, 0.839, 0.493, 0.259])
    
    fig, (ax_dofs, ax_h) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.92, wspace=0.22)
    
    # --- Subplot (a): Convergence vs Degrees of Freedom (DoFs) ---
    ax_dofs.loglog(dofs_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                   linewidth=2.0, markersize=7.0, label=r"Pure Edge FEM (Nédélec, 1 DoF/edge)")
    ax_dofs.loglog(dofs_hybrid, err_hybrid, 's--', color=COLOR_ORANGE, markerfacecolor=COLOR_ORANGE,
                   linewidth=2.0, markersize=7.0, label=r"Balanced Hybrid ($\rho_{\mathrm{edge}}\approx\rho_{\mathrm{node}}$)")
    ax_dofs.loglog(dofs_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   linewidth=2.0, markersize=7.0, label=r"Pure VNMM 2D ($\mathcal{P}^1$, 1 DoF/node)")
    
    ax_dofs.set_xlabel(r"Number of Active Degrees of Freedom ($N_{\mathrm{DoF}}$)")
    ax_dofs.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_dofs.set_title(r"(a) Spectral Convergence vs. Problem Size ($N_{\mathrm{DoF}}$)")
    ax_dofs.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_dofs.legend(loc="upper right", framealpha=0.92, fontsize=9.2)
    
    # --- Subplot (b): Convergence vs Characteristic DoF Spacing h_DoF ---
    p_fem = np.polyfit(np.log(h_fem), np.log(err_fem), 1)[0]
    p_hybrid = np.polyfit(np.log(h_hybrid), np.log(err_hybrid), 1)[0]
    p_vnmm = np.polyfit(np.log(h_vnmm), np.log(err_vnmm), 1)[0]
    
    ax_h.loglog(h_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                linewidth=2.0, markersize=7.0, label=f"Edge FEM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_fem:.2f}}})$")
    ax_h.loglog(h_hybrid, err_hybrid, 's--', color=COLOR_ORANGE, markerfacecolor=COLOR_ORANGE,
                linewidth=2.0, markersize=7.0, label=f"Balanced Hybrid: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_hybrid:.2f}}})$")
    ax_h.loglog(h_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                linewidth=2.0, markersize=7.0, label=f"Pure VNMM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_vnmm:.2f}}})$")
    
    # Reference lines
    h_ref = np.linspace(0.060, 0.25, 100)
    ax_h.loglog(h_ref, 18.0 * (h_ref / h_ref[-1])**2, 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$")
    ax_h.loglog(h_ref, 2.5 * (h_ref / h_ref[-1])**1, 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$")
    
    ax_h.set_xlabel(r"Characteristic DoF Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]")
    ax_h.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_h.set_title(r"(b) Convergence with Characteristic Spacing ($h_{\mathrm{DoF}} \to 0$)")
    ax_h.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_h.legend(loc="lower right", framealpha=0.92, fontsize=8.8)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")



# =============================================================================
# FIGURE 4: Robustness Under Stochastic Perturbations & Floor Role (Section 6.4)
# =============================================================================
def generate_figure_4():
    print(">>> Generating Figure 4: Stochastic Robustness & Tolerance Floor (Section 6.4)...")
    
    # Numerical data for Section 6.3 on [0, pi]^2
    # Mesh refinement with random nodal jitter (25%) and random director angles (0 to 2pi)
    h_vals = np.array([0.5236, 0.3491, 0.2244, 0.1496, 0.0982, 0.0654])
    
    # WITHOUT floor: Tol_det proportional to h^4 (permissive unconstrained scaling)
    E_rms_sem = np.array([1.5734e-1, 6.8018e-2, 2.1490e-2, 8.8302e-3, 3.9952e-3, 1.9496e-3])
    rot_rms_sem = np.array([3.9767e-1, 2.4035e-1, 1.6794e-1, 1.1930e-1, 8.6573e-2, 6.0053e-2])
    
    # WITH tolerance floor Tol_floor = 1.5e-5
    E_rms_com = np.array([1.2430e-1, 3.9749e-2, 1.4656e-2, 6.3940e-3, 2.8591e-3, 1.2100e-3])
    rot_rms_com = np.array([2.5542e-1, 1.6616e-1, 1.0686e-1, 7.2390e-2, 4.8940e-2, 3.2000e-2])
    
    # Tolerance sweep in the fine mesh (N = 4192, h = 0.0654 m)
    tol_sweep = np.array([1.0e-6, 3.0e-6, 8.0e-6, 1.5e-5, 3.0e-5, 6.0e-5, 1.0e-4])
    E_rms_sweep = np.array([2.21e-3, 1.34e-3, 1.15e-3, 1.21e-3, 1.41e-3, 1.80e-3, 2.09e-3])
    rot_rms_sweep = np.array([6.86e-2, 4.09e-2, 3.27e-2, 3.20e-2, 3.33e-2, 3.33e-2, 3.46e-2])
    K_avg_sweep = np.array([6.3, 7.0, 8.8, 10.3, 11.3, 11.9, 13.5])
    
    fig, (ax_conv, ax_tol) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.15, top=0.92, wspace=0.34)
    
    # --- Subplot (a): Convergence under Stochastic Perturbations ---
    ax_conv.loglog(h_vals, rot_rms_sem, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                   markeredgewidth=1.8, label=r"Curl RMS: Without Floor ($\mathrm{Tol}_{\mathrm{det}} \propto h^4$)")
    ax_conv.loglog(h_vals, rot_rms_com, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                   label=r"Curl RMS: With Floor ($\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$)")
    
    ax_conv.loglog(h_vals, E_rms_sem, '^--', color="#7f8c8d", markerfacecolor='white',
                   markeredgewidth=1.8, label=r"$\mathbf{E}$ RMS: Without Floor")
    ax_conv.loglog(h_vals, E_rms_com, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   label=r"$\mathbf{E}$ RMS: With Floor ($\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$)")
    
    # Reference lines
    h_ref = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = E_rms_com[2] / (h_vals[2]**2)
    c_h1 = rot_rms_com[2] / (h_vals[2]**1)
    ax_conv.loglog(h_ref, c_h2 * (h_ref**2), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^2)$ slope")
    ax_conv.loglog(h_ref, c_h1 * (h_ref**1), 'gray', linestyle='-.', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^1)$ slope")
    
    ax_conv.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{avg}}$ [m]")
    ax_conv.set_ylabel("RMS Interpolation Error")
    ax_conv.set_title(r"(a) Convergence Under Stochastic Perturbations")
    ax_conv.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_conv.set_ylim(8.0e-4, 1.2)
    ax_conv.legend(loc="lower right", fontsize=8.2, framealpha=0.92)
    
    # --- Subplot (b): Stabilizing Role of the Tolerance Floor ---
    ax_tol.set_xscale('log')
    ax_tol.set_yscale('log')
    
    l1 = ax_tol.plot(tol_sweep, rot_rms_sweep, 'd-', color=COLOR_CRIMSON, linewidth=2.0,
                     markerfacecolor=COLOR_CRIMSON, label=r"Curl RMS $\|(\nabla \times \mathbf{E})_z - (\nabla \times \mathbf{E}^h)_z\|$")
    l2 = ax_tol.plot(tol_sweep, E_rms_sweep, 'o-', color=COLOR_NAVY, linewidth=2.0,
                     markerfacecolor=COLOR_NAVY, label=r"Field RMS $\|\mathbf{E} - \mathbf{E}^h\|$")
    
    ax_tol.set_xlabel(r"Determinant Threshold Criterion $\mathrm{Tol}_{\mathrm{det}}$")
    ax_tol.set_ylabel("RMS Interpolation Error", color="#2c3e50")
    ax_tol.tick_params(axis='y', labelcolor="#2c3e50")
    ax_tol.set_xlim(7.0e-7, 1.5e-4)
    ax_tol.set_ylim(8.0e-4, 0.12)
    
    # Twin axis for neighborhood size K_avg
    ax_k = ax_tol.twinx()
    l3 = ax_k.plot(tol_sweep, K_avg_sweep, 's--', color=COLOR_ORANGE, linewidth=2.0,
                   markerfacecolor='white', markeredgewidth=1.8, label=r"Neighborhood $K_{\mathrm{avg}}$ (right axis)")
    ax_k.set_ylabel(r"Candidate Neighbors Queried $K_{\mathrm{avg}}$", color=COLOR_ORANGE)
    ax_k.tick_params(axis='y', labelcolor=COLOR_ORANGE)
    ax_k.set_ylim(5, 16)
    
    # Vertical line for floor reference
    l4 = [ax_tol.axvline(1.5e-5, color="#27ae60", linestyle="--", linewidth=1.6, alpha=0.85,
                         label=r"Stabilizing Floor $\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$")]
    
    # Combined legend for ax_tol and ax_k
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax_tol.legend(lines, labels, loc="upper right", fontsize=8.2, framealpha=0.92)
    
    ax_tol.set_title(r"(b) Stabilization vs. Tolerance Floor ($N = 4{,}192$)")
    ax_tol.grid(True, which="both", linestyle="--", alpha=0.5)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_robustez_perturbacao_estocastica.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_robustez_perturbacao_estocastica.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("  GENERATING PUBLICATION-QUALITY FIGURES FOR VNMM 2D PAPER")
    print("  Output format: PDF (Vector) + PNG (300 DPI Preview)")
    print(f"  Destination directory: {OUTPUT_DIR}")
    print("=================================================================\n")
    
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    
    print("\nAll 4 publication figures generated successfully!")
