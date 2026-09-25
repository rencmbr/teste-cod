import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc

# Directories
DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
os.makedirs(DIRETORIO_RELATORIOS, exist_ok=True)

def draw_vector_arrow(ax, start, vec, color, length=0.18, label=None, label_offset=(0, 0), lw=3.0, zorder=5):
    """Draws a normalized unit vector arrow originating at start."""
    v_norm = vec / np.linalg.norm(vec)
    end = start + length * v_norm
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>,head_length=7.0,head_width=4.8",
                            color=color, lw=lw, zorder=zorder)
    ax.add_patch(arrow)
    if label:
        ax.text(end[0] + label_offset[0], end[1] + label_offset[1], label,
                fontsize=15.5, fontweight="bold", color=color, ha="center", va="center", zorder=zorder+1)
    return end

def draw_line_of_action(ax, pt, vec, color, t_range=(-0.4, 0.5), lw=1.6, ls='--', alpha=0.65, zorder=2):
    """Draws the infinite line of action passing through pt along vec."""
    v_norm = vec / np.linalg.norm(vec)
    p_start = pt + t_range[0] * v_norm
    p_end = pt + t_range[1] * v_norm
    ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]],
            color=color, linestyle=ls, lw=lw, alpha=alpha, zorder=zorder)

def plot_node(ax, pt, label=None, color="#2c3e50", offset=(0.04, -0.04), markersize=9.0, zorder=6):
    """Plots a support node marker and its label."""
    ax.plot(pt[0], pt[1], 'o', color=color, markersize=markersize, markeredgecolor='white', markeredgewidth=1.6, zorder=zorder)
    if label:
        ax.text(pt[0] + offset[0], pt[1] + offset[1], label,
                fontsize=15.5, fontweight="bold", color=color, ha="left", va="center", zorder=zorder+1)

def gerar_figura_singularidades():
    """
    Generates high-resolution publication-quality figure illustrating the
    geometric and directional singularities of moment matrix A for VNMM 2D (L1 basis).
    
    Optimized for paper readability:
      - All cramped bulleted math cards and top status banners removed (details in caption).
      - Font size increased to 15.0-16.5 pt bold matching body text.
      - Clean visual geometry utilizing the full subplot area.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 13.5,
        'axes.labelsize': 15.0,
        'axes.titlesize': 15.0,
        'xtick.labelsize': 12.5,
        'ytick.labelsize': 12.5,
        'legend.fontsize': 12.5
    })

    fig = plt.figure(figsize=(14.0, 9.2), facecolor='white')
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.94, bottom=0.07,
                          wspace=0.22, hspace=0.28)

    c_tri = "#34495e"
    c_red = "#c0392b"
    c_blue = "#2980b9"
    c_green = "#27ae60"
    c_purple = "#8e44ad"
    c_orange = "#d35400"
    c_bg = "#fdfefe"

    # =========================================================================
    # PANEL (a): Global Vector Parallelism (t1 || t2 || t3)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(c_bg)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_aspect('equal')
    ax_a.set_title(r"$\mathbf{(a)}$ Parallel Vectors ($\mathbf{t}_1 \parallel \mathbf{t}_2 \parallel \mathbf{t}_3$)",
                   pad=10, fontweight="bold", color="#1c2833")

    p1_a = np.array([0.18, 0.30])
    p2_a = np.array([0.80, 0.35])
    p3_a = np.array([0.48, 0.80])
    tri_a = np.array([p1_a, p2_a, p3_a])

    poly_a = patches.Polygon(tri_a, closed=True, facecolor="#eaf2f8", edgecolor="#aed6f1",
                             linestyle="--", lw=2.0, zorder=1)
    ax_a.add_patch(poly_a)

    theta_a = np.deg2rad(15.0)
    t_par = np.array([np.cos(theta_a), np.sin(theta_a)])

    nodes_a = [p1_a, p2_a, p3_a]
    colors_a = [c_red, c_blue, c_green]
    lbl_offsets_a = [(-0.06, -0.07), (0.04, -0.05), (-0.02, 0.08)]
    arrow_offsets_a = [(0.04, 0.04), (0.04, 0.04), (0.04, 0.04)]
    
    for i in range(3):
        plot_node(ax_a, nodes_a[i], rf"$\mathbf{{x}}_{i+1}$", color=c_tri, offset=lbl_offsets_a[i])
        draw_line_of_action(ax_a, nodes_a[i], t_par, color="#95a5a6", t_range=(-0.16, 0.30), lw=1.5, ls=":")
        draw_vector_arrow(ax_a, nodes_a[i], t_par, color=colors_a[i], length=0.20,
                          label=rf"$\mathbf{{t}}_{i+1}$", label_offset=arrow_offsets_a[i])

    ax_a.set_xlabel(r"$x$")
    ax_a.set_ylabel(r"$y$")
    ax_a.grid(True, linestyle=":", alpha=0.45)

    # =========================================================================
    # PANEL (b): Concurrency of Lines of Action (Focal Point Pc)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(c_bg)
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_aspect('equal')
    ax_b.set_title(r"$\mathbf{(b)}$ Concurrent Lines of Action ($P_c$)",
                   pad=10, fontweight="bold", color="#1c2833")

    Pc = np.array([0.50, 0.52])
    p1_b = np.array([0.20, 0.26])
    p2_b = np.array([0.80, 0.26])
    p3_b = np.array([0.50, 0.88])
    nodes_b = [p1_b, p2_b, p3_b]
    colors_b = [c_red, c_blue, c_green]
    lbl_offsets_b = [(-0.06, -0.07), (0.04, -0.06), (0.04, 0.02)]
    arrow_lbl_offsets_b = [(-0.04, 0.05), (0.04, 0.05), (-0.05, 0.0)]

    tb = []
    for pt in nodes_b:
        vec = Pc - pt
        tb.append(vec / np.linalg.norm(vec))

    for i in range(3):
        plot_node(ax_b, nodes_b[i], rf"$\mathbf{{x}}_{i+1}$", color=c_tri, offset=lbl_offsets_b[i])
        draw_line_of_action(ax_b, nodes_b[i], tb[i], color="#e74c3c", t_range=(-0.08, 0.48), lw=1.6, ls="--", alpha=0.65)
        draw_vector_arrow(ax_b, nodes_b[i], tb[i], color=colors_b[i], length=0.18,
                          label=rf"$\mathbf{{t}}_{i+1}$", label_offset=arrow_lbl_offsets_b[i])

    ax_b.plot(Pc[0], Pc[1], 'X', color=c_red, markersize=12, markeredgecolor='black', markeredgewidth=1.4, zorder=8)
    ax_b.text(Pc[0] + 0.04, Pc[1] + 0.03, r"$P_c$",
              fontsize=16.5, fontweight="bold", color=c_red, ha="left", va="bottom", zorder=9)

    ax_b.set_xlabel(r"$x$")
    ax_b.set_ylabel(r"$y$")
    ax_b.grid(True, linestyle=":", alpha=0.45)

    # =========================================================================
    # PANEL (c): Overlapping Lines of Action (x1, x2 in L)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor(c_bg)
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_aspect('equal')
    ax_c.set_title(r"$\mathbf{(c)}$ Shared Line of Action ($\mathbf{x}_1, \mathbf{x}_2 \in \mathcal{L}$)",
                   pad=10, fontweight="bold", color="#1c2833")

    theta_c = np.deg2rad(32.0)
    dir_L = np.array([np.cos(theta_c), np.sin(theta_c)])
    p1_c = np.array([0.18, 0.32])
    p2_c = p1_c + 0.46 * dir_L
    p3_c = np.array([0.42, 0.82])

    draw_line_of_action(ax_c, p1_c, dir_L, color="#e74c3c", t_range=(-0.16, 0.76), lw=2.2, ls="-.", alpha=0.75)
    ax_c.text(0.74, 0.74, r"$\mathcal{L}$",
              fontsize=17.0, fontweight="bold", color=c_red, ha="center", va="bottom")

    plot_node(ax_c, p1_c, r"$\mathbf{x}_1$", color=c_tri, offset=(-0.08, -0.05))
    plot_node(ax_c, p2_c, r"$\mathbf{x}_2$", color=c_tri, offset=(0.04, -0.05))
    plot_node(ax_c, p3_c, r"$\mathbf{x}_3$", color=c_tri, offset=(0.04, 0.02))

    draw_vector_arrow(ax_c, p1_c, dir_L, color=c_red, length=0.18, label=r"$\mathbf{t}_1$", label_offset=(0.03, 0.04))
    draw_vector_arrow(ax_c, p2_c, dir_L, color=c_blue, length=0.18, label=r"$\mathbf{t}_2$", label_offset=(0.03, 0.04))

    t3_c = np.array([-0.7, -0.7])
    t3_c = t3_c / np.linalg.norm(t3_c)
    draw_vector_arrow(ax_c, p3_c, t3_c, color=c_green, length=0.17, label=r"$\mathbf{t}_3$", label_offset=(-0.05, -0.03))

    ax_c.set_xlabel(r"$x$")
    ax_c.set_ylabel(r"$y$")
    ax_c.grid(True, linestyle=":", alpha=0.45)

    # =========================================================================
    # PANEL (d): Nodal Coincidence / Spatial Collapse (x1 ~ x2)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    ax_d.set_facecolor(c_bg)
    ax_d.set_xlim(0.0, 1.0)
    ax_d.set_ylim(0.0, 1.0)
    ax_d.set_aspect('equal')
    ax_d.set_title(r"$\mathbf{(d)}$ Nodal Coincidence ($\mathbf{x}_1 \approx \mathbf{x}_2$)",
                   pad=10, fontweight="bold", color="#1c2833")

    p1_d = np.array([0.32, 0.44])
    p2_d = np.array([0.40, 0.49]) # Very close
    p3_d = np.array([0.76, 0.80])

    halo = patches.Ellipse((0.36, 0.465), 0.28, 0.22, angle=28,
                           facecolor="#fadbd8", edgecolor="#e74c3c", linestyle="--", lw=1.8, zorder=2)
    ax_d.add_patch(halo)

    plot_node(ax_d, p1_d, r"$\mathbf{x}_1$", color=c_red, offset=(-0.09, 0.04))
    plot_node(ax_d, p2_d, r"$\mathbf{x}_2$", color=c_blue, offset=(0.04, -0.06))
    plot_node(ax_d, p3_d, r"$\mathbf{x}_3$", color=c_tri, offset=(0.04, -0.04))

    ax_d.annotate("", xy=p2_d, xytext=p1_d - np.array([0.06, 0.05]),
                  arrowprops=dict(arrowstyle="->", color=c_red, lw=2.0))

    t1_d = np.array([0.9, 0.43])
    t1_d = t1_d / np.linalg.norm(t1_d)
    t2_d = np.array([0.85, 0.52])
    t2_d = t2_d / np.linalg.norm(t2_d)
    t3_d = np.array([-0.3, 0.95])
    t3_d = t3_d / np.linalg.norm(t3_d)

    draw_vector_arrow(ax_d, p1_d, t1_d, color=c_red, length=0.18, label=r"$\mathbf{t}_1$", label_offset=(0.05, -0.03))
    draw_vector_arrow(ax_d, p2_d, t2_d, color=c_blue, length=0.18, label=r"$\mathbf{t}_2$", label_offset=(0.05, 0.04))
    draw_vector_arrow(ax_d, p3_d, t3_d, color=c_green, length=0.18, label=r"$\mathbf{t}_3$", label_offset=(0.05, -0.02))

    ax_d.set_xlabel(r"$x$")
    ax_d.set_ylabel(r"$y$")
    ax_d.grid(True, linestyle=":", alpha=0.45)

    # =========================================================================
    # PANEL (e): Decoupling: Spatial Geometry vs. Vector Alignment
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.set_facecolor(c_bg)
    ax_e.set_xlim(0.0, 1.0)
    ax_e.set_ylim(0.0, 1.0)
    ax_e.set_aspect('equal')
    ax_e.set_title(r"$\mathbf{(e)}$ Decoupling: Spatial vs. Directional",
                   pad=10, fontweight="bold", color="#1c2833")

    ax_e.axhline(0.50, color="#bdc3c7", linestyle=":", lw=1.6, zorder=1)

    # SUB-CASE E1 (Top): Spatially Non-collinear, but Vectors Parallel
    p1_e1 = np.array([0.20, 0.66])
    p2_e1 = np.array([0.80, 0.66])
    p3_e1 = np.array([0.50, 0.90])
    tri_e1 = np.array([p1_e1, p2_e1, p3_e1])
    poly_e1 = patches.Polygon(tri_e1, closed=True, facecolor="#fadbd8", edgecolor="#e74c3c", lw=1.5, zorder=2)
    ax_e.add_patch(poly_e1)

    t_e1 = np.array([1.0, 0.0])
    for pt in [p1_e1, p2_e1, p3_e1]:
        ax_e.plot(pt[0], pt[1], 'o', color=c_tri, markersize=7.5, zorder=4)
        draw_vector_arrow(ax_e, pt, t_e1, color=c_red, length=0.14, lw=2.2)

    ax_e.text(0.50, 0.57, r"Non-collinear nodes, $\mathbf{t}_k \parallel \mathbf{t}_j$" + "\n" + r"$\det(\mathbf{A}) = 0$ (Singular)",
              fontsize=12.5, fontweight="bold", color=c_red, ha="center", va="center", zorder=6)

    # SUB-CASE E2 (Bottom): Collinear Nodes, but Vectors Non-parallel
    p1_e2 = np.array([0.20, 0.25])
    p2_e2 = np.array([0.50, 0.25])
    p3_e2 = np.array([0.80, 0.25])
    ax_e.plot([p1_e2[0], p3_e2[0]], [p1_e2[1], p3_e2[1]], color="#d35400", lw=2.6, zorder=2)

    t1_e2 = np.array([0.0, 1.0])
    t2_e2 = np.array([1.0, 0.0])
    t3_e2 = np.array([0.0, 1.0])
    dirs_e2 = [t1_e2, t2_e2, t3_e2]
    for i, pt in enumerate([p1_e2, p2_e2, p3_e2]):
        ax_e.plot(pt[0], pt[1], 's', color=c_tri, markersize=8.0, zorder=4)
        draw_vector_arrow(ax_e, pt, dirs_e2[i], color=c_purple, length=0.14, lw=2.2)

    ax_e.text(0.50, 0.10, r"Collinear nodes, $\mathbf{t}_k \nparallel \mathbf{t}_j$" + "\n" + r"$\det(\mathbf{A}) \neq 0$ (Zero 2D span)",
              fontsize=12.5, fontweight="bold", color="#b7950b", ha="center", va="center", zorder=6)

    ax_e.set_xlabel(r"$x$")
    ax_e.set_ylabel(r"$y$")
    ax_e.grid(True, linestyle=":", alpha=0.45)

    # =========================================================================
    # PANEL (f): Robust Admissible Support (Circulation)
    # =========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    ax_f.set_facecolor(c_bg)
    ax_f.set_xlim(0.0, 1.0)
    ax_f.set_ylim(0.0, 1.0)
    ax_f.set_aspect('equal')
    ax_f.set_title(r"$\mathbf{(f)}$ Admissible Support (Circulation)",
                   pad=10, fontweight="bold", color="#1c2833")

    P_eval = np.array([0.50, 0.55])
    v1_f = np.array([0.18, 0.28])
    v2_f = np.array([0.84, 0.28])
    v3_f = np.array([0.50, 0.88])
    macro_tri = np.array([v1_f, v2_f, v3_f])

    poly_macro = patches.Polygon(macro_tri, closed=True, facecolor="#eafaf1", edgecolor="#a9dfbf",
                                 linestyle="-", lw=2.0, zorder=1)
    ax_f.add_patch(poly_macro)

    m1_f = 0.5 * (v1_f + v2_f)
    m2_f = 0.5 * (v2_f + v3_f)
    m3_f = 0.5 * (v3_f + v1_f)
    nodes_f = [m1_f, m2_f, m3_f]
    lbl_offsets_f = [(0.0, -0.06), (0.04, -0.02), (-0.11, 0.02)]

    t1_f = (v2_f - v1_f) / np.linalg.norm(v2_f - v1_f)
    t2_f = (v3_f - v2_f) / np.linalg.norm(v3_f - v2_f)
    t3_f = (v1_f - v3_f) / np.linalg.norm(v1_f - v3_f)
    dirs_f = [t1_f, t2_f, t3_f]
    colors_f = [c_red, c_blue, c_green]
    arrow_lbl_offsets_f = [(0.04, 0.03), (0.04, 0.03), (-0.04, -0.03)]

    for i in range(3):
        draw_line_of_action(ax_f, nodes_f[i], dirs_f[i], color="#27ae60", t_range=(-0.25, 0.45), lw=1.5, ls="--", alpha=0.65)
        plot_node(ax_f, nodes_f[i], rf"$\mathbf{{x}}_{i+1}$", color=c_tri, offset=lbl_offsets_f[i])
        draw_vector_arrow(ax_f, nodes_f[i], dirs_f[i], color=colors_f[i], length=0.18,
                          label=rf"$\mathbf{{t}}_{i+1}$", label_offset=arrow_lbl_offsets_f[i])

    ax_f.plot(P_eval[0], P_eval[1], '*', color=c_orange, markersize=14, markeredgecolor='black', markeredgewidth=1.2, zorder=8)
    ax_f.text(P_eval[0] + 0.05, P_eval[1] - 0.04, r"$P$",
              fontsize=16.5, fontweight="bold", color=c_orange, ha="left", va="center", zorder=9)

    arc = Arc(P_eval, 0.26, 0.26, angle=0, theta1=20, theta2=280,
              color=c_orange, linestyle=":", lw=2.2, zorder=6)
    ax_f.add_patch(arc)
    ax_f.annotate("", xy=(P_eval[0] + 0.04, P_eval[1] + 0.13), xytext=(P_eval[0] - 0.04, P_eval[1] + 0.13),
                  arrowprops=dict(arrowstyle="-|>", color=c_orange, lw=2.4, mutation_scale=16), zorder=7)

    ax_f.set_xlabel(r"$x$")
    ax_f.set_ylabel(r"$y$")
    ax_f.grid(True, linestyle=":", alpha=0.45)

    # Save outputs
    caminho_pdf = os.path.join(DIRETORIO_RELATORIOS, "figura_singularidades_matriz_A.pdf")
    caminho_png = os.path.join(DIRETORIO_RELATORIOS, "figura_singularidades_matriz_A.png")
    
    plt.savefig(caminho_pdf, format="pdf", dpi=300)
    plt.savefig(caminho_png, format="png", dpi=300)
    plt.close()
    print(f"Figura salva com sucesso em:\n  - {caminho_pdf}\n  - {caminho_png}")

if __name__ == "__main__":
    gerar_figura_singularidades()
