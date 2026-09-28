import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Directories
DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
os.makedirs(DIRETORIO_RELATORIOS, exist_ok=True)

def criar_figura_equivalencia_dual():
    """
    Generates high-resolution publication-quality figure (Elsevier / IEEE standard)
    illustrating the exact dual equivalence between:
      - Whitney Edge Finite Elements (lowest-order Nedelec 1-forms on triangles)
      - Vector Nodal Meshless Method 2D (VNMM 2D) with the solenoidal L1 basis.
    
    Design criteria:
      - Fully removed cramped equations from inside diagram panels.
      - Removed the overlapping green equation box.
      - Text labels are prominent, bold, and sized to match common publication body fonts.
      - Clean 5-panel layout with balanced vertical and horizontal spacing.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'mathtext.fontset': 'dejavusans',
        'font.size': 13.5,
        'axes.labelsize': 15.5,
        'axes.titlesize': 15.5,
        'xtick.labelsize': 13.0,
        'ytick.labelsize': 13.0,
        'legend.fontsize': 13.0
    })

    # Geometry of a general scalene, non-degenerate reference triangle
    v1 = np.array([0.15, 0.15])
    v2 = np.array([1.25, 0.25])
    v3 = np.array([0.55, 1.15])
    vertices = np.array([v1, v2, v3])

    edges = [(v1, v2), (v2, v3), (v3, v1)]
    L = [np.linalg.norm(eb - ea) for ea, eb in edges]
    t = [(eb - ea) / np.linalg.norm(eb - ea) for ea, eb in edges]
    m = [0.5 * (ea + eb) for ea, eb in edges]

    det2 = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    area = 0.5 * det2

    grad_L = np.zeros((3, 2))
    grad_L[0] = [(v2[1] - v3[1]) / det2, (v3[0] - v2[0]) / det2]
    grad_L[1] = [(v3[1] - v1[1]) / det2, (v1[0] - v3[0]) / det2]
    grad_L[2] = [(v1[1] - v2[1]) / det2, (v2[0] - v1[0]) / det2]

    def bary_coords(x, y):
        l1 = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / det2
        l2 = ((v3[1] - v1[1]) * (x - v1[0]) + (v1[0] - v3[0]) * (y - v1[1])) / det2
        l3 = 1.0 - l1 - l2
        return l1, l2, l3

    def whitney_w1(x, y):
        l1, l2, l3 = bary_coords(x, y)
        wx = l1 * grad_L[1, 0] - l2 * grad_L[0, 0]
        wy = l1 * grad_L[1, 1] - l2 * grad_L[0, 1]
        return wx, wy

    # VNMM L1 moment matrix with nodes placed at edge midpoints m[0], m[1], m[2]
    A = np.zeros((3, 3))
    for i in range(3):
        A[i, 0] = t[i][0]
        A[i, 1] = t[i][1]
        A[i, 2] = m[i][1] * t[i][0] - m[i][0] * t[i][1]
    beta = np.linalg.inv(A)

    def vnmm_N1(x, y):
        Nx = beta[0, 0] + beta[2, 0] * y
        Ny = beta[1, 0] - beta[2, 0] * x
        return Nx, Ny

    # Canvas dimensions: 13.5 x 8.6 inches
    fig = plt.figure(figsize=(13.5, 8.6))
    
    # Top Grid: 3 conceptual diagrams (a), (b), (c)
    gs_top = fig.add_gridspec(1, 3, left=0.04, right=0.96, top=0.95, bottom=0.52, wspace=0.18)
    
    # Bottom Grid: 2 vector field panels (d), (e) side-by-side
    gs_bottom = fig.add_gridspec(1, 2, left=0.08, right=0.92, top=0.45, bottom=0.07, wspace=0.22)

    c_tri = "#2c3e50"
    c_edge1 = "#c0392b"
    c_edge2 = "#2980b9"
    c_edge3 = "#27ae60"
    c_vnmm = "#8e44ad"
    c_bg = "#fdfefe"

    # =========================================================================
    # PANEL (a): Whitney Edge Element (Nedelec 1-forms on Triangle)
    # =========================================================================
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_facecolor(c_bg)

    tri_patch_a = patches.Polygon(vertices, closed=True, facecolor="#ebf5fb", edgecolor=c_tri, linewidth=2.5, zorder=2)
    ax_a.add_patch(tri_patch_a)

    for i, v in enumerate(vertices):
        ax_a.plot(v[0], v[1], 'o', color=c_tri, markersize=9.5, zorder=6)
    ax_a.text(v1[0] - 0.10, v1[1] - 0.07, r"$V_1$", fontsize=16.5, fontweight="bold", color=c_tri)
    ax_a.text(v2[0] + 0.04, v2[1] - 0.05, r"$V_2$", fontsize=16.5, fontweight="bold", color=c_tri)
    ax_a.text(v3[0] - 0.02, v3[1] + 0.08, r"$V_3$", fontsize=16.5, fontweight="bold", color=c_tri)

    cores_arestas = [c_edge1, c_edge2, c_edge3]
    labels_arestas = [r"$e_1$", r"$e_2$", r"$e_3$"]
    for i, (ea, eb) in enumerate(edges):
        ax_a.annotate("", xy=eb, xytext=ea,
                      arrowprops=dict(arrowstyle="-|>", color=cores_arestas[i], lw=3.6, mutation_scale=22),
                      zorder=4)
        mid = m[i]
        if i == 0:
            offset = np.array([0.0, -0.13])
        elif i == 1:
            offset = np.array([0.10, 0.04])
        else:
            offset = np.array([-0.12, 0.03])
        ax_a.text(mid[0] + offset[0], mid[1] + offset[1], labels_arestas[i],
                  fontsize=16.5, fontweight="bold", color=cores_arestas[i], ha="center")

    centroid = np.mean(vertices, axis=0)
    ax_a.text(centroid[0], centroid[1], r"$\Omega_e$",
              fontsize=17.0, fontweight="bold", color="#34495e", ha="center", va="center",
              bbox=dict(boxstyle="circle,pad=0.35", facecolor="white", edgecolor="#bdc3c7", lw=1.5, alpha=0.95))

    ax_a.set_xlim(-0.14, 1.48)
    ax_a.set_ylim(-0.10, 1.40)
    ax_a.set_aspect('equal')
    ax_a.set_title(r"$\mathbf{(a)}$ FEM Whitney Edge Element", pad=12, fontsize=15.5, fontweight="bold")
    ax_a.axis("off")

    # =========================================================================
    # PANEL (b): Asymptotic Dual Collapse Limit (Lk -> 0)
    # Centered at x = 0.65 to align with title/caption
    # =========================================================================
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.set_facecolor(c_bg)

    y_pos = [0.88, 0.52, 0.16]
    x_c = 0.65  # Center coordinate for panel (b)

    # Stage 1: Finite continuous edge (centered at x_c)
    ax_b.plot([x_c - 0.35, x_c + 0.35], [y_pos[0], y_pos[0]], lw=4.5, color=c_edge1, solid_capstyle='round')
    ax_b.annotate("", xy=(x_c + 0.35, y_pos[0]), xytext=(x_c - 0.35, y_pos[0]),
                  arrowprops=dict(arrowstyle="-|>", color=c_edge1, lw=3.2, mutation_scale=20))
    ax_b.plot(x_c, y_pos[0], 'o', color="#2c3e50", markersize=8.0)
    ax_b.text(x_c, y_pos[0] + 0.10, r"1. Continuous Edge $e_k$ ($L_k$)",
              fontsize=15.0, fontweight="bold", color=c_edge1, ha="center")

    # Arrow 1: Edge contraction (vertically centered, group centered around x_c)
    y_mid_1 = 0.745
    ax_b.annotate("", xy=(x_c - 0.20, y_mid_1 - 0.055), xytext=(x_c - 0.20, y_mid_1 + 0.055),
                  arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.4, mutation_scale=16))
    ax_b.text(x_c - 0.14, y_mid_1, r"Edge contraction ($L_k \to 0$)", fontsize=13.5, color="#555555", fontstyle="italic", va="center")

    # Stage 2: Shortened edge (centered at x_c)
    ax_b.plot([x_c - 0.20, x_c + 0.20], [y_pos[1], y_pos[1]], lw=4.0, color="#e67e22", solid_capstyle='round')
    ax_b.annotate("", xy=(x_c + 0.20, y_pos[1]), xytext=(x_c - 0.20, y_pos[1]),
                  arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.8, mutation_scale=18))
    ax_b.plot(x_c, y_pos[1], 'o', color="#2c3e50", markersize=8.0)
    ax_b.text(x_c, y_pos[1] + 0.10, r"2. Contracted Edge ($L_k \ll 1$)",
              fontsize=15.0, fontweight="bold", color="#d35400", ha="center")

    # Arrow 2: Pointwise limit (vertically centered, group centered around x_c)
    y_mid_2 = 0.385
    ax_b.annotate("", xy=(x_c - 0.15, y_mid_2 - 0.055), xytext=(x_c - 0.15, y_mid_2 + 0.055),
                  arrowprops=dict(arrowstyle="-|>", color="#555555", lw=2.4, mutation_scale=16))
    ax_b.text(x_c - 0.09, y_mid_2, r"Pointwise limit", fontsize=13.5, color="#555555", fontstyle="italic", va="center")

    # Stage 3: VNMM Vector Node (centered at x_c)
    p_node = np.array([x_c - 0.14, y_pos[2]])
    t_vec = np.array([0.28, 0.0])
    ax_b.plot(p_node[0], p_node[1], 'o', color=c_vnmm, markersize=10.5, zorder=5)
    ax_b.annotate("", xy=(p_node[0] + t_vec[0], p_node[1]), xytext=(p_node[0], p_node[1]),
                  arrowprops=dict(arrowstyle="-|>", color=c_vnmm, lw=3.8, mutation_scale=22), zorder=4)
    ax_b.text(p_node[0] - 0.06, p_node[1], r"$P_k$", fontsize=16.0, fontweight="bold", color=c_vnmm, ha="right", va="center")
    ax_b.text(p_node[0] + t_vec[0] + 0.06, p_node[1], r"$\mathbf{t}_k$", fontsize=16.0, fontweight="bold", color=c_vnmm, ha="left", va="center")
    ax_b.text(x_c, y_pos[2] + 0.10, r"3. VNMM Vector Node ($P_k, \mathbf{t}_k$)",
              fontsize=15.0, fontweight="bold", color=c_vnmm, ha="center")

    # Set xlim symmetric around x_c = 0.65: (-0.10, 1.40) -> midpoint is 0.65
    ax_b.set_xlim(-0.10, 1.40)
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.set_title(r"$\mathbf{(b)}$ Dual Collapse Limit ($L_k \to 0$)", pad=12, fontsize=15.5, fontweight="bold")
    ax_b.axis("off")

    # =========================================================================
    # PANEL (c): VNMM 2D Support Stencil
    # =========================================================================
    ax_c = fig.add_subplot(gs_top[0, 2])
    ax_c.set_facecolor(c_bg)

    tri_patch_c = patches.Polygon(vertices, closed=True, facecolor="#f8f9f9", edgecolor="#7f8c8d",
                                  linestyle="--", linewidth=2.0, zorder=2)
    ax_c.add_patch(tri_patch_c)

    P_eval = np.array([0.62, 0.52])
    circle_eval = patches.Circle(P_eval, 0.44, facecolor="#af7ac5", alpha=0.14, edgecolor="#8e44ad",
                                 linestyle=":", linewidth=2.2, zorder=3)
    ax_c.add_patch(circle_eval)
    ax_c.plot(P_eval[0], P_eval[1], 'X', color="#8e44ad", markersize=10.5, zorder=6)
    ax_c.text(P_eval[0] + 0.05, P_eval[1] - 0.03, r"Point $P$",
              fontsize=15.0, fontweight="bold", color="#6c3483")

    cores_nos = [c_edge1, c_edge2, c_edge3]
    labels_nos = [r"$(P_1, \mathbf{t}_1)$", r"$(P_2, \mathbf{t}_2)$", r"$(P_3, \mathbf{t}_3)$"]
    vec_len = 0.22

    for i in range(3):
        p_i = m[i]
        t_i = t[i]
        ax_c.plot(p_i[0], p_i[1], 'o', color=cores_nos[i], markersize=9.5, zorder=5)
        ax_c.annotate("", xy=p_i + vec_len * t_i, xytext=p_i,
                      arrowprops=dict(arrowstyle="-|>", color=cores_nos[i], lw=3.4, mutation_scale=20),
                      zorder=4)
        if i == 0:
            off_t = np.array([0.0, -0.13])
        elif i == 1:
            off_t = np.array([0.11, 0.02])
        else:
            off_t = np.array([-0.13, 0.02])
        ax_c.text(p_i[0] + off_t[0], p_i[1] + off_t[1], labels_nos[i],
                  fontsize=15.0, fontweight="bold", color=cores_nos[i], ha="center")

    ax_c.text(0.62, 0.94, r"Support $\Omega_x$",
              fontsize=14.5, color="#6c3483", fontstyle="italic", ha="center")

    ax_c.set_xlim(-0.16, 1.48)
    ax_c.set_ylim(-0.10, 1.40)
    ax_c.set_aspect('equal')
    ax_c.set_title(r"$\mathbf{(c)}$ VNMM 2D Vector Stencil", pad=12, fontsize=15.5, fontweight="bold")
    ax_c.axis("off")

    # =========================================================================
    # VECTOR FIELDS (BOTTOM ROW)
    # =========================================================================
    nx, ny = 45, 45
    x_grid = np.linspace(v1[0] - 0.02, v2[0] + 0.02, nx)
    y_grid = np.linspace(v1[1] - 0.02, v3[1] + 0.02, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    tri_path = Path(vertices)
    pts = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask = tri_path.contains_points(pts).reshape(X.shape)

    Wx, Wy = np.zeros_like(X), np.zeros_like(Y)
    Nx, Ny = np.zeros_like(X), np.zeros_like(Y)

    for iy in range(ny):
        for ix in range(nx):
            if inside_mask[iy, ix]:
                wx, wy = whitney_w1(X[iy, ix], Y[iy, ix])
                nx_val, ny_val = vnmm_N1(X[iy, ix], Y[iy, ix])
                Wx[iy, ix] = wx
                Wy[iy, ix] = wy
                Nx[iy, ix] = nx_val / L[0]
                Ny[iy, ix] = ny_val / L[0]
            else:
                Wx[iy, ix] = np.nan
                Wy[iy, ix] = np.nan
                Nx[iy, ix] = np.nan
                Ny[iy, ix] = np.nan

    W_mag = np.sqrt(Wx**2 + Wy**2)
    N_mag = np.sqrt(Nx**2 + Ny**2)
    norm_mag = Normalize(vmin=0.0, vmax=np.nanmax(W_mag))

    # =========================================================================
    # PANEL (d): Whitney 1-Form Vector Field w1(x, y)
    # =========================================================================
    ax_d = fig.add_subplot(gs_bottom[0, 0])
    ax_d.set_facecolor(c_bg)

    cf_d = ax_d.contourf(X, Y, W_mag, levels=25, cmap="YlGnBu", norm=norm_mag, alpha=0.88)
    ax_d.add_patch(patches.Polygon(vertices, closed=True, fill=False, edgecolor=c_tri, lw=2.4, zorder=5))

    step = 3
    ax_d.quiver(X[::step, ::step], Y[::step, ::step],
                Wx[::step, ::step], Wy[::step, ::step],
                color="#1b4f72", scale=16.0, width=0.0068, headwidth=4.2, headlength=5.2, zorder=4)

    ax_d.plot([v1[0], v2[0]], [v1[1], v2[1]], color=c_edge1, lw=5.0, zorder=6)
    ax_d.text(m[0][0], m[0][1] - 0.12, r"Active Edge $e_1$",
              fontsize=16.0, fontweight="bold", color=c_edge1, ha="center")

    ax_d.set_xlim(-0.02, 1.40)
    ax_d.set_ylim(-0.06, 1.30)
    ax_d.set_aspect('equal')
    ax_d.set_title(r"$\mathbf{(d)}$ FEM Whitney 1-Form $\mathbf{w}_1(\mathbf{x})$", pad=12, fontsize=15.5, fontweight="bold")
    ax_d.set_xlabel(r"$x$", fontsize=15.5)
    ax_d.set_ylabel(r"$y$", fontsize=15.5)
    ax_d.tick_params(labelsize=13.0)

    # =========================================================================
    # PANEL (e): VNMM L1 Vector Field (N1 / L1) with Attached Colorbar
    # =========================================================================
    ax_e = fig.add_subplot(gs_bottom[0, 1])
    ax_e.set_facecolor(c_bg)

    cf_e = ax_e.contourf(X, Y, N_mag, levels=25, cmap="YlGnBu", norm=norm_mag, alpha=0.88)
    ax_e.add_patch(patches.Polygon(vertices, closed=True, fill=False, edgecolor=c_tri, lw=2.4, zorder=5))

    step = 3
    ax_e.quiver(X[::step, ::step], Y[::step, ::step],
                Nx[::step, ::step], Ny[::step, ::step],
                color="#1b4f72", scale=16.0, width=0.0068, headwidth=4.2, headlength=5.2, zorder=4)

    ax_e.plot(m[0][0], m[0][1], 'o', color=c_edge1, markersize=10.0, zorder=7)
    ax_e.annotate("", xy=m[0] + 0.20 * t[0], xytext=m[0],
                  arrowprops=dict(arrowstyle="-|>", color=c_edge1, lw=3.4, mutation_scale=20), zorder=8)
    ax_e.text(m[0][0], m[0][1] - 0.12, r"Active Node $P_1$",
              fontsize=16.0, fontweight="bold", color=c_edge1, ha="center")

    ax_e.set_xlim(-0.02, 1.40)
    ax_e.set_ylim(-0.06, 1.30)
    ax_e.set_aspect('equal')
    ax_e.set_title(r"$\mathbf{(e)}$ VNMM 2D Basis $\frac{1}{L_1}\mathbf{N}_1(\mathbf{x})$", pad=12, fontsize=15.5, fontweight="bold")
    ax_e.set_xlabel(r"$x$", fontsize=15.5)
    ax_e.tick_params(labelsize=13.0)

    # Colorbar strictly bounded to ax_e
    divider = make_axes_locatable(ax_e)
    cax = divider.append_axes("right", size="5.5%", pad=0.10)
    cbar = fig.colorbar(cf_e, cax=cax)
    cbar.set_label(r"Field Magnitude $\|\mathbf{E}\|$", fontsize=15.0)
    cbar.ax.tick_params(labelsize=13.0)

    caminho_png = os.path.join(DIRETORIO_RELATORIOS, "figura_equivalencia_dual_whitney_vnmm.png")
    caminho_pdf = os.path.join(DIRETORIO_RELATORIOS, "figura_equivalencia_dual_whitney_vnmm.pdf")
    
    fig.savefig(caminho_png, dpi=300)
    fig.savefig(caminho_pdf)
    plt.close(fig)
    
    print(f"[SUCCESS] High-res PNG figure saved to: {caminho_png}")
    print(f"[SUCCESS] Vector PDF figure saved to: {caminho_pdf}")

if __name__ == "__main__":
    criar_figura_equivalencia_dual()
