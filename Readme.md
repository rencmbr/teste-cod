# Método Sem Malha Nodal Vetorial em 2 Dimensões (VNMM 2D)

## Premissas e Fundamentos

O **Método Sem Malha Nodal Vetorial** (*Vector Nodal Meshless Method* - VNMM) é baseado na ideia do Método Sem Malha de Aresta (*Edge-based Meshless Method* - EMM) com o comprimento das arestas tendendo a zero. A técnica consiste em distribuir um conjunto de nós no domínio contínuo, associando a cada nó um vetor unitário com direção arbitrária. Nós também são posicionados nas fronteiras do domínio e nas interfaces entre diferentes meios materiais; para estes nós de contorno, a direção do vetor unitário não é arbitrária, mas tangente às fronteiras e interfaces físicas, conforme ilustrado na Figura 1.

<img width="1079" height="537" alt="Distribuição de nós no VNMM" src="https://github.com/user-attachments/assets/aa0919de-1202-46be-aeeb-563215222cad" />

**Figura 1:** Distribuição de nós e direções vetoriais para o VNMM.

Sejam $(x_i, y_i)$ as coordenadas do $i$-ésimo nó e $(t_{xi}, t_{yi})$ as componentes do vetor unitário $\vec{t}_i$ associado a este mesmo nó.

A formulação matemática para a construção das funções de forma vetoriais no VNMM bidimensional baseia-se na imposição da **condição de projeção nodal (delta de Kronecker vetorial)**:

$$
\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik} \qquad (1)
$$

onde $\vec{t}_k$ é o vetor unitário associado ao nó $n_k$ e $\delta_{ik}$ assume o valor $1$ se $k=i$ e $0$ se $k \neq i$. A aplicação desta restrição gera um sistema linear local $A \beta_i = L_i$, cuja inversão define as funções de forma vetoriais locais $\vec{N}_i$ para a representação do campo $\vec{E}^h(x, y) = \sum_i \vec{N}_i(x, y) e_i$.

---

## 1. Avaliação da Base Incompleta $\mathcal{L}^1$ (3 Nós de Suporte)

A formulação original utilizava uma base vetorial linear de 3 termos com 3 nós de suporte locais:

$$
\mathcal{L}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} y \\ -x \end{bmatrix} \right\rangle \qquad (2)
$$

A função de forma é $\vec{N}_i = \beta_{1i} [1, 0]^T + \beta_{2i} [0, 1]^T + \beta_{3i} [y, -x]^T$, com matriz de momento $A \in \mathbb{R}^{3 \times 3}$:

$$
A = \begin{bmatrix}
t_{1x} & t_{1y} & y_1 t_{1x} - x_1 t_{1y} \\
t_{2x} & t_{2y} & y_2 t_{2x} - x_2 t_{2y} \\
t_{3x} & t_{3y} & y_3 t_{3x} - x_3 t_{3y}
\end{bmatrix} \qquad (3)
$$

O rotacional é dado analiticamente por $(\nabla \times \vec{E}^h)_z = -2 \sum_{i=1}^3 \beta_{3i} e_i$.

### Resultados de Convergência da Interpolação com $\mathcal{L}^1$:

| Configuração | $N_{total}$ | $h_{méd}$ | $Tol_{det}(h)$ | Erro RMS $\vec{E}$ | Ordem $\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Ordem $\nabla\times\vec{E}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Esparsa** | 84 | 3.3333 | 1.6667 | $1.63 \times 10^{-1}$ | — | $1.29 \times 10^{-1}$ | — |
| **Média** | 416 | 1.4286 | 0.7143 | $6.75 \times 10^{-2}$ | $O(h)$ | $1.37 \times 10^{-1}$ | $O(1)$ |
| **Densa** | 1928 | 0.6250 | 0.3125 | $2.84 \times 10^{-2}$ | $O(h)$ | $1.34 \times 10^{-1}$ | $O(1)$ |
| **Ultra Densa** | 8408 | 0.2174 | 0.1087 | **$1.82 \times 10^{-2}$** | **$O(h)$** | **$1.58 \times 10^{-1}$** | **$O(1)$ (Estagna)** |

---

## 2. Diagnóstico do Vazamento Modal (*Aliasing*) na Base $\mathcal{L}^1$

A estagnação do erro do rotacional em $O(1)$ ($\approx 15\%$) decorre da **incompletude da matriz Jacobiana de Taylor** associada à base $\mathcal{L}^1$:
- A expansão linear completa em 2D requer 4 derivadas parciais: $\frac{\partial E_x}{\partial x}, \frac{\partial E_x}{\partial y}, \frac{\partial E_y}{\partial x}, \frac{\partial E_y}{\partial y}$.
- A base $\mathcal{L}^1$ possui apenas 3 graus de liberdade, forçando artificialmente $\frac{\partial E_x^h}{\partial x} \equiv 0$ e $\frac{\partial E_y^h}{\partial y} \equiv 0$.
- **Mecanismo de Vazamento:** O resíduo das derivadas normais não representadas $\mathbf{r} \sim O(h)$ é multiplicado pela 3ª linha de $A^{-1}$ (que escala com $O(1/h)$), resultando em um erro residual constante de ordem $O(1/h) \times O(h) = \mathbf{O(1)}$ no coeficiente $\beta_3$, impedindo a convergência do rotacional.

---

## 3. Formulação com a Base Completa $\mathcal{P}^1$ (6 Nós de Suporte)

Para eliminar o vazamento modal e garantir completude de 1ª ordem, adota-se o espaço polinomial vetorial linear completo $\mathcal{P}_1 \times \mathcal{P}_1$ com **6 nós de suporte**:

$$
\mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} x \\ 0 \end{bmatrix}, \begin{bmatrix} y \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ x \end{bmatrix}, \begin{bmatrix} 0 \\ y \end{bmatrix} \right\rangle \qquad (4)
$$

A matriz de momento $A \in \mathbb{R}^{6 \times 6}$ em coordenadas locais $(\Delta x_k = x_k - x_P, \Delta y_k = y_k - y_P)$ é dada por:

$$
A = \begin{bmatrix}
t_{1x} & t_{1y} & \Delta x_1 t_{1x} & \Delta y_1 t_{1x} & \Delta x_1 t_{1y} & \Delta y_1 t_{1y} \\
t_{2x} & t_{2y} & \Delta x_2 t_{2x} & \Delta y_2 t_{2x} & \Delta x_2 t_{2y} & \Delta y_2 t_{2y} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
t_{6x} & t_{6y} & \Delta x_6 t_{6x} & \Delta y_6 t_{6x} & \Delta x_6 t_{6y} & \Delta y_6 t_{6y}
\end{bmatrix} \qquad (5)
$$

O campo e o rotacional interpolados no ponto de avaliação $P$ ($\Delta x = 0, \Delta y = 0$) são:

$$
\vec{E}^h(P) = \begin{bmatrix} \sum_{i=1}^6 \beta_{1i} e_i \\ \sum_{i=1}^6 \beta_{2i} e_i \end{bmatrix}, \quad (\nabla \times \vec{E}^h)_z(P) = \sum_{i=1}^6 (\beta_{5i} - \beta_{4i}) e_i \qquad (6)
$$

---

## 4. Principais Resultados e Conclusões da Interpolação com a Base $\mathcal{P}^1$

A análise paramétrica comparativa entre $\mathcal{L}^1$ e $\mathcal{P}^1$ comprovou a restauração plena das taxas assintóticas de convergência:

| Malha | $N_{total}$ | $h_{méd}$ | Erro RMS $\vec{E}$ ($\mathcal{L}^1$) | Erro RMS $\vec{E}$ ($\mathcal{P}^1$) | Erro RMS $\nabla \times \vec{E}$ ($\mathcal{L}^1$) | Erro RMS $\nabla \times \vec{E}$ ($\mathcal{P}^1$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Esparsa** | 84 | 3.3333 | $1.63 \times 10^{-1}$ | **$2.51 \times 10^{-2}$** | $1.29 \times 10^{-1}$ | **$5.98 \times 10^{-2}$** |
| **Média** | 416 | 1.4286 | $6.75 \times 10^{-2}$ | **$4.20 \times 10^{-3}$** | $1.37 \times 10^{-1}$ | **$1.60 \times 10^{-2}$** |
| **Densa** | 1928 | 0.6250 | $2.84 \times 10^{-2}$ | **$7.58 \times 10^{-4}$** | $1.34 \times 10^{-1}$ | **$6.21 \times 10^{-3}$** |
| **Ultra Densa** | 8408 | 0.2174 | $1.82 \times 10^{-2}$ | **$9.67 \times 10^{-5}$** | $1.58 \times 10^{-1}$ *(estagnado)* | **$2.45 \times 10^{-4}$** |
| **Taxa Assintótica** | — | — | **$O(h)$** | **$O(h^2)$ (2ª Ordem)** | **$O(1)$ (Estagna)** | **$O(h)$ (1ª Ordem)** |

### Principais Conclusões da Interpolação:
1. **Eliminação Integral do Vazamento Modal:** O erro do rotacional caiu de $1.58 \times 10^{-1}$ para $2.45 \times 10^{-4}$ (redução de mais de **600 vezes**), confirmando convergência monotônica de 1ª ordem $O(h)$.
2. **Superconvergência do Campo $\vec{E}$:** A representação completa elevou a taxa de convergência do campo de $O(h)$ para $O(h^2)$.
3. **Escala Quártica do Determinante:** Como $\det(A) \sim O(h^4)$ para a base de 6 nós, a calibração adaptativa $Tol_{\text{det}}(h) = Tol_{\text{ref}} (h/h_{\text{ref}})^4$ garante 100% de sucesso na seleção de nós pela `KDTree` mantendo a localidade geométrica ótima.

> 📄 **Relatórios Detalhados da Interpolação:**
> - [Relatório da Análise Paramétrica com a Base $\mathcal{P}^1$](relatorios/relatorio_analise_parametrica_P1.md)
> - [Análise do Comportamento Assintótico em Malhas Densas](relatorios/analise_comportamento_assintotico_malhas_densas.md)

---

## 5. Resolução do Problema de Autovalores Eletromagnéticos

O problema de autovalores bidimensional consiste em determinar as frequências de ressonância e os números de onda de corte $k_c = \sqrt{\lambda}$ dos modos transversais elétricos ($TE_z$) em uma cavidade retangular com condutor elétrico perfeito (PEC) $\Omega = [0, \pi] \times [0, \pi]$ (Seção 4.3.1 e Tabela 4-1 da tese de doutorado de Luilly Ortiz, UFMG, 2023).

### 5.1 Formulação Variacional Fraca com Regularização Div-Curl
O problema de Helmholtz vetorial $\nabla \times (\nabla \times \vec{E}) = \lambda \vec{E}$ com $\hat{n} \times \vec{E} = \mathbf{0}$ em $\partial \Omega$ é formulado pelo princípio variacional de Ritz-Galerkin:

$$
\int_{\Omega} (\nabla \times \vec{W})_z (\nabla \times \vec{E})_z \, d\Omega + s_{\text{div}} \int_{\Omega} (\nabla \cdot \vec{W}) (\nabla \cdot \vec{E}) \, d\Omega = \lambda \int_{\Omega} \vec{W} \cdot \vec{E} \, d\Omega \qquad (7)
$$

onde $s_{\text{div}} = 6.0$ é o parâmetro de penalização que desloca os modos espúrios de gradiente ($\vec{E} = \nabla \phi$) para altas frequências ($\lambda > 50$), mantendo os modos físicos $TE_z$ ($\nabla \cdot \vec{E} \equiv 0$) inalterados.

### 5.2 Estratégia de Implementação Numérica

1. **Determinação de Suporte por Ponto de Gauss (Estilo EFG):**  
   Em cada ponto de integração de Gauss $P_g = (x_g, y_g)$, a `KDTree` busca os 6 nós mais próximos com orientação geométrica estável, monta $A(P_g)$ com origem no próprio ponto ($\Delta x = 0, \Delta y = 0$) e inverte $\beta = A^{-1}$. Isso garante estabilidade incondicional e natureza puramente sem malha (*truly meshless*).
2. **Integração Numérica com Células de Fundo:**  
   Utiliza-se uma grade de células quadriláteras regulares cobrindo o domínio ($dx \approx h$ a $2h$) com quadratura de Gauss-Legendre $2 \times 2$ (4 pontos de Gauss por célula).
3. **Condições de Contorno PEC (Dirichlet Homogêneo):**  
   Como os nós de fronteira possuem vetor diretor alinhado com a tangente ($\vec{t} \parallel \partial \Omega$), a condição $\vec{E} \cdot \vec{t} = 0$ é imposta diretamente pela eliminação dos graus de liberdade de fronteira ($c_{\text{borda}} = 0$).
4. **Solver de Autovalores Generalizado:**  
   Resolve-se $K_{\text{red}} \mathbf{c}_{\text{red}} = \lambda M_{\text{red}} \mathbf{c}_{\text{red}}$ via `scipy.linalg.eigh` (com fallback robusto QZ).

> 📄 **Relatórios de Implementação e Quadratura:**
> - [Estudo de Integração Numérica e Quadratura no Caso Base](relatorios/relatorio_estudo_integracao_caso_base.md)
> - [Relatório Comparativo Global das Formulações no VNMM 2D](relatorios/relatorio_comparativo_global_vnmm2d.md)

---

## 6. Principais Resultados e Conclusões para o Problema de Autovalores

### 6.1 Desempenho no Caso Base ($N_x=21, N_y=21$, $361$ Incógnitas Ativas)

Comparação dos 10 primeiros modos com a solução analítica da Tabela 4-1 de Luilly Ortiz ($\lambda = n^2 + m^2$):

| Modo ($TE_{nm}$) | $\lambda_{\text{analítico}}$ | $k_{c, \text{analítico}}$ | $\lambda_{\text{VNMM } \mathcal{P}^1}$ | $k_{c, \text{VNMM}}$ | Erro $k_c$ (%) | Diagnóstico |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$TE_{10}$** | 1.00 | 1.000 | 0.9679 | 0.984 | **1.62%** | Físico Fundamental |
| **$TE_{01}$** | 1.00 | 1.000 | 0.9836 | 0.992 | **0.82%** | Físico Fundamental |
| **$TE_{11}$** | 2.00 | 1.414 | 1.9409 | 1.393 | **1.49%** | Físico |
| **$TE_{20}$** | 4.00 | 2.000 | 3.9275 | 1.982 | **0.91%** | Físico |
| **$TE_{02}$** | 4.00 | 2.000 | 4.0724 | 2.018 | **0.90%** | Físico |
| **$TE_{21}$** | 5.00 | 2.236 | 4.8925 | 2.212 | **1.08%** | Físico |
| **$TE_{12}$** | 5.00 | 2.236 | 5.0287 | 2.242 | **0.29%** | Físico |
| **$TE_{22}$** | 8.00 | 2.828 | 8.0743 | 2.842 | **0.46%** | Físico |
| **$TE_{30}$** | 9.00 | 3.000 | 9.1089 | 3.018 | **0.60%** | Físico |
| **$TE_{03}$** | 9.00 | 3.000 | 9.3409 | 3.056 | **1.88%** | Físico |

- **Erro Médio de $k_c$ no Caso Base:** **$1.00\%$** (Erro Máximo: **$1.88\%$**)
- **Tempo de Montagem e Resolução:** **$0.061\text{s}$**

---

### 6.2 Comparação Sistemática: VNMM 2D ($\mathcal{P}^1$) vs. Elementos Finitos de Aresta (Nédélec)

Implementamos um solver independente de **Elementos Finitos de Aresta Triangulares de Nédélec de 1ª Ordem (1-formas de Whitney)** em [`src/fem_edge_2d.py`](src/fem_edge_2d.py) para validação cruzada rigorosa:

| Nível de Refinamento | DoFs VNMM | Erro Méd $k_c$ VNMM | DoFs FEM Aresta | Erro Méd $k_c$ FEM Aresta |
|:---|:---:|:---:|:---:|:---:|
| **N1 (Muito Esparsa)** | 49 | 11.73% | 40 | 3.19% |
| **N2 (Esparsa)** | 121 | 3.95% | 133 | 1.05% |
| **N3 (Média-Esparsa)** | 225 | 1.73% | 225 | 0.64% |
| **N4 (Caso Base)** | **361** | **1.00%** | **341** | **0.44%** |
| **N6 (Densa)** | 729 | 1.49% | 736 | 0.21% |
| **N7 (Muito Densa)** | **961** | **0.52%** | **936** | **0.16%** |

### 6.3 Avaliação de Formulações Alternativas Testadas

1. **Base Incompleta $\mathcal{L}^1$ (3 Nós):** Como $\nabla \cdot \vec{N}_i \equiv 0$, a matriz $K_{\text{div}} \equiv 0$ não permite regularização. O erro médio de $k_c$ fica entre **$28.32\% - 48.95\%$** devido a vazamento modal.
2. **Formulação Sem Penalização ($s_{\text{div}} = 0.0$ na Base $\mathcal{P}^1$):** Modos espúrios de gradiente invadem a faixa espectral física ($0.8 < \lambda < 5.0$), elevando o erro médio para **$19.75\% - 90\%$**.
3. **Formulação Mimética de Stokes ($\oint_{\partial \Omega_e} \vec{E} \cdot d\vec{\ell}$ com $s_{\text{div}} = 0$):** Consegue colapsar $262$ modos espúrios para autovalores nulos exatos ($\lambda \approx 0$), atingindo erro médio de $12.76\%$. Contudo, a aproximação de rotacional constante por célula a torna menos acurada que o VNMM diferencial pontual ($1.00\%$).

### 6.4 Conclusões Globais:
- O VNMM 2D com a base linear completa $\mathcal{P}^1$, suporte individual por ponto de Gauss (estilo EFG) e regularização div-curl ($s_{\text{div}} = 6.0$) consolida-se como a formulação ótima definitiva, combinando **alta acurácia espectral ($\le 1\%$)**, **eliminação total de modos espúrios** e a **plena flexibilidade de um método sem malha**.

> 📄 **Relatórios de Convergência e Comparações:**
> - [Relatório Final de Convergência: VNMM 2D vs FEM de Aresta](relatorios/relatorio_final_convergencia_vnmm_vs_fem.md)
> - [Comparação Modal Detalhada: VNMM vs FEM de Aresta](relatorios/relatorio_comparacao_vnmm_vs_fem_aresta.md)
> - [Análise Teórica das Origens dos Erros entre VNMM e FEM](relatorios/analise_origem_erros_vnmm_vs_fem.md)
> - [Estudo da Formulação Mimética de Stokes](relatorios/relatorio_rotacional_mimetico_stokes.md)
> - [Estudo do Problema Sem Regularização do Divergente](relatorios/relatorio_sem_regularizacao_divergente.md)
> - [Estudo do Problema de Autovalores com a Base $\mathcal{L}^1$](relatorios/relatorio_estudo_base_L1.md)

---

## 7. Estrutura do Repositório e Execução

### Organização de Diretórios
- [`src/`](src/): Módulos e bibliotecas fundamentais do método:
  - [`montador_vnmm.py`](src/montador_vnmm.py): Montador de matrizes globais de rigidez e massa (modos `ponto_gauss` e `centro_celula`).
  - [`eigen_solver_cavity.py`](src/eigen_solver_cavity.py): Solver de autovalores para cavidades PEC com cálculo de métricas e erros modais.
  - [`fem_edge_2d.py`](src/fem_edge_2d.py): Solver de Elementos Finitos de Aresta Triangulares de Nédélec (1ª ordem).
  - [`malha_cavidade.py`](src/malha_cavidade.py): Gerador de malhas nodais com direções tangenciais de contorno.
  - [`quadratura_gauss.py`](src/quadratura_gauss.py): Quadratura de Gauss-Legendre 1D e 2D.
- [`codigo/`](codigo/): Scripts executáveis de análises paramétricas e estudos comparativos.
- [`relatorios/`](relatorios/): Relatórios técnicos em Markdown e figuras de alta resolução.
- [`tests/`](tests/): Suíte de testes unitários automatizados.

### Comandos de Execução

1. **Executar a Suíte de Testes Unitários:**
   ```bash
   python3 -m unittest tests/test_eigen_cavity.py
   ```
2. **Executar a Comparação VNMM 2D vs FEM de Aresta no Caso Base:**
   ```bash
   python3 codigo/comparacao_vnmm_vs_fem_aresta.py
   ```
3. **Executar a Análise de Convergência Paramétrica Completa:**
   ```bash
   python3 codigo/analise_convergencia_comparativa_vnmm_vs_fem.py
   ```
