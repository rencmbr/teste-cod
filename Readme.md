# Método Sem Malha Nodal Vetorial em 2 Dimensões e 3 Nós de Suporte

## Premissas

O Método Sem Malha Nodal Vetorial (*Vector Nodal Meshless Method* - VNMM) é baseado na ideia do Método Sem Malha de Aresta (EMM) com o comprimento das arestas tendendo a zero. A ideia consiste em distribuir um conjunto de nós no domínio, associando a cada nó um vetor unitário com direção arbitrária. Nós também são posicionados nas fronteiras do domínio e nas interfaces entre diferentes materiais; para estes nós de contorno, a direção do vetor unitário não é arbitrária, mas tangente às fronteiras e interfaces, conforme ilustrado na Figura 1.

<img width="1079" height="537" alt="image" src="https://github.com/user-attachments/assets/aa0919de-1202-46be-aeeb-563215222cad" />

**Figura 1:** Distribuição de nós e direções vetoriais para o VNMM.

Sejam $(x_i, y_i)$ as coordenadas do $i$-ésimo nó e $(t_{xi}, t_{yi})$ as componentes do vetor unitário associado a este mesmo nó.

A formulação matemática para a construção das funções de forma vetoriais utilizando três nós de suporte no VNMM bidimensional considera um polinômio de ordem igual a 1:

$$
\mathcal{L}^1 = \left\langle \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}, \begin{bmatrix} y \\\\ -x \end{bmatrix} \right\rangle \qquad (1)
$$

A partir desta base vetorial, a função de forma $\vec{N}_i$ associada ao $i$-ésimo nó é expressa como uma combinação linear de seus termos componentes:

$$
\vec{N}_i = \beta_{1i} \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} + \beta_{2i} \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} + \beta_{3i} \begin{bmatrix} y \\\\ -x \end{bmatrix} \qquad (2)
$$

Nesta expressão, $\beta_{1i}$, $\beta_{2i}$ e $\beta_{3i}$ representam os coeficientes incógnitos da interpolação a serem determinados. Como há três coeficientes, utilizam-se três nós de suporte no domínio local.

Para garantir a coerência física e matemática da aproximação em $H(\text{curl})$, impõe-se que a função de forma $\vec{N}_i$ possua projeção não nula exclusivamente na direção do vetor associado ao seu respectivo nó. Consequentemente, impõe-se a condição de projeção em que a $k$-ésima função de forma tenha projeção igual a 1 na direção do vetor unitário de seu próprio nó e 0 nas direções dos vetores associados aos demais nós de suporte. Esta restrição, que traduz a propriedade do delta de Kronecker à formulação vetorial, é definida por:

$$
\vec{N}_i \cdot \vec{t}_k = \delta_{ik} \qquad (3)
$$

onde $\vec{t}_k$ é o vetor unitário associado ao nó $n_k$ e $\delta_{ik}$ assume o valor 1 se $k=i$ e 0 se $k \neq i$. A aplicação dessa condição resulta nos sistemas lineares:

$$
A \beta_i = L_i, \quad \text{para } i = 1, 2, 3 \qquad (4)
$$

A matriz de interpolação $A$, os vetores de coeficientes locais $\beta_i$ e os vetores canônicos $L_i$ são definidos por:

$$
A = \begin{bmatrix}
t_{1x} & t_{1y} & y_1 t_{1x} - x_1 t_{1y} \\\\
t_{2x} & t_{2y} & y_2 t_{2x} - x_2 t_{2y} \\\\
t_{3x} & t_{3y} & y_3 t_{3x} - x_3 t_{3y}
\end{bmatrix} \qquad (5)
$$

$$
\beta_i = \begin{bmatrix} \beta_{1i} \\\\ \beta_{2i} \\\\ \beta_{3i} \end{bmatrix} \qquad (6)
$$

$$
L_1 = \begin{bmatrix} 1 \\\\ 0 \\\\ 0 \end{bmatrix}, \quad
L_2 = \begin{bmatrix} 0 \\\\ 1 \\\\ 0 \end{bmatrix}, \quad
L_3 = \begin{bmatrix} 0 \\\\ 0 \\\\ 1 \end{bmatrix} \qquad (7)
$$

Nestes sistemas, $t_{kx}$ e $t_{ky}$ correspondem às componentes cartesianas do vetor unitário de direção atrelado ao $k$-ésimo nó de suporte.

Uma vez determinados os coeficientes $\beta_i$ ($i=1,2,3$), as funções de forma $\vec{N}_i$ estarão determinadas para o domínio de suporte e a aproximação de uma função vetorial $\vec{E}$ no domínio de suporte é dada por:

$$
\vec{E}^h = \sum_{i=1}^3 \vec{N}_i e_i = \Phi(x,y) e_s \qquad (8)
$$

onde $e_s$ é o vetor com as projeções de $\vec{E}$ na direção de cada vetor unitário $\vec{t}_i$ e $\Phi(x,y)$ é a matriz de funções de forma:

$$
\Phi(x,y) = \begin{bmatrix} \vec{N}_1 & \vec{N}_2 & \vec{N}_3 \end{bmatrix}, \quad e_s = \begin{bmatrix} e_1 \\\\ e_2 \\\\ e_3 \end{bmatrix} \qquad (9)
$$

O rotacional da aproximação, $\nabla \times \vec{E}^h$, é dado por:

$$
\nabla \times \vec{E}^h = \begin{bmatrix} \nabla \times \vec{N}_1 & \nabla \times \vec{N}_2 & \nabla \times \vec{N}_3 \end{bmatrix} \begin{bmatrix} e_1 \\\\ e_2 \\\\ e_3 \end{bmatrix} \qquad (10)
$$

com:

$$
\nabla \times \vec{N}_i = \beta_{1i} \nabla \times \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} + \beta_{2i} \nabla \times \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} + \beta_{3i} \nabla \times \begin{bmatrix} y \\\\ -x \end{bmatrix} \qquad (11)
$$

Como o rotacional aplicado a vetores constantes é nulo, ele é não nulo apenas para o último termo da base:

$$
\nabla \times \vec{N}_i = \begin{bmatrix} 0 \\\\ 0 \\\\ -2\beta_{3i} \end{bmatrix} \qquad (12)
$$

Logo, o rotacional da aproximação se reduz a:

$$
\nabla \times \vec{E}^h = \begin{bmatrix} 0 \\\\ 0 \\\\ -2 \sum_{i=1}^3 \beta_{3i} e_i \end{bmatrix} \qquad (13)
$$

É fundamental garantir que as aproximações $\vec{E}^h$ e $\nabla \times \vec{E}^h$ sejam as melhores possíveis.

---

## 1. Análise Paramétrica da Formulação com Base $\mathcal{L}^1$ (3 Nós de Suporte)

Realizou-se uma análise paramétrica sistemática avaliando a interpolação do campo vetorial $\vec{E}$ e de seu rotacional $\nabla \times \vec{E}$ para o modo $\text{TE}_{11}$ em cavidade ressonante bidimensional com condutor elétrico perfeito (PEC), considerando:

1. **Tolerância do Determinante ($Tol_{det}$):** Adoção de um algoritmo adaptativo de vizinhança $K$ que expande a busca local caso nenhum trio de nós satisfaça $|\det(A)| \ge Tol_{det}$, assegurando 100% de sucesso na seleção de nós e eliminando configurações quase-singulares.
2. **Densidade de Nós ($N_{total}$) e Escala Proporcional $Tol_{det}(h) \propto h$:** Como a terceira coluna da matriz $A$ possui dimensão de comprimento, dada pela expressão $(y_i - y_P)t_{xi} - (x_i - x_P)t_{yi} \sim O(h)$, o determinante $\det(A)$ escala linearmente com $O(h)$. Ao fixar $Tol_{det}(h) = Tol_{ref} \cdot (h / h_{ref})$, a qualidade geométrica adimensional $|\det(A)|/h$ mantém-se constante, fazendo com que o número de vizinhos efetivos permaneça estritamente confinado em $K_{méd} \approx 4 - 5$ e $K_{máx} \le 9$ ao longo de toda a variação de densidade (de 84 a 8408 nós, fator de $100\times$).

### Resultados de Convergência Obtidos:

| Configuração | $N_{total}$ | $h_{méd}$ | $Tol_{det}(h)$ | $\vert\det(A)\vert_{méd}$ | $K_{méd}$ | Erro RMS $\vec{E}$ | Ordem $\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Ordem $\nabla\times\vec{E}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Esparsa** | 84 | 3.3333 | 1.6667 | 2.5751 | 4.3 | $1.63 \times 10^{-1}$ | — | $1.29 \times 10^{-1}$ | — |
| **Média-Esparsa** | 186 | 2.2222 | 1.1111 | 1.7097 | 4.6 | $1.16 \times 10^{-1}$ | $O(h)$ | $1.23 \times 10^{-1}$ | $O(1)$ |
| **Média** | 416 | 1.4286 | 0.7143 | 1.0547 | 4.5 | $6.75 \times 10^{-2}$ | $O(h)$ | $1.37 \times 10^{-1}$ | $O(1)$ |
| **Média-Densa** | 884 | 0.9524 | 0.4762 | 0.7268 | 4.6 | $5.69 \times 10^{-2}$ | $O(h)$ | $1.52 \times 10^{-1}$ | $O(1)$ |
| **Densa** | 1928 | 0.6250 | 0.3125 | 0.4749 | 4.4 | $2.84 \times 10^{-2}$ | $O(h)$ | $1.34 \times 10^{-1}$ | $O(1)$ |
| **Muito Densa** | 4192 | 0.4167 | 0.2083 | 0.3232 | 4.3 | $2.40 \times 10^{-2}$ | $O(h)$ | $1.76 \times 10^{-1}$ | $O(1)$ |
| **Ultra Densa** | 8408 | 0.2174 | 0.1087 | 0.1911 | 3.9 | **$1.82 \times 10^{-2}$** | **$O(h)$** | **$1.58 \times 10^{-1}$** | **$O(1)$ (Estagna)** |

Observa-se que:
- O erro da função vetorial $\vec{E}^h$ reduziu em praticamente uma ordem de grandeza ($1.63 \times 10^{-1} \to 1.82 \times 10^{-2}$), exibindo **convergência contínua de ordem $O(h)$**.
- O erro do rotacional $\nabla \times \vec{E}^h$ **não convergiu**, permanecendo estagnado na faixa de $0.12 - 0.17$ ($O(1)$), independente do refinamento da malha.

---

## 2. Diagnóstico Físico-Matemático: Por que o Rotacional Estagna na Base $\mathcal{L}^1$?

A causa raiz da estagnação do rotacional reside na **incompletude da base polinomial local $\mathcal{L}^1$** em relação à expansão em Série de Taylor do campo físico real:

### 2.1 Incompletude da Matriz Jacobiana
A expansão de Taylor de 1ª ordem do campo real $\vec{E}$ em torno do ponto de avaliação $P=(0,0)$ requer as 4 derivadas espaciais da matriz Jacobiana:

$$
\mathbf{J} = \begin{bmatrix} \frac{\partial E_x}{\partial x} & \frac{\partial E_x}{\partial y} \\\\ \frac{\partial E_y}{\partial x} & \frac{\partial E_y}{\partial y} \end{bmatrix} \qquad (14)
$$

Contudo, a base $\mathcal{L}^1 = \left\langle [1, 0]^T, [0, 1]^T, [y, -x]^T \right\rangle$ possui apenas 3 graus de liberdade, impondo restrições rígidas:
- Representa o campo constante $\vec{E}(P) = [c_1, c_2]^T$ (2 GDL);
- Representa o rotacional $(\nabla \times \vec{E}^h)_z = -2c_3$ (1 GDL);
- **Força artificialmente:** $\frac{\partial E_x^h}{\partial x} \equiv 0$ e $\frac{\partial E_y^h}{\partial y} \equiv 0$.

Para o modo $\text{TE}_{11}$, as derivadas normais são não nulas ($\frac{\partial E_x}{\partial x} = -\frac{\pi}{20}\sin\dots \ne 0$).

### 2.2 O Mecanismo do Vazamento Modal (*Aliasing*)
Ao resolver o sistema de colocação nodal $A \mathbf{c} = \mathbf{e}_s$, o vetor de dados reais $\mathbf{e}_s$ contém contribuições das derivadas normais não representadas $\mathbf{r} \sim O(h)$.

Pela inversão do sistema $\mathbf{c} = A^{-1}\mathbf{e}_s$:
- O resíduo físico não representado $\mathbf{r}$ escala com $O(h)$;
- Como a 3ª coluna de $A$ é de ordem $O(h)$, a 3ª linha de $A^{-1}$ é de ordem $O(1/h)$.

O coeficiente $c_3 = \sum_{i=1}^3 \beta_{3i}e_i$, responsável pelo rotacional, sofre uma contaminação direta:

$$
\text{Vazamento em } c_3 = (\text{3ª linha de } A^{-1}) \cdot \mathbf{r} \sim O\left(\frac{1}{h}\right) \times O(h) = \mathbf{O(1) \quad (Constante)} \qquad (15)
$$

Como o rotacional aproximado é $(\nabla \times \vec{E}^h)_z = -2c_3$, ele herda esse erro residual de ordem $O(1)$ que não desaparece com o refinamento $h$.

### 2.3 Comparação com Elementos Finitos de Aresta de Nédélec
Nos Elementos Finitos de Aresta de Nédélec, os graus de liberdade são **integrais de circulação ao longo do contorno fechado $\partial T$ das arestas do elemento**:

$$
(\nabla \times \vec{E}^h)_z = \frac{1}{\text{Área}(T)} \oint_{\partial T} \vec{E} \cdot d\vec{\ell} \qquad (16)
$$

Pelo **Teorema de Stokes**, a circulação de qualquer campo gradiente/conservativo $\nabla \phi$ (que contém as derivadas normais $\frac{\partial E_x}{\partial x}, \frac{\partial E_y}{\partial y}$) ao longo de um contorno fechado é **identicamente nula**:

$$
\oint_{\partial T} \nabla \phi \cdot d\vec{\ell} \equiv 0 \qquad (17)
$$

No VNMM, por ser um método puramente sem malha (*meshless*) baseado em colocações nodais pontuais discretas, essa anulação topológica de contorno fechado não ocorre automaticamente.

---

## 3. Formulação com Base Completa $\mathcal{P}^1$ e Colocação em 6 Nós de Suporte

Para eliminar o vazamento modal e garantir a convergência estrita tanto do campo quanto do rotacional no contexto sem malha nodal, adota-se o **espaço polinomial vetorial linear completo $\mathcal{P}_1 \times \mathcal{P}_1$ (6 termos)** utilizando **6 nós de suporte** no domínio local de colocação:

$$
\mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}, \begin{bmatrix} x \\\\ 0 \end{bmatrix}, \begin{bmatrix} y \\\\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\\\ x \end{bmatrix}, \begin{bmatrix} 0 \\\\ y \end{bmatrix} \right\rangle \qquad (18)
$$

Em coordenadas locais $(\Delta x_k = x_k - x_P, \Delta y_k = y_k - y_P)$ em torno do ponto de avaliação $P$, a função de forma $\vec{N}_i$ associada ao $i$-ésimo nó de suporte é dada pela combinação linear dos 6 termos da base:

$$
\vec{N}_i(x, y) = \beta_{1i} \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} + \beta_{2i} \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} + \beta_{3i} \begin{bmatrix} x \\\\ 0 \end{bmatrix} + \beta_{4i} \begin{bmatrix} y \\\\ 0 \end{bmatrix} + \beta_{5i} \begin{bmatrix} 0 \\\\ x \end{bmatrix} + \beta_{6i} \begin{bmatrix} 0 \\\\ y \end{bmatrix} \qquad (19)
$$

onde $\beta_i = [\beta_{1i}, \beta_{2i}, \beta_{3i}, \beta_{4i}, \beta_{5i}, \beta_{6i}]^T$ é o vetor de coeficientes locais incógnitos.

### 3.1 Condição de Projeção e Sistema Linear $6 \times 6$

Impondo a condição de colocação nodal (delta de Kronecker vetorial) nos 6 nós de suporte:

$$
\vec{N}_i(x_k, y_k) \cdot \vec{t}_k = \delta_{ik}, \quad \text{para } i, k = 1, 2, \dots, 6 \qquad (20)
$$

onde $\vec{t}_k = [t_{kx}, t_{ky}]^T$ é o vetor unitário atrelado ao $k$-ésimo nó de suporte, obtém-se o sistema linear:

$$
A \beta_i = L_i, \quad \text{para } i = 1, 2, \dots, 6 \qquad (21)
$$

onde $L_i$ é a $i$-ésima coluna da matriz identidade de dimensão 6 ($I_6$). A matriz de coeficientes locais é dada diretamente por $\beta = [\beta_1, \dots, \beta_6] = A^{-1}$.

A **matriz de interpolação de momento $A \in \mathbb{R}^{6 \times 6}$** toma a seguinte forma explícita:

$$
A = \begin{bmatrix}
t_{1x} & t_{1y} & (x_1 - x_P)t_{1x} & (y_1 - y_P)t_{1x} & (x_1 - x_P)t_{1y} & (y_1 - y_P)t_{1y} \\\\
t_{2x} & t_{2y} & (x_2 - x_P)t_{2x} & (y_2 - y_P)t_{2x} & (x_2 - x_P)t_{2y} & (y_2 - y_P)t_{2y} \\\\
t_{3x} & t_{3y} & (x_3 - x_P)t_{3x} & (y_3 - y_P)t_{3x} & (x_3 - x_P)t_{3y} & (y_3 - y_P)t_{3y} \\\\
t_{4x} & t_{4y} & (x_4 - x_P)t_{4x} & (y_4 - y_P)t_{4x} & (x_4 - x_P)t_{4y} & (y_4 - y_P)t_{4y} \\\\
t_{5x} & t_{5y} & (x_5 - x_P)t_{5x} & (y_5 - y_P)t_{5x} & (x_5 - x_P)t_{5y} & (y_5 - y_P)t_{5y} \\\\
t_{6x} & t_{6y} & (x_6 - x_P)t_{6x} & (y_6 - y_P)t_{6x} & (x_6 - x_P)t_{6y} & (y_6 - y_P)t_{6y}
\end{bmatrix} \qquad (22)
$$

### 3.2 Interpolação do Campo e do Rotacional no Ponto $P$

No ponto de avaliação $P$ (origem do sistema local $\Delta x = 0, \Delta y = 0$):

#### 1. Funções de forma no ponto $P$:

$$
\vec{N}_i(P) = \begin{bmatrix} \beta_{1i} \\\\ \beta_{2i} \end{bmatrix} \qquad (23)
$$

$$
\Phi(P) = \begin{bmatrix} \vec{N}_1(P) & \vec{N}_2(P) & \dots & \vec{N}_6(P) \end{bmatrix} = \begin{bmatrix} \beta_{11} & \beta_{12} & \dots & \beta_{16} \\\\ \beta_{21} & \beta_{22} & \dots & \beta_{26} \end{bmatrix} \qquad (24)
$$

#### 2. Campo vetorial interpolado:

$$
\vec{E}^h(P) = \sum_{i=1}^6 \vec{N}_i(P) e_i = \Phi(P) e_s \qquad (25)
$$

onde $e_s = [e_1, e_2, \dots, e_6]^T$, com $e_i = \vec{E}(\mathbf{x}_i) \cdot \vec{t}_i$.

#### 3. Rotacional das funções de forma e do campo:

Calculando o rotacional de cada termo da base polinomial:

$$
\nabla \times \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} = \vec{0}, \quad
\nabla \times \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} = \vec{0}, \quad
\nabla \times \begin{bmatrix} x \\\\ 0 \end{bmatrix} = \vec{0} \qquad (26)
$$

$$
\nabla \times \begin{bmatrix} y \\\\ 0 \end{bmatrix} = -\hat{z}, \quad
\nabla \times \begin{bmatrix} 0 \\\\ x \end{bmatrix} = +\hat{z}, \quad
\nabla \times \begin{bmatrix} 0 \\\\ y \end{bmatrix} = \vec{0} \qquad (27)
$$

Logo, o rotacional de cada função de forma $\vec{N}_i$ é dado por:

$$
\nabla \times \vec{N}_i = \begin{bmatrix} 0 \\\\ 0 \\\\ \beta_{5i} - \beta_{4i} \end{bmatrix} \qquad (28)
$$

E o rotacional da aproximação no ponto $P$ resulta em:

$$
\nabla \times \vec{E}^h(P) = \begin{bmatrix} 0 \\\\ 0 \\\\ \sum_{i=1}^6 (\beta_{5i} - \beta_{4i}) e_i \end{bmatrix} \qquad (29)
$$

### 3.3 Propriedades e Taxas Assintóticas da Colocação com $\mathcal{P}^1$
- **Representação Completa:** As 4 derivadas da Jacobiana ($\frac{\partial E_x}{\partial x} = \beta_{3}$, $\frac{\partial E_x}{\partial y} = \beta_{4}$, $\frac{\partial E_y}{\partial x} = \beta_{5}$ e $\frac{\partial E_y}{\partial y} = \beta_{6}$) são resolvidas independentemente.
- **Ausência de Vazamento Modal:** Nenhuma derivada física é forçada a zero, eliminando o erro de projeção $O(1)$.
- **Ordens de Convergência:**
  - Campo vetorial $\vec{E}^h$: **Convergência de 2ª ordem $O(h^2)$**;
  - Rotacional $\nabla \times \vec{E}^h$: **Convergência de 1ª ordem $O(h)$**.
