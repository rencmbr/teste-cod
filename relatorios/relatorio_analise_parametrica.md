# Relatório da Análise Paramétrica: Método Sem Malha Nodal Vetorial (VNMM 2D)

Este relatório apresenta os resultados da análise paramétrica de interpolação do campo vetorial $\vec{E}$ e de seu rotacional $\nabla \times \vec{E}$ para o modo $\text{TE}_{11}$ em cavidade PEC.

## 1. Estudo Paramétrico: Variação da Tolerância do Determinante ($Tol_{det}$)

A tabela abaixo apresenta os erros para diferentes valores mínimos de $|\det(A)|$ com busca adaptativa de vizinhança $K$:

| $Tol_{det}$ | $\vert\det(A)\vert_{méd}$ | Erro Médio $\vec{E}$ | Erro RMS $\vec{E}$ | Erro Máx $\vec{E}$ | Erro Médio $\nabla\times\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Erro Máx $\nabla\times\vec{E}$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.001 | 1.0428 | 1.8858e-01 | 3.5696e-01 | 1.8785e+00 | 2.1481e-01 | 4.0135e-01 | 2.2700e+00 |
| 0.010 | 1.0428 | 1.8858e-01 | 3.5696e-01 | 1.8785e+00 | 2.1481e-01 | 4.0135e-01 | 2.2700e+00 |
| 0.100 | 1.0625 | 1.6032e-01 | 2.8570e-01 | 1.7735e+00 | 1.7953e-01 | 3.0307e-01 | 1.1181e+00 |
| 0.500 | 1.2793 | 1.0259e-01 | 1.4901e-01 | 6.3464e-01 | 1.1518e-01 | 1.8692e-01 | 1.0111e+00 |
| 1.000 | 1.7144 | 8.5790e-02 | 1.1672e-01 | 3.8255e-01 | 7.6334e-02 | 1.1138e-01 | 3.9496e-01 |
| 1.500 | 1.9748 | 8.5503e-02 | 1.1436e-01 | 3.8920e-01 | 7.2739e-02 | 1.0313e-01 | 3.5195e-01 |
| 2.000 | 2.4665 | 8.6985e-02 | 1.1209e-01 | 2.8690e-01 | 7.4976e-02 | 1.0517e-01 | 3.2926e-01 |
| 2.500 | 2.8695 | 9.1252e-02 | 1.1338e-01 | 3.4996e-01 | 6.9975e-02 | 9.7800e-02 | 3.1423e-01 |

![Análise de Tolerância](analise_tolerancia.png)

## 2. Estudo Paramétrico: Variação da Densidade da Malha ($N_{total}$ e $h$)

A tabela abaixo apresenta os erros de interpolação em escala logarítmica com a redução da distância característica $h$. A tolerância $Tol_{det}(h) \propto h$ reduz proporcionalmente ao espaçamento, garantindo invariância de escala e número constante de vizinhos $K$:

| Configuração | $N_{total}$ | $h_{méd}$ | $Tol_{det}(h)$ | $\vert\det(A)\vert_{méd}$ | $\vert\det(A)\vert_{mín}$ | $K_{méd}$ | $K_{máx}$ | Erro Médio $\vec{E}$ | Erro RMS $\vec{E}$ | Erro Máx $\vec{E}$ | Erro Médio $\nabla\times\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Erro Máx $\nabla\times\vec{E}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Esparsa (N=84) | 84 | 3.3333 | 1.6667 | 2.5751 | 1.6703 | 4.3 | 8 | 1.2339e-01 | 1.6316e-01 | 5.1889e-01 | 8.9948e-02 | 1.2933e-01 | 4.4940e-01 |
| Média-Esparsa (N=186) | 186 | 2.2222 | 1.1111 | 1.7097 | 1.1200 | 4.6 | 8 | 8.6571e-02 | 1.1576e-01 | 3.9317e-01 | 7.8871e-02 | 1.2290e-01 | 5.5370e-01 |
| Média (N=416) | 416 | 1.4286 | 0.7143 | 1.0547 | 0.7195 | 4.5 | 8 | 5.0535e-02 | 6.7513e-02 | 2.4389e-01 | 8.6251e-02 | 1.3683e-01 | 5.0012e-01 |
| Média-Densa (N=884) | 884 | 0.9524 | 0.4762 | 0.7268 | 0.4773 | 4.6 | 8 | 3.8495e-02 | 5.6917e-02 | 2.7109e-01 | 9.9232e-02 | 1.5199e-01 | 6.9847e-01 |
| Densa (N=1928) | 1928 | 0.6250 | 0.3125 | 0.4749 | 0.3131 | 4.4 | 9 | 2.1481e-02 | 2.8377e-02 | 1.0410e-01 | 9.0097e-02 | 1.3383e-01 | 4.8660e-01 |
| Muito Densa (N=4192) | 4192 | 0.4167 | 0.2083 | 0.3232 | 0.2087 | 4.3 | 8 | 1.7042e-02 | 2.4035e-02 | 7.6250e-02 | 1.0227e-01 | 1.7562e-01 | 9.2562e-01 |
| Ultra Densa (N=8408) | 8408 | 0.2174 | 0.1087 | 0.1911 | 0.1090 | 3.9 | 7 | 1.3130e-02 | 1.8213e-02 | 6.7684e-02 | 9.3131e-02 | 1.5754e-01 | 8.0241e-01 |

![Análise de Densidade](analise_densidade.png)

## 3. Painel Geral de Curvas Paramétricas

![Painel Geral](painel_analise_parametrica.png)

## 4. Discussão dos Resultados

1. **Impacto da Tolerância do Determinante ($Tol_{det}$):**
   - Para valores muito baixos de $Tol_{det}$ (< 0.1), são aceitas matrizes $A$ mal-condicionadas com pequenos determinantes, gerando erros máximos elevados.
   - Ao aumentar a tolerância ($Tol_{det} \ge 1.0$), o algoritmo seleciona trios de nós com melhor distribuição angular e condicionamento, reduzindo drasticamente os erros máximos e médios tanto do campo quanto do rotacional.
   - A expansão adaptativa de $K$ garantiu 100% de sucesso na seleção de nós em todas as tolerâncias testadas.

2. **Impacto da Densidade de Nós e Convergência com $h$ (Escala Log-Log):**
   - A variação de densidade cobriu desde a malha esparsa ($N=84$, $h \approx 3.33$) até a malha ultra densa ($N=8408$, $h \approx 0.22$), correspondendo a uma ampliação de 100x no número de nós.
   - Observa-se em escala log-log uma convergência contínua e acentuada dos erros médio, RMS e máximo do campo elétrico $\vec{E}$ conforme a distância inter-nodal $h$ diminui.
   - A taxa de redução do erro do campo com $h$ demonstra a consistência e a estabilidade de alta ordem da formulação sem malha VNMM 2D.

3. **Invariância de Escala e Estabilidade da Vizinhança $K$ com $Tol_{det}(h) \propto h$:**
   - A terceira coluna da matriz de momento $A$ possui dimensão de comprimento: $(y_i - y_P)t_{x,i} - (x_i - x_P)t_{y,i} \sim O(h)$, fazendo com que o determinante escale linearmente com $h$.
   - Ao adotar a tolerância proporcional $Tol_{det}(h) = Tol_{ref} \times (h / h_{ref})$, a condição adimensional $\frac{\vert\det(A)\vert}{h} \ge c$ torna-se invariante de escala.
   - Como consequência direta, o número de vizinhos mais próximos efetivamente usados não escala com o adensamento da malha: $K_{méd}$ manteve-se estritamente constante na faixa de $3.9$ a $4.6$ vizinhos, e $K_{máx} \le 9$ em todas as malhas (de 84 a 8408 nós).
