# Método sem malha nodal vetorial em 2 dimensões e 3 nós de suporte

# Premissas:

O Método Sem Malha Nodal Vetorial, Vector Nodal Meshless Method, VNMM,  é baseado na ideia do Método Sem Malha de Aresta , EMM, com o comprimento das arestas tendendo para zero. A ideia é espalhar um conjunto de nós no domı́nio, sendo que para cada nó é associado um vetor unitário com direção arbitrária. Nós também são espalhados nas fronteiras do domínio e na interface entre diferentes materiais. Para estes nós, a direção vetorial do vetor unitário não é mais arbitrária, mas tangente às fronteiras e interfaces, conforme mostrado na figura 1\. 

<img width="1079" height="537" alt="image" src="https://github.com/user-attachments/assets/aa0919de-1202-46be-aeeb-563215222cad" />


Figura 1: Distribuição de nós e direções vetoriais para o VNMM

Sejam $(x\_i, y\_i)$  as coordenadas do $i\_{\\acute{e}simo}$ nó e $$(t\_{xi}, t\_{yi})$$ as componentes do vetor unitário associado a este mesmo nó.

A formulação matemática para a construção das funções de forma vetoriais utilizando três nós de suporte no Método Sem Malha Nodal Vetorial (VNMM) bidimensional considera um polinômio de ordem igual a 1:

$$\\mathcal{L}^{1} \= \\left\\langle \\begin{bmatrix} 1 \\  
0 \\end{bmatrix}, \\begin{bmatrix} 0 \\ 
1 \\end{bmatrix}, \\begin{bmatrix} y \\ 
\-x \\end{bmatrix} \\right\\rangle$$

A partir desta base vetorial, a função de forma $\\vec{N}\_{i}$  associada ao $i\_{\\acute{e}simo}$ nó é expressa como uma combinação linear de seus termos componentes:

$$\\vec{N}\_{i} \= \\beta\_{1i}\\begin{bmatrix} 1\\\\
0 \\end{bmatrix} \+ \\beta\_{2i}\\begin{bmatrix} 0 \\\\  
1 \\end{bmatrix} \+ \\beta\_{3i}\\begin{bmatrix} y \\\\  
\-x \\end{bmatrix}$$

Nesta expressão, $$\\beta\_{1i}$$, $$\\beta\_{2i}$$ e $$\\beta\_{3i}$$ representam os coeficientes incógnitos da interpolação que necessitam ser determinados. Tendo em vista que há exatamente três coeficientes a serem encontrados, utilizam-se três nós de suporte no domínio local.

Para garantir a coerência física e matemática da aproximação em $$H(curl)$$, é imperativo que a função de forma $$\\vec{N}\_{i}$$ *possua componente tangencial exclusivamente na direção do vetor associado ao seu respectivo nó. Consequentemente, impõe-se uma condição de projeção determinando que a késima função de forma tenha projeção igual a 1 na direção do vetor unitário de seu próprio nó e igual a 0 nas direções dos vetores associados aos demais nós de suporte. Esta restrição, que traduz a propriedade de delta de Kronecker à formulação vetorial, é definida algebricamente pelo produto escalar:

$$\\vec{N}\_{i} \\cdot \\vec{t}\_{k} \= \\delta\_{ik} $$

onde $$\\vec{t}\_{k}$$ *é o vetor unitário associado ao nó $$n\_k$$ e $$\\delta\_{ik}$$ assume o valor 1 se $k=i$ e 0 se $k \\neq i$. A aplicação dessa condição resulta nos sistemas de equações lineares definidos por:

$$A\\beta\_{i} \= L\_{i}, \\quad para \\quad i=1, 2, 3 $$

A matriz de interpolação $$A$$, os vetores de coeficientes locais $$\\beta\_{i}$$ e os vetores canônicos $$L\_{i}$$ tomam a seguinte forma matricial:

$$ A \= \\begin{bmatrix} t\_{1x} & t\_{1y} & y\_{1}t\_{1x} \- x\_{1}t\_{1y} \\\\  
			 t\_{2x} & t\_{2y} & y\_{2}t\_{2x} \- x\_{2}t\_{2y} \\\\  
			 t\_{3x} & t\_{3y} & y\_{3}t\_{3x} \- x\_{3}t\_{3y} \\end{bmatrix} $$

$$ \\beta\_{i} \= \\begin{bmatrix} \\beta\_{1i} \\\\  
				 \\beta\_{2i} \\\\  
				 \\beta\_{3i} \\end{bmatrix} $$

$$ L\_{1} \= \\begin{bmatrix} 1 \\  
0 \\  
0 \\end{bmatrix}, \\quad L\_{2} \= \\begin{bmatrix} 0 \\  
1 \\  
0 \\end{bmatrix}, \\quad L\_{3} \= \\begin{bmatrix} 0 \\  
0 \\ 
1 \\end{bmatrix} $$

Nestes sistemas, $$t\_{kx}$$ e $$t\_{ky}$$ correspondem às componentes cartesianas do vetor unitário de direção atrelado ao $$k$$-ésimo nó de suporte. 

Uma vez determinados os coeficientes $$\\beta\_{i} , i=1,2,3$$, as funções de forma $$\\vec{N}\_{i}$$ estarão determinadas para o domínio de suporte e a aproximação de uma função vetorial $$\\vec{E}$$ no domínio de suporte é dada por:

$$\\vec{E}^h=\\sum\_{i=1}^{3} \\vec{N}\_{i}e\_{i} \= \\Phi(x,y)e\_{s} $$

onde $$e\_s$$ é um vetor com as projeções de $$\\vec{E}$$ na direção de cada vetor unitário $$\\vec{t}\_i$$ e $$\\Phi(x,y)$$ é a matriz de funções de forma:

$$ \\Phi(x,y) \= \\begin{bmatrix} \\vec{N}\_{1} & \\vec{N}\_{2} & \\vec{N}\_{3} \\end{bmatrix} \\text{ e } e\_{s} \= \\begin{bmatrix} e\_{1} \\\\  
e\_{2} \\\\  
e\_{3} \\end{bmatrix} $$

O rotacional da aproximação, $$\\nabla \\times \\vec{E}^h$$ é dado por

$$\\nabla \\times \\vec{E}^h= \\begin{bmatrix} \\nabla \\times \\vec{N}\_{1} & \\nabla \\times \\vec{N}\_{2} & \\nabla \\times \\vec{N}\_{3} \\end{bmatrix} \\begin{bmatrix} e\_{1} \\\\  
e\_{2} \\\\  
e\_{3} \\end{bmatrix}  $$

com

$$\\nabla \\times \\vec{N}\_i \= \\beta\_{1i} \\nabla \\times \\begin{bmatrix} 1 \\\\  
0 \\end{bmatrix} \+ \\beta\_{2i} \\nabla \\times \\begin{bmatrix} 0 \\\\  
1 \\end{bmatrix} \+ \\beta\_{3i}\\nabla \\times \\begin{bmatrix} y \\\\  
\-x \\end{bmatrix} $$

Nota-se que o rotacional aplicado a vetores constantes é nulo e, portanto, ele é diferente de zero apenas para o último termo da equação, resultando em:

$$\\nabla \\times \\vec{N}\_i \= \\begin{bmatrix} 0 \\\\  
0  \\\\  
\-2\\beta\_{3i} \\end{bmatrix} $$

Logo, o rotacional da aproximação se reduz a: 

$$\\nabla \\times \\vec{E}^h \= \\begin{bmatrix} 0 \\\\  
0  \\\\  
\-2 \\sum\_{i=1}^{3} \\beta\_{3i}e\_{i}  \\end{bmatrix} $$

É fundamental garantir que as aproximações $$\\vec{E}^h$$ e $$\\nabla \\times \\vec{E}^h$$ sejam as melhores possíveis. Exploraremos, incrementalmente, as possibilidades nas seções seguintes. Essas possibilidades foram geradas a partir da descrição acima e da “discussão” com a IA (Google Gemini, modelo 3.1 Pro (Matemática e programação avançada), Estendido (Raciocínio Complexo \- solução de problemas complexos)).
